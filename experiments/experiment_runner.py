import torch
import torch.nn as nn
import torch.profiler as profiler
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging
import random

import pandas as pd

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, Callback
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.loggers import WandbLogger as LightningWandbLogger
from transformers import set_seed

from src.data.BAIT_dataset import BAITDataset, BAITDatasetConfig
from src.data.PADBench_dataset import PADBenchDataset, PADBenchDatasetConfig
from src.data.loader_base import ModelDataModule
from src.data.dataset_base import ModelDataset, ModelDatasetConfig, ActivationDataset, strDataset
from experiments.configs.input_ds_config import InputDSConfig
from src.BackdoorID.llm_model import LLMModel, LLMModelConfig, LLMModelConfigList
from src.BackdoorID.base import BackdoorID, BackdoorIDConfig, TestResult
from src.utils.better_logging import get_logger
from src.utils.metric_logger import ClearmlLogger
from src.utils.wandb_logger import WandbLogger
from src.utils.base_config import BaseConfig

from src.utils.timing import TimingManager

logger = get_logger(__name__, level=logging.INFO)


class TrainingConfig(BaseConfig):
    """
    Configuration for training parameters and reproducibility.

    Attributes:
        max_epochs (int): Maximum number of training epochs. 
        val_ratio (float): Ratio of training data to use for validation (if no validation set is provided). 
        num_workers (int): Number of worker processes for data loading. 
        accelerator (str): Accelerator type (e.g., "auto", "gpu", "cpu").
        log_every_n_steps (int): Frequency of logging training metrics. 
        enable_progress_bar (bool): Whether to show the training progress bar.
        deterministic (bool): If True, enables deterministic cuda algorithms. 
        seed (Optional[int]): Random seed for global reproducibility. If None, a random one is sampled.
        dataloader_seed (Optional[int]): Seed specifically for dataloader shuffling. Defaults to `seed`.
    """
    max_epochs: int = 10 
    val_ratio: float = 0.2
    num_workers: int = 0
    accelerator: str = "auto"
    log_every_n_steps: int = 1
    enable_progress_bar: bool = True
    deterministic: bool = False
    seed: Optional[int] = None
    dataloader_seed: Optional[int] = None
    
    def model_post_init(self, __context: Any) -> None:
        """Initialize seed values if not provided."""
        # If seed is None, sample a random one
        if self.seed is None:
            self.seed = random.randint(0, 2**32 - 1)
        
        # If dataloader_seed is None, set it to the seed parameter
        if self.dataloader_seed is None:
            self.dataloader_seed = self.seed


class CheckpointConfig(BaseConfig):
    """
    Configuration for model checkpointing and early stopping.

    Attributes:
        save_top_k (int): Number of best checkpoints to save. Defaults to 1.
        monitor (str): Metric name to monitor for checkpointing/stopping. Defaults to "loss/val".
        mode (str): One of {"min", "max"} to determine if metric should be minimized/maximized. Defaults to "min".
        early_stopping_patience (int): Number of epochs to wait before stopping training if no improvement. 
        save_last (bool): Whether to save the last checkpoint of training. Defaults to False.
    """
    save_top_k: int = 1
    monitor: str = "loss/val"
    mode: str = "min"
    early_stopping_patience: int = 5
    save_last: bool = False


class LoggingConfig(BaseConfig):
    """
    Configuration for experiment logging and tracking.

    Attributes:
        experiment_name (str): Unique identifier for the current experiment run.
        project_name (str): Name of the project for logging grouping. 
        log_dir (str): Local directory path to store logs and weights. Defaults to "logs".
        logger (str): Logger platform to use, either "clearml" or "wandb". Defaults to "wandb".
        disabled (bool): If True, disables all external logging. Defaults to False.
    """
    experiment_name: str
    project_name: str = "LLM_ULP"
    log_dir: str = "logs"
    logger: str = "wandb"  # Options: "clearml", "wandb"
    disabled: bool = False
    
    def model_post_init(self, __context: Any) -> None:
        """Add timestamp to experiment name if not already present."""
        # if not any(char.isdigit() for char in self.experiment_name[-10:]):
        date_str = datetime.now().strftime("%m_%d_%H_%M_%S")
        self.experiment_name = f"{self.experiment_name}_{date_str}"
        
        # Validate logger choice
        valid_loggers = {"clearml", "wandb"}
        if self.logger not in valid_loggers:
            raise ValueError(f"Invalid logger '{self.logger}'. Must be one of {valid_loggers}")


class ClassifierConfig(BaseConfig):
    """
    Configuration for the classifier model architecture.

    Attributes:
        classifier_class (type[nn.Module]): The class of the neural network to instantiate.
        classifier_config (BaseConfig): Configuration object specific to the classifier_class.
    """
    classifier_class: type[nn.Module]
    classifier_config: BaseConfig


class ExperimentRunner:
    """
    Orchestrates the complete experiment workflow for training and evaluating BackdoorDetector.
    
    This class provides a modular interface for:
    - Loading datasets from various sources (BAIT, PADBench, custom)
    - Managing base LLM models
    - Creating and configuring classifier networks
    - Training with PyTorch Lightning
    - Evaluating on multiple test datasets
    - Comprehensive logging and checkpointing
    
    Args:
        base_model_configs: LLMModelConfigList containing base model configurations
        detector_config: BackdoorDetectorConfig for the BackdoorDetector
        training_config: Training hyperparameters configuration
        logging_config: Logging and experiment tracking configuration
        checkpoint_config: Model checkpointing configuration
        classifier_config: classifier architecture configuration
        device: Device to run models on (defaults to auto-detect)
    """
    
    def __init__(
        self,
        base_model_configs: LLMModelConfigList,
        detector_config: BackdoorIDConfig,
        training_config: TrainingConfig,
        logging_config: LoggingConfig,
        checkpoint_config: CheckpointConfig,
        classifier_config: ClassifierConfig,
        device: Optional[torch.device] = None,
        tags: Optional[List[str]] = None,
        additional_hparams: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        self.base_model_configs = base_model_configs
        self.detector_config = detector_config
        self.training_config = training_config
        self.logging_config = logging_config
        self.checkpoint_config = checkpoint_config
        self.classifier_config = classifier_config
        self.tags = tags if tags is not None else []
        self.additional_hparams = additional_hparams if additional_hparams is not None else {}
        
        # Auto-detect device if not specified
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() and training_config.accelerator != "cpu" 
                else "cpu"
            )
        else:
            self.device = device
            
        # Initialize components (to be created during run)
        self.base_models: Dict[str, LLMModel] = {}
        self.classifier: Optional[nn.Module] = None
        self.lightning_model: Optional[BackdoorID] = None
        self.trainer: Optional[L.Trainer] = None
        self.metric_logger: Optional[ClearmlLogger | WandbLogger] = None
        self.tb_logger: Optional[TensorBoardLogger] = None
        self.timing_manager = TimingManager.get_instance()
        
        # Verify logging directory does not exist
        self.verify_logging_dir_not_exists()
        
        # Initialize loggers
        self.initialize_logger()
    
    def verify_logging_dir_not_exists(self) -> None:
        """
        Verify that the logging directory does not already exist.
        
        Raises:
            FileExistsError: If the logging directory already exists
        """
        log_path = Path(self.logging_config.log_dir) / self.logging_config.experiment_name
        if log_path.exists():
            raise FileExistsError(
                f"Logging directory already exists: {log_path}\n"
                f"Please use a different experiment name to avoid overwriting existing logs."
            )
        
    def load_base_models(self, data_module: ModelDataModule) -> Dict[str, LLMModel]:
        """
        Load all base LLM models specified in the configuration and are present in the datasets.
        
        Returns:
            Dictionary mapping base model names to LLMModel instances
        """
        logger.info("-" * 80)
        logger.info("Loading Base LLM Models")
        logger.info("-" * 80)

        model_names = []
        datasets = [data_module.train_dataset]
        datasets.extend([data_module.val_dataset] if data_module.val_dataset is not None else [])
        datasets.extend([data_module.test_dataset] if data_module.test_dataset is not None else [])
        for dataset in datasets:
            if dataset is not None:
                model_names.extend(dataset.get_base_model_names())
        
        base_models = {}
        for model_name, model_config in self.base_model_configs.model_configs.items():
            if model_name not in model_names:
                logger.debug(f"Skipping base model '{model_name}' as it is not present in the datasets")
                continue
            logger.info(f"Loading model: {model_name}")
            base_models[model_name] = LLMModel(
                model_config=model_config,
                device=torch.device('cpu'),  # Load initially on CPU
            )
            logger.info(f"✓ Loaded: {model_config.model_name_or_path}")
        
        self.base_models = base_models
        logger.info(f"✓ Total base models loaded: {len(base_models)}")
        return base_models
    
    def load_dataset(
        self,
        dataset_config: ModelDatasetConfig,
    ) -> ModelDataset:
        """
        Load a dataset based on the provided configuration.
        
        Args:
            dataset_config: Configuration for the dataset (must implement load_dataset method)
            
        Returns:
            Loaded LoRADataset instance
        """
        logger.info("-" * 80)
        logger.info("Loading Dataset")
        logger.info("-" * 80)
        
        # Use the load_dataset method from the config
        dataset = dataset_config.load_dataset()
        
        # Log dataset type
        if isinstance(dataset_config, BAITDatasetConfig):
            logger.info(f"✓ Loaded BAIT dataset")
        elif isinstance(dataset_config, PADBenchDatasetConfig):
            logger.info(f"✓ Loaded PADBench dataset")
        else:
            logger.info(f"✓ Loaded dataset of type: {type(dataset).__name__}")
        
        logger.info(f"✓ Total samples: {len(dataset)}")
        # Count clean and poisoned samples
        if len(dataset) > 0:
            clean_count = sum(1 for sample in dataset.data if sample.label == '0' or sample.label == 0)
            poisoned_count = sum(1 for sample in dataset.data if sample.label == '1' or sample.label == 1)
            logger.info(f"  - Clean samples: {clean_count}")
            logger.info(f"  - Poisoned samples: {poisoned_count}")
        
        if len(dataset) == 0:
            logger.warning("⚠ No samples found! Check dataset filters.")
        
        return dataset
    
    def create_datamodule(
        self,
        train_dataset: ModelDataset,
        val_dataset: Optional[ModelDataset] = None,
        test_dataset: Optional[ModelDataset] = None,
    ) -> ModelDataModule:
        """
        Create a PyTorch Lightning DataModule for the datasets.
        
        Args:
            train_dataset: Training dataset (will be split into train/val if val_dataset is None)
            val_dataset: Optional validation dataset (if provided, no splitting is done)
            test_dataset: Optional separate test dataset
            
        Returns:
            Configured LoRADataModule instance
        """
        logger.info("-" * 80)
        logger.info("Creating DataModule")
        logger.info("-" * 80)
        
        datamodule = ModelDataModule(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            val_ratio=self.training_config.val_ratio,
            test_dataset=test_dataset,
            batch_size=self.detector_config.batch_size,
            num_workers=self.training_config.num_workers,
            seed=self.training_config.dataloader_seed,
        )
        
        logger.info(f"✓ Train samples: {len(datamodule.train_split)}")
        logger.info(f"✓ Val samples: {len(datamodule.val_split)}")
        if test_dataset is not None:
            logger.info(f"✓ Test samples: {len(test_dataset)}")
        
        return datamodule
    
    def create_classifier(self) -> nn.Module:
        """
        Create the classifier network based on configuration.
        
        Returns:
            Initialized classifier module
        """
        logger.info("-" * 80)
        logger.info("Creating Classifier Network")
        logger.info("-" * 80)
        
        # Instantiate the classifier using the class and config
        cls_class = self.classifier_config.classifier_class
        cls_config = self.classifier_config.classifier_config
        classifier = cls_class(cls_config)
        
        logger.info(f"✓ Classifier architecture:")
        logger.info(f"  Class: {self.classifier_config.classifier_class.__name__}")
        logger.info(f"  Config: {cls_config}")
        
        # Log parameter dtypes
        dtypes = {name: param.dtype for name, param in classifier.named_parameters()}
        unique_dtypes = set(dtypes.values())
        if len(unique_dtypes) == 1:
            logger.info(f"  Parameter dtype: {next(iter(unique_dtypes))}")
        else:
            logger.info(f"  Parameter dtypes (mixed):")
            for name, dtype in dtypes.items():
                logger.info(f"    {name}: {dtype}")
        
        self.classifier = classifier
        return classifier
    
    def configure_scheduler_kwargs(self, datamodule: ModelDataModule) -> None:
        """
        Configure scheduler kwargs based on scheduler type.
        For OneCycleLR, automatically calculates total_steps if not provided.
        
        Args:
            datamodule: DataModule to get training data information from
        """
        scheduler_type = self.detector_config.scheduler.lower()
        
        if scheduler_type == "onecycle":
            # Calculate total_steps if not already provided
            if "total_steps" not in self.detector_config.scheduler_kwargs:
                # Calculate steps per epoch
                train_dataloader = datamodule.train_dataloader()
                steps_per_epoch = len(train_dataloader)
                
                # Calculate total steps
                total_steps = self.training_config.max_epochs * steps_per_epoch
                
                # Update scheduler kwargs
                self.detector_config.scheduler_kwargs["total_steps"] = total_steps
                
                logger.info("-" * 80)
                logger.info("Configuring OneCycleLR Scheduler")
                logger.info("-" * 80)
                logger.info(f"✓ Calculated total_steps: {total_steps}")
                logger.info(f"  - Max epochs: {self.training_config.max_epochs}")
                logger.info(f"  - Steps per epoch: {steps_per_epoch}")
            else:
                logger.info("-" * 80)
                logger.info("OneCycleLR Scheduler")
                logger.info("-" * 80)
                logger.info(f"✓ Using provided total_steps: {self.detector_config.scheduler_kwargs['total_steps']}")
    
    def create_lightning_model(self, input_ds_config: Optional[InputDSConfig] = None) -> BackdoorID:
        """
        Create the BasicNoiseClassifier Lightning module.
        
        Args:
            input_ds_config: Optional InputDSConfig to build dataset and pass to BackdoorDetector
        
        Returns:
            Initialized BasicNoiseClassifier instance
        """
        logger.info("-" * 80)
        logger.info("Creating BackdoorDetector")
        logger.info("-" * 80)
        
        if self.classifier is None:
            raise RuntimeError("Classifier must be created before creating lightning model")
        
        # Build the input dataset if config is provided
        input_dataset = None
        if input_ds_config is not None:
            logger.info(f"✓ Built input dataset from config: {input_ds_config.name}")
            logger.info(f"Input dataset config: {input_ds_config}")
            input_dataset = input_ds_config.build_dataset()
            logger.info(f"  - Samples: {len(input_dataset)}")
        
        lightning_model = BackdoorID(
            base_models=self.base_models,
            classifier=self.classifier,
            config=self.detector_config,
            input_dataset=input_dataset,
        )
        lightning_model.to(self.device)
        
        logger.info(f"✓ BackdoorDetector created and moved to device: {self.device}")
        logger.info(f"  - Activation layers: {self.detector_config.classifier_activation_layers}")
        logger.info(f"  - Learning rate: {self.detector_config.optimizer_kwargs.get('lr', 'N/A')}")
        logger.info(f"  - Weight decay: {self.detector_config.optimizer_kwargs.get('weight_decay', 'N/A')}")
        logger.info(f"  - Samples per adapter: {self.detector_config.cls_inputs_per_sample}")
        
        self.lightning_model = lightning_model
        return lightning_model
    
    def sanity_check_forward(self, dataset: ModelDataset) -> None:
        """
        Perform a sanity check by loading a single sample and running forward pass.
        
        Args:
            dataset: Dataset to load a sample from
        """
        logger.info("-" * 80)
        logger.info("Sanity Check - Testing Forward Pass")
        logger.info("-" * 80)
        
        if self.lightning_model is None:
            raise RuntimeError("Lightning model must be created before sanity check")
        
        if len(dataset) == 0:
            logger.warning("⚠ Dataset is empty, skipping sanity check")
            return
        
        try:
            # Get a single sample from the dataset
            sample = dataset[0]
            logger.info(f"✓ Loaded sample. base_model: {sample.base_model} || Label: {sample.label}")
            
            # Set model to eval mode for sanity check
            self.lightning_model.eval()
            
            with torch.no_grad():
                output = self.lightning_model.forward(sample)
            
            logger.info(f"✓ Forward pass successful!")
            logger.info(f"  - Output shape: {output.shape if isinstance(output, torch.Tensor) else 'N/A'}")
            logger.info(f"  - Output type: {type(output)}")
            
            # Set model back to train mode
            self.lightning_model.train()

            # print model summary
            logger.info(f"\n[Model Summary]")
            logger.info(f"{self.lightning_model}")
            
        except Exception as e:
            logger.exception(f"✗ Sanity check failed: {e}")
            logger.error(f"  Sample: {sample.sample_path}")
            raise RuntimeError(f"Sanity check forward pass failed: {e}") from e
    
    def initialize_logger(self) -> None:
        """
        Initialize the metric logger and TensorBoard logger.
        
        This method creates the logger instances based on configuration
        but does not log any configurations yet.
        """
        # Initialize metric logger based on configuration
        if self.logging_config.logger == "clearml":
            self.metric_logger = ClearmlLogger(
                run_name=self.logging_config.experiment_name,
                project=self.logging_config.project_name,
                disabled=self.logging_config.disabled,
                auto_connect_frameworks={
                    'matplotlib': True, 
                    'tensorflow': False, 
                    'tensorboard': True,
                    'pytorch': False, 
                    'xgboost': False, 
                    'scikit': True, 
                    'fastai': True,
                    'lightgbm': False, 
                    'hydra': False, 
                    'detect_repository': True, 
                    'tfdefines': False,
                    'joblib': False, 
                    'megengine': False, 
                    'catboost': False, 
                    'gradio': False,
                },
            )
        elif self.logging_config.logger == "wandb":
            self.metric_logger = WandbLogger(
                run_name=self.logging_config.experiment_name,
                project=self.logging_config.project_name,
                root_dir=self.logging_config.log_dir,
                disabled=self.logging_config.disabled,
            )
        else:
            raise ValueError(f"Unsupported logger type: {self.logging_config.logger}")
        
        # Initialize TensorBoard logger (default for Lightning trainer)
        self.tb_logger = TensorBoardLogger(
            save_dir=self.logging_config.log_dir,
            name=self.logging_config.experiment_name,
            version=None,  # Auto-increment version
        )
        
        # Initialize Lightning WandbLogger that shares the same wandb.Run
        # This lets Lightning's self.log() calls write to W&B with correct global_step
        self.wandb_lightning_logger = None
        if (
            self.logging_config.logger == "wandb"
            and isinstance(self.metric_logger, WandbLogger)
            and self.metric_logger.wandb_run is not None
        ):
            self.wandb_lightning_logger = LightningWandbLogger(
                experiment=self.metric_logger.wandb_run,
            )
    
    def setup_logging(
        self,
        train_dataset_config: Optional[List[ModelDatasetConfig]] = None,
        val_dataset_config: Optional[List[ModelDatasetConfig]] = None,
        test_dataset_configs: Optional[Dict[str, ModelDatasetConfig]] = None,
        input_ds_config: Optional[InputDSConfig] = None,
    ) -> tuple[ClearmlLogger | WandbLogger, TensorBoardLogger]:
        """
        Setup experiment logging by logging all configurations and tags.
        
        Args:
            train_dataset_config: Configuration for training dataset
            val_dataset_config: Optional configuration for validation dataset
            test_dataset_configs: Optional dict of test dataset configurations
            input_ds_config: Optional InputDSConfig for input dataset
        
        Returns:
            Tuple of (MetricLogger, TensorBoard Logger)
        """
        logger.info("-" * 80)
        logger.info("Setting up Experiment Logging")
        logger.info("-" * 80)
        
        # Log all configurations to metric logger
        config_dict = {
            "base_model_configs": self.base_model_configs.model_dump(),
            "detector_config": self.detector_config.model_dump(),
            "training_config": self.training_config.model_dump(),
            "logging_config": self.logging_config.model_dump(),
            "checkpoint_config": self.checkpoint_config.model_dump(),
            "classifier_config": self.classifier_config.classifier_config.model_dump(),
        }
        if input_ds_config is not None:
            config_dict["input_ds_config"] = input_ds_config.model_dump()

        for config_name, config in config_dict.items():
            self.metric_logger.report_hparams(config_name, config)
        
        # Log additional hyperparameters
        for category, hparams in self.additional_hparams.items():
            self.metric_logger.report_hparams(category, hparams)
            logger.info(f"✓ Logged additional hyperparameters: {category}")
        
        # Log train dataset configuration
        if train_dataset_config is not None:
            train_configs_dict = {
                f"dataset_{i}": config.model_dump()
                for i, config in enumerate(train_dataset_config)
            }
            self.metric_logger.report_hparams("trainset_configs", train_configs_dict)
            logger.info(f"✓ Logged {len(train_dataset_config)} train dataset configuration(s)")
        
        # Log validation dataset configuration
        if val_dataset_config is not None:
            val_configs_dict = {
                f"dataset_{i}": config.model_dump()
                for i, config in enumerate(val_dataset_config)
            }
            self.metric_logger.report_hparams("valset_configs", val_configs_dict)
            logger.info(f"✓ Logged {len(val_dataset_config)} validation dataset configuration(s)")
        
        # Log test dataset configurations
        if test_dataset_configs is not None:
            test_configs_dict = {
                f"{name}": config.model_dump()
                for name, config in test_dataset_configs.items()
            }
            self.metric_logger.report_hparams("testset_configs", test_configs_dict)
            logger.info(f"✓ Logged {len(test_dataset_configs)} test dataset configuration(s)")

        # Add tags to the experiment
        tags = [
            # f"cls:{self.classifier_config.classifier_class.__name__}",
            # f"n_samples:{self.detector_config.noise_generation_config.n_samples}",
        ]
        # Add custom tags if provided
        tags.extend(self.tags)
        self.metric_logger.set_tags(*tags)
        
        logger.info(f"✓ TensorBoard logging to: {self.tb_logger.log_dir}")
        logger.info(f"✓ Metric logging with: {self.logging_config.logger}")
        
        return self.metric_logger, self.tb_logger
    
    def setup_callbacks(self) -> List[Callback]:
        """
        Create PyTorch Lightning callbacks for training.
        
        Returns:
            List of configured callbacks
        """
        logger.info("-" * 80)
        logger.info("Setting up Training Callbacks")
        logger.info("-" * 80)
        
        if self.tb_logger is None:
            raise RuntimeError("TensorBoard logger must be initialized before setting up callbacks")
        
        # Get log directory from TensorBoard logger
        log_dir = self.tb_logger.log_dir
        
        # Checkpoint callback - save best models based on validation metric
        checkpoint_callback = ModelCheckpoint(
            dirpath=Path(log_dir) / "checkpoints",
            filename="epoch={epoch:02d}-val_metric={" + self.checkpoint_config.monitor + ":.4f}",
            monitor=self.checkpoint_config.monitor,
            mode=self.checkpoint_config.mode,
            save_top_k=self.checkpoint_config.save_top_k,
            save_last=self.checkpoint_config.save_last,
            auto_insert_metric_name=False,
        )
        
        # Early stopping callback
        early_stopping_callback = EarlyStopping(
            monitor=self.checkpoint_config.monitor,
            patience=self.checkpoint_config.early_stopping_patience,
            mode=self.checkpoint_config.mode,
            verbose=True,
        )
        
        callbacks = [checkpoint_callback, early_stopping_callback]
        
        logger.info(f"✓ ModelCheckpoint:")
        logger.info(f"  - Monitoring: {self.checkpoint_config.monitor} ({self.checkpoint_config.mode})")
        logger.info(f"  - Saving top {self.checkpoint_config.save_top_k} models")
        logger.info(f"✓ EarlyStopping: patience={self.checkpoint_config.early_stopping_patience}")
        
        return callbacks
    
    def create_trainer(self, callbacks: Optional[List[Callback]] = None) -> L.Trainer:
        """
        Create PyTorch Lightning Trainer.
        
        Args:
            callbacks: Optional list of callbacks (will use setup_callbacks if None)
            
        Returns:
            Configured Lightning Trainer
        """
        logger.info("-" * 80)
        logger.info("Initializing PyTorch Lightning Trainer")
        logger.info("-" * 80)
        
        if callbacks is None:
            callbacks = self.setup_callbacks()
        
        if self.tb_logger is None:
            raise RuntimeError("TensorBoard logger must be initialized before creating trainer")
        
        # Build logger list: always include TensorBoard, optionally add W&B
        loggers = [self.tb_logger]
        if self.wandb_lightning_logger is not None:
            loggers.append(self.wandb_lightning_logger)
        
        trainer = L.Trainer(
            max_epochs=self.training_config.max_epochs,
            accelerator=self.training_config.accelerator,
            logger=loggers,
            callbacks=callbacks,
            log_every_n_steps=self.training_config.log_every_n_steps,
            enable_progress_bar=self.training_config.enable_progress_bar,
            enable_model_summary=True,
            deterministic=self.training_config.deterministic,
        )
        
        logger.info(f"✓ Trainer configured:")
        logger.info(f"  - Max epochs: {self.training_config.max_epochs}")
        logger.info(f"  - Accelerator: {self.training_config.accelerator}")
        
        self.trainer = trainer
        return trainer
    
    def train(self, datamodule: ModelDataModule) -> None:
        """
        Train the model using PyTorch Lightning.
        
        Args:
            datamodule: Configured data module for training
        """
        logger.info("=" * 80)
        logger.info("Starting Training")
        logger.info("=" * 80)
        
        if self.trainer is None:
            raise RuntimeError("Trainer must be created before training")
        
        if self.lightning_model is None:
            raise RuntimeError("Lightning model must be created before training")
        
        try:
            self.timing_manager.start("time/fit", sync_cuda=True, pre_sync_cuda=False)
            self.trainer.fit(self.lightning_model, datamodule=datamodule)
            fit_time = self.timing_manager.stop("time/fit")
            logger.info(f"\n✓ Training completed successfully! Time taken: {fit_time:.2f} seconds")
            if self.metric_logger is not None:
                # self.metric_logger.report_scalars({"time/fit": fit_time}, step=0)
                # self.tb_logger.log_metrics({"time/fit": fit_time}, step=0)
                self.metric_logger.report_globals({"time/fit": fit_time})
        except Exception as e:
            logger.error(f"\n✗ Training failed with error: {e}")
            fit_time = self.timing_manager.stop("time/fit")
            logger.info(f"✓ Training time until failure: {fit_time:.2f} seconds")
            if self.metric_logger is not None:
                self.metric_logger.close()
            raise
    
    def load_best_checkpoint(self) -> None:
        """
        Load the best model checkpoint from training.
        
        This method should be called after training and before evaluation
        to ensure evaluation is performed on the best model.
        """
        if self.trainer is None:
            raise RuntimeError("Trainer must be created before loading checkpoint")
        
        if self.lightning_model is None:
            raise RuntimeError("Lightning model must be created before loading checkpoint")
        
        # Find the checkpoint callback
        checkpoint_callback = None
        for callback in self.trainer.callbacks:
            if isinstance(callback, ModelCheckpoint):
                checkpoint_callback = callback
                break
        
        if checkpoint_callback is not None and checkpoint_callback.best_model_path:
            logger.info("-" * 80)
            logger.info("Loading Best Model Checkpoint")
            logger.info("-" * 80)
            logger.info(f"Loading from: {checkpoint_callback.best_model_path}")
            
            # Load the best checkpoint
            prev_config = self.lightning_model.hparams.copy()
            cache = self.lightning_model.dataset_cache
            
            loaded_model = BackdoorID.load_from_checkpoint(
                checkpoint_callback.best_model_path,
                map_location=self.device,
                # NOTE: Make sure to pass any required args for loading
                base_models=self.base_models,
                input_dataset=self.lightning_model.input_dataset,
            )
            self.lightning_model = loaded_model
            self.lightning_model.dataset_cache = cache  # Restore dataset cache
            logger.info(f"✓ Loaded best model (metric: {checkpoint_callback.best_model_score:.4f})")
            if self.lightning_model.hparams != prev_config:
                for key in prev_config:
                    if self.lightning_model.hparams[key] != prev_config[key]:
                        logger.error(f"  - Hyperparameter changed: {key}: {prev_config[key]} -> {self.lightning_model.hparams[key]}")
            self.lightning_model.to(self.device)
        else:
            logger.warning("No best model checkpoint found, using current model state")
    
    def evaluate(
        self,
        datamodule: Optional[ModelDataModule] = None,
        test_datasets: Optional[Dict[str, ModelDataset]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate the trained model.
        
        Args:
            datamodule: DataModule with validation set (will be added to test_datasets as "validation")
            test_datasets: Optional dictionary of test datasets {name: dataset}
                          to evaluate on multiple test sets        
        Returns:
            Dictionary containing evaluation results
        """
        logger.info("=" * 80)
        logger.info("Starting Evaluation")
        logger.info("=" * 80)
        
        if self.trainer is None:
            raise RuntimeError("Trainer must be created before evaluation")
        
        if self.lightning_model is None:
            raise RuntimeError("Lightning model must be created before evaluation")
        
        all_results = {}
        
        # Add validation set to test_datasets if datamodule provided
        if datamodule is not None:
            if test_datasets is None:
                test_datasets = {}
            # Convert Subset to LoRADataset by extracting the relevant samples
            val_subset = datamodule.val_split
            
            # Check if val_subset is a Subset (has indices) or a direct dataset
            if hasattr(val_subset, 'indices'):
                # It's a Subset - extract data using indices
                val_data = [val_subset.dataset.data[i] for i in val_subset.indices]
                val_dataset = ModelDataset(val_data)
                
                # Verify the datasets contain the same items
                assert len(val_dataset) == len(val_subset), f"Length mismatch: {len(val_dataset)} vs {len(val_subset)}"
                for i in range(len(val_dataset)):
                    assert val_dataset.data[i] == val_subset.dataset.data[val_subset.indices[i]], \
                        f"Data mismatch at index {i}. original: {val_subset.dataset.data[val_subset.indices[i]]}, new: {val_dataset.data[i]}"
            else:
                # It's already a dataset - use it directly
                val_dataset = val_subset
            
        # Make sure validation set is first
        # if test_datasets is not None and "validation" not in test_datasets:
        test_datasets = {"validation": val_dataset, **test_datasets}
        
        # Evaluate on additional test datasets
        if test_datasets is not None:
            for test_name, test_dataset in test_datasets.items():
                logger.info("-" * 80)
                logger.info(f"Evaluating on Test Set: {test_name}")
                logger.info("-" * 80)
                
                # Create datamodule for this test set
                test_datamodule = ModelDataModule(
                    train_dataset=test_dataset,  # Dummy, won't be used
                    val_ratio=0.0,  # No split needed
                    test_dataset=test_dataset,
                    batch_size=self.detector_config.batch_size,
                    num_workers=self.training_config.num_workers,
                    seed=self.training_config.dataloader_seed,
                )
                self.lightning_model._test_name = f"_{test_name}" # set test name for propper logging
                #NOTE: Reset cache so that test activations are computed fresh
                self.lightning_model.dataset_cache.clear()
                # Run test for this dataset
                test_results = self.trainer.test(self.lightning_model, datamodule=test_datamodule)
                
                # Retrieve TestResult objects for this test set
                test_result_objects = self.lightning_model.get_test_results()
                
                if test_results:
                    logger.info(f"\nTest Results ({test_name}):")
                    for key, value in test_results[0].items():
                        logger.info(f"  {key}: {value:.4f}")
                    
                    # Store both metrics and TestResult objects
                    all_results[test_name] = {
                        "metrics": test_results[0],
                        "test_results": test_result_objects,
                    }

                    # Log to ClearML
                    if self.metric_logger is not None:
                        self.metric_logger.report_globals({
                            f"final_{test_name}_{k}": v for k, v in test_results[0].items()
                        })
                        
        # Save all test results to CSV
        if test_datasets is not None and len(all_results) > 0:
            self.save_test_results_csv(all_results)
        
        return all_results

    def save_test_results_csv(self, all_results: Dict[str, Any]) -> None:
        """
        Save all test results to a CSV file.
        
        Args:
            all_results: Dictionary containing test results for each test set
        """
        if self.tb_logger is None:
            logger.warning("TensorBoard logger not available, cannot determine log directory")
            return
        
        logger.info("-" * 80)
        logger.info("Saving Test Results to CSV")
        logger.info("-" * 80)
        
        # Collect all test results into a list of dictionaries
        rows = []
        for test_name, result_dict in all_results.items():
            # Skip if this is just validation results (doesn't have test_results key)
            if isinstance(result_dict, dict) and "test_results" in result_dict:
                test_result_objects:list[TestResult] = result_dict["test_results"]
                
                for test_result in test_result_objects:
                    row = {
                        "test_name": test_name,
                        "sample_path": test_result.sample_path,
                        "pred": test_result.pred,
                        "label": test_result.label,
                        "time": test_result.time,
                        "input_generation_time": getattr(test_result, "input_generation_time", None),
                        "cache_used": test_result.cache_used,
                        "peft_adapter": test_result.peft_adapter,
                        # Convert logits tensor to list for CSV storage
                        "logits": test_result.logits.numpy().tolist(),
                    }
                    rows.append(row)
        
        if len(rows) == 0:
            logger.warning("No test results to save")
            return
        
        # Create DataFrame
        df = pd.DataFrame(rows)
        
        # Save to CSV in the logging directory
        log_dir = Path(self.tb_logger.log_dir)
        csv_path = log_dir / "test_results.csv"
        df.to_csv(csv_path, index=False)
        
        logger.info(f"✓ Saved {len(rows)} test results to: {csv_path}")
        logger.info(f"  - Test sets: {df['test_name'].unique().tolist()}")
        logger.info(f"  - Total samples: {len(df)}")
        logger.info(f"  - Columns: {df.columns.tolist()}")
        
        # Log summary statistics
        logger.info(f"\nSummary Statistics:")
        logger.info(f"  - Average time: {df['time'].mean():.4f}s")
        logger.info(f"  - Cache hit rate: {df['cache_used'].mean():.2%}")
        logger.info(f"  - Accuracy: {(df['pred'] == df['label']).mean():.2%}")
    
    
    def save_configs(
        self,
        train_dataset_config: Optional[List[ModelDatasetConfig]] = None,
        val_dataset_config: Optional[List[ModelDatasetConfig]] = None,
        test_dataset_configs: Optional[Dict[str, ModelDatasetConfig]] = None,
        input_ds_config: Optional[InputDSConfig] = None,
    ) -> None:
        """
        Save all experiment configurations to the log directory.

        Saves the following configs as JSON files:
        - base_model_configs         → l_l_m_model_config_list.json
        - detector_config            → backdoor_detector_config.json
        - training_config            → training_config.json
        - logging_config             → logging_config.json
        - checkpoint_config          → checkpoint_config.json
        - classifier_config          → classifier_config.json  (inner BaseConfig)
        - additional_hparams         → additional_hparams.json
        - train_dataset_config[i]    → train_dataset_config_<i>.json
        - val_dataset_config[i]      → val_dataset_config_<i>.json
        - test_dataset_configs[name] → test_dataset_config_<name>.json
        - input_ds_config            → input_ds_config.json
        """
        if self.tb_logger is None:
            logger.warning("TensorBoard logger not available, skipping config saving")
            return

        log_dir = Path(self.tb_logger.log_dir)
        logger.info("-" * 80)
        logger.info(f"Saving Experiment Configurations to: {log_dir}")
        logger.info("-" * 80)

        configs_to_save = [
            self.base_model_configs,
            self.detector_config,
            self.training_config,
            self.logging_config,
            self.checkpoint_config,
        ]

        for config in configs_to_save:
            try:
                config.save(log_dir, overwrite=True)
                logger.info(f"✓ Saved {type(config).__name__}")
            except Exception as e:
                logger.warning(f"⚠ Failed to save {type(config).__name__}: {e}")

        # ClassifierConfig contains a non-serializable `classifier_class` type field,
        # so save only the inner classifier_config (the actual BaseConfig).
        try:
            self.classifier_config.classifier_config.save(
                log_dir, file_name="classifier_config.json", overwrite=True
            )
            logger.info(
                f"✓ Saved classifier_config ({type(self.classifier_config.classifier_config).__name__})"
            )
        except Exception as e:
            logger.warning(f"⚠ Failed to save classifier_config: {e}")

        # Save additional hyperparameters supplied by the experiment caller.
        try:
            additional_hparams_path = log_dir / "additional_hparams.json"
            with open(additional_hparams_path, "w", encoding="utf-8") as f:
                json.dump(self.additional_hparams, f, indent=2, ensure_ascii=False, default=str)
            logger.info(f"✓ Saved additional_hparams to: {additional_hparams_path}")
        except Exception as e:
            logger.warning(f"⚠ Failed to save additional_hparams: {e}")

        # Save train dataset configs
        if train_dataset_config is not None:
            for i, config in enumerate(train_dataset_config):
                try:
                    config.save(log_dir, file_name=f"train_dataset_config_{i}.json", overwrite=True)
                    logger.info(f"✓ Saved train_dataset_config[{i}] ({type(config).__name__})")
                except Exception as e:
                    logger.warning(f"⚠ Failed to save train_dataset_config[{i}]: {e}")

        # Save validation dataset configs
        if val_dataset_config is not None:
            for i, config in enumerate(val_dataset_config):
                try:
                    config.save(log_dir, file_name=f"val_dataset_config_{i}.json", overwrite=True)
                    logger.info(f"✓ Saved val_dataset_config[{i}] ({type(config).__name__})")
                except Exception as e:
                    logger.warning(f"⚠ Failed to save val_dataset_config[{i}]: {e}")

        # Save test dataset configs
        if test_dataset_configs is not None:
            for name, config in test_dataset_configs.items():
                try:
                    config.save(log_dir, file_name=f"test_dataset_config_{name}.json", overwrite=True)
                    logger.info(f"✓ Saved test_dataset_config[{name!r}] ({type(config).__name__})")
                except Exception as e:
                    logger.warning(f"⚠ Failed to save test_dataset_config[{name!r}]: {e}")

        # Save input dataset config
        if input_ds_config is not None:
            try:
                input_ds_config.save(log_dir, file_name="input_ds_config.json", overwrite=True)
                logger.info(f"✓ Saved input_ds_config ({type(input_ds_config).__name__})")
            except Exception as e:
                logger.warning(f"⚠ Failed to save input_ds_config: {e}")

    def save_checkpoint_info(self) -> None:
        """Log checkpoint information to ClearML and console."""
        if self.trainer is None:
            logger.warning("Trainer not available, skipping checkpoint info")
            return
        
        # Get checkpoint callback
        checkpoint_callback = None
        for callback in self.trainer.callbacks:
            if isinstance(callback, ModelCheckpoint):
                checkpoint_callback = callback
                break
        
        if checkpoint_callback is None:
            logger.warning("No ModelCheckpoint callback found")
            return
        
        logger.info("=" * 80)
        logger.info("Checkpoint Information")
        logger.info("=" * 80)
        logger.info(f"Best model checkpoint: {checkpoint_callback.best_model_path}")
        logger.info(f"Best validation metric: {checkpoint_callback.best_model_score:.4f}")
        
        # Log to ClearML
        if self.metric_logger is not None:
            self.metric_logger.report_globals({
                "best_val_metric": float(checkpoint_callback.best_model_score),
            })
    
    def cleanup(self) -> None:
        """Close loggers and cleanup resources."""
        logger.info("=" * 80)
        logger.info("Cleanup")
        logger.info("=" * 80)
        
        if self.metric_logger is not None:
            self.metric_logger.close()
            logger.info(f"✓ {self.logging_config.logger} logger closed")
        
        # Log TensorBoard directory
        if self.tb_logger is not None:
            logger.info(f"✓ TensorBoard logs: {self.tb_logger.log_dir}")
    
    def run(
        self,
        train_dataset_config: List[ModelDatasetConfig],
        val_dataset_config: Optional[List[ModelDatasetConfig]] = None,
        test_dataset_configs: Optional[Dict[str, ModelDatasetConfig]] = None,
        input_ds_config: Optional[InputDSConfig] = None,
    ) -> Dict[str, Any]:
        """
        Run the complete experiment workflow.
        
        This method orchestrates the entire experiment:
        1. Load datasets
        2. Load base models
        3. Initialize classifier
        4. Initialize BasicNoiseClassifier
        5. Setup logging
        6. Create trainer
        7. Train model
        8. Load best checkpoint
        9. Evaluate model
        10. Save checkpoint info
        11. Cleanup
        
        Args:
            train_dataset_config: List of dataset configurations to concatenate for training
            val_dataset_config: Optional list of dataset configurations to concatenate for validation
            test_dataset_configs: Optional dict of {name: config} for additional test datasets
            input_ds_config: Optional InputDSConfig to build dataset and pass to BackdoorDetector
        
        Returns:
            Dictionary containing all evaluation results
        """
        # Initialize random seed for reproducibility
        logger.info("=" * 80)
        logger.info("Setting Random Seed")
        logger.info("=" * 80)
        logger.info(f"Seed: {self.training_config.seed}")
        logger.info(f"Dataloader seed: {self.training_config.dataloader_seed}")
        set_seed(self.training_config.seed, deterministic=self.training_config.deterministic)
        
        logger.info("=" * 80)
        logger.info("EXPERIMENT RUNNER - Starting Complete Workflow")
        logger.info("=" * 80)
        logger.info(f"Experiment: {self.logging_config.experiment_name}")
        logger.info(f"Project: {self.logging_config.project_name}")
        logger.info(f"Device: {self.device}")
        logger.info("=" * 80)
        
        try:
            # Step 1: Load training dataset(s)
            logger.info("-" * 80)
            logger.info("Loading Training Dataset")
            logger.info(f"Loading and concatenating {len(train_dataset_config)} datasets")
            train_dataset = None
            for i, config in enumerate(train_dataset_config):
                logger.info(f"Loading dataset {i+1}/{len(train_dataset_config)}")
                dataset = self.load_dataset(config)
                if train_dataset is None:
                    train_dataset = dataset
                else:
                    train_dataset = train_dataset + dataset
            logger.info(f"✓ Combined dataset total: {len(train_dataset)} samples")
            
            # Step 2: Load validation dataset(s) if provided
            val_dataset = None
            if val_dataset_config is not None:
                logger.info("-" * 80)
                logger.info("Loading Validation Dataset")
                logger.info(f"Loading and concatenating {len(val_dataset_config)} datasets")
                for i, config in enumerate(val_dataset_config):
                    logger.info(f"Loading dataset {i+1}/{len(val_dataset_config)}")
                    dataset = self.load_dataset(config)
                    if val_dataset is None:
                        val_dataset = dataset
                    else:
                        val_dataset = val_dataset + dataset
                logger.info(f"✓ Combined dataset total: {len(val_dataset)} samples")
            
            # Step 3: Load test datasets (if any)
            test_datasets = None
            if test_dataset_configs is not None:
                test_datasets = {}
                for name, config in test_dataset_configs.items():
                    logger.info("-" * 80)
                    logger.info(f"Loading test dataset: {name}")
                    test_datasets[name] = self.load_dataset(config)
            
            # Step 4: Create datamodule
            datamodule = self.create_datamodule(train_dataset, val_dataset=val_dataset)
            
            # Step 5: Configure scheduler kwargs (e.g., calculate total_steps for OneCycleLR)
            self.configure_scheduler_kwargs(datamodule)
            
            # Step 6: Load base models
            self.load_base_models(data_module=datamodule)
            
            # Step 7: Create classifier
            self.create_classifier()
            
            # Step 8: Create lightning model
            self.create_lightning_model(input_ds_config=input_ds_config)
            
            # Step 8.5: Sanity check - test forward pass with a single sample
            self.sanity_check_forward(train_dataset)
            
            # Step 9: Setup logging
            self.setup_logging(
                train_dataset_config=train_dataset_config,
                val_dataset_config=val_dataset_config,
                test_dataset_configs=test_dataset_configs,
                input_ds_config=input_ds_config,
            )
            
            # Step 10: Create trainer
            self.create_trainer()
            
            # Step 11: Train
            self.train(datamodule)
            
            # Step 12: Load best checkpoint
            self.load_best_checkpoint()
            
            # Step 13: Evaluate
            results = self.evaluate(datamodule=datamodule, test_datasets=test_datasets)
            
            # Step 14: Save checkpoint info
            self.save_checkpoint_info()

            # Step 15: Save all experiment configurations
            self.save_configs(
                train_dataset_config=train_dataset_config,
                val_dataset_config=val_dataset_config,
                test_dataset_configs=test_dataset_configs,
                input_ds_config=input_ds_config,
            )

            # Step 16: Cleanup
            self.cleanup()
            
            logger.info("=" * 80)
            logger.info("EXPERIMENT COMPLETED SUCCESSFULLY!")
            logger.info("=" * 80)
            
            return results
            
        except Exception as e:
            logger.error(f"Experiment failed with error: {e}")
            self.cleanup()
            raise
