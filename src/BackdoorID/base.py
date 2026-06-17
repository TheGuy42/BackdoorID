"""
Base PyTorch Lightning Model Template for LLM ULP Training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import gc
import lightning as L
from typing import Any, Dict, Optional, Tuple, NamedTuple, Callable
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import random
import string
import logging
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from torch.utils.data import Dataset, DataLoader
from peft import PeftModel, PeftConfig, load_peft_weights
import os

from src.data.dataset_base import ModelDataset, ModelSample, strDataset, ActivationDataset
from src.BackdoorID.llm_model import LLMModel, LLMModelConfig
from src.BackdoorID.classifiers import ClassifierBase

from src.utils.hf_utils import synchronize_special_tokens
from src.utils.better_logging import get_logger
from src.utils.metric_logger import ClearmlLogger
from src.utils.hooks import HookManager, HookConfig
from src.utils.activation_extractor import ActivationExtractor
from src.utils.base_config import BaseConfig
from src.utils.FunctionSwapper import VarSwapper, FunctionSwapper
from src.utils.timing import TimingManager, EventTimer

logger = get_logger(__name__, level=logging.DEBUG)


class TestResult(BaseConfig):
    """
    Class to hold test results for a single sample.
    
    Automatically detaches tensors and moves them to CPU to prevent high GPU memory usage.

    Attributes:
        logits (torch.Tensor): Output logits from the classifier.
        pred (int): Predicted class index.
        label (int): Ground truth class index.
        time (float): Total processing time for the sample.
        input_generation_time (float): Time spent generating inputs for activation extraction.
        cache_used (bool): Whether results were retrieved from cache.
        sample_path (str): Path to the Model \ Adapter being tested.
        base_model (str): Name or path of the base model used.
    """
    model_config = {"arbitrary_types_allowed": True}
    
    logits: torch.Tensor
    pred: int
    label: int
    time: float
    input_generation_time: float
    cache_used: bool
    sample_path: str
    base_model: str
    peft_adapter: bool
    
    def __init__(self, **data):
        # Detach and move tensors to CPU before storing
        if 'logits' in data and isinstance(data['logits'], torch.Tensor):
            data['logits'] = data['logits'].detach().cpu()
        if 'pred' in data and isinstance(data['pred'], torch.Tensor):
            data['pred'] = int(data['pred'].item())
        if 'label' in data and isinstance(data['label'], torch.Tensor):
            data['label'] = int(data['label'].item())
        
        super().__init__(**data)


def clear_gpu_memory():
    """Clear GPU memory cache if CUDA is available."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()


class InputGenerationConfig(BaseConfig):
    """
    Configuration for input text generation.

    Attributes:
        n_samples (int): Number of inputs to generate/use per target model. Defaults to 100.
        generation_kwargs (dict): Keyword arguments for transformer generation (e.g., do_sample).
        tokenizer_kwargs (dict): Keyword arguments for the tokenizer.
        batch_size (int): Batch size for text generation. Defaults to 10.
    """
    n_samples: int = 100
    generation_kwargs: dict = {
        "do_sample": True,
        "min_new_tokens": 5,
        "max_new_tokens": 100,
    }
    tokenizer_kwargs: dict = {
        "max_length": 100,
    }
    batch_size: int = 10
    
class BackdoorIDConfig(BaseConfig):
    """
    Main configuration for the BackdoorID detection framework.

    Attributes:
        classifier_activation_layers (dict[str, list[str]]): Mapping of model types to layer names for activation extraction.
        loss_fn_reduction (str): Reduction method for the loss function ('mean' or 'sum'). Defaults to 'mean'.
        input_generation_config (InputGenerationConfig): Configuration for generating inputs.
        cache (bool): Whether to cache generated activation datasets. Defaults to True.
        cache_reset_interval (int): Number of samples after which to reset the cache. -1 means no reset.
        optimizer_kwargs (dict): Keyword arguments for the optimizer (e.g., lr, weight_decay).
        scheduler (str): Name of the learning rate scheduler. Defaults to "constant".
        scheduler_kwargs (dict): Additional parameters for the scheduler.
        batch_size (int): Global batch (number of target models) size for training. Defaults to 8.
        input_ds_batch_size (int): Batch size for processing the input dataset by target models. Defaults to 10.
        classifier_batch_size (int): Batch size for the classifier. Defaults to 16.
        cls_inputs_per_sample (int): Deprecated, keep at 1.
        dynamic_input_length (bool): Whether to apply the dynamic input extraction strategy. Defaults to False.
    """
    classifier_activation_layers: dict[str, list[str]] = {}
    loss_fn_reduction:str = 'mean'
    input_generation_config: InputGenerationConfig = InputGenerationConfig()
    cache: bool = True
    cache_reset_interval: int = -1  # Number of samples after which to reset cache (-1 means no reset)
    optimizer_kwargs: dict = {
        "lr": 1e-3,
        "weight_decay": 1e-5,
    }
    scheduler: str = "constant"  # Options: "constant", "onecycle", "cosine", "plateau"
    scheduler_kwargs: dict = {}  # Scheduler-specific parameters
    batch_size: int = 8
    input_ds_batch_size: int = 10
    classifier_batch_size: int = 16
    cls_inputs_per_sample: int = 1
    dynamic_input_length: bool = False

dummy_str_dataset = strDataset(["This is a dummy sample."]*20)


class BackdoorID(L.LightningModule):
    """
    Base Lightning Module template for LLM training and fine-tuning.

    This template provides the structure for:
    - Model initialization
    - Forward pass
    - Training, validation, and test steps
    - Optimizer and scheduler configuration
    - Metric logging
    """

    def __init__(
        self,
        base_models: dict[str, LLMModel],
        classifier: ClassifierBase,
        config: BackdoorIDConfig,
        input_dataset: Optional[strDataset] = None,
        **kwargs,
    ):
        """
        Initialize the Lightning module.

        Args:
            base_models: Dictionary mapping base model names to LLMModel instances
            classifier: Neural network classifier module
            config: Configuration object for the model
            input_dataset: Optional dataset for input samples; if None, noise dataset will be generated
        """
        super().__init__()
        self.save_hyperparameters(ignore=['base_models', 'input_dataset'])

        self.config = config
        self.input_dataset = input_dataset

        self.loss_fn = nn.BCEWithLogitsLoss(reduction=self.config.loss_fn_reduction) 

        self.llms: dict[str, LLMModel] = base_models
        self.classifier: ClassifierBase = classifier
        
        # Dataset cache mapping sample_path to ActivationDataset
        self.dataset_cache: dict[str, ActivationDataset] = {}

        self.timing_manager = TimingManager.get_instance()
        self._test_name = ""

    @torch.no_grad()
    def generate_noise_dataset(
        self,
        model: LLMModel,
        noise_config: InputGenerationConfig,
    ) -> strDataset:
        """
        Generate a noise dataset using the provided model.

        Args:
            model: LLMModel instance to use for noise generation
            n_samples: Number of noise samples to generate
            generation_kwargs: Additional generation parameters

        Returns:
            Dataset containing generated noise samples
        """
        n_samples = noise_config.n_samples
        generation_kwargs = noise_config.generation_kwargs
        batch_size = noise_config.batch_size

        generated_samples = []
        inputs = []
        ## Vocab - all non-special tokenizer tokens
        vocab_items = sorted(model.tokenizer.get_vocab().items(), key=lambda item: item[1])
        special_tokens = set(model.tokenizer.all_special_tokens)
        tokens = [token for token, _ in vocab_items if token not in special_tokens]
        
        # Generate n_samples of noise
        # Randomly choose the first character (alphanumeric + common punctuation)
        inputs = random.choices(tokens, k=n_samples)
        
        n_batches = (n_samples + batch_size - 1) // batch_size
        with VarSwapper((model.model_config, 'apply_chat_template', False)):
            for i in tqdm(range(n_batches), desc="Generating noise samples", total=n_batches, leave=False):
                ## Get the first character for this batch
                batch_inputs = inputs[i * batch_size:(i + 1) * batch_size]
                ## Process the first character through the model's tokenizer
                processed_input = model.process(batch_inputs, tokenizer_kwargs=noise_config.tokenizer_kwargs)

                # Generate text using the model
                generated_ids = model.generate(input_ids=processed_input["input_ids"], attention_mask=processed_input.get("attention_mask"), **generation_kwargs)

                # Decode the generated tokens to text
                generated_text = model.decode(generated_ids, skip_special_tokens=True)

                if isinstance(generated_text, str):
                    generated_text = [generated_text]

                generated_samples.extend(generated_text)
               
            # Clear GPU cache after generation loop
            clear_gpu_memory()

        # Return a dataset of the generated strings
        return strDataset(generated_samples)

    def prepare_dataset(self, model:LLMModel) -> strDataset:
        """
        Retrieve the input dataset if available; otherwise, generate a noise dataset.
        make sure to call this function after model preparation.

        Args:
            model: LLMModel instance to use for dataset generation
        Returns:
            Input dataset
        """
        input_dataset = self.input_dataset
        if input_dataset is None:
            # Generate noise dataset using one of the models
            training = model.training
            model.eval()
            input_dataset = self.generate_noise_dataset(model, noise_config=self.config.input_generation_config)
            if training:
                logger.warning("Model was in training mode before dataset preparation. Restoring to training mode.")
                model.train()
        
        return input_dataset

    def get_activation_dataset(self, sample_path: str) -> Optional[ActivationDataset]:
        """
        Retrieve cached activation dataset for a given adapter path.

        Args:
            model_path: Path to the model sample (used as cache key)
        Returns:
            Cached ActivationDataset if available, None otherwise
        """
        if self.config.cache and sample_path in self.dataset_cache:
            # logger.debug(f"Using cached activation dataset for adapter: {model_path}")
            return self.dataset_cache[sample_path]
        return None

    def get_model(self, base_model_name: str) -> LLMModel:
        """
        Retrieve the LLMModel instance for a given base model name.

        Args:
            base_model_name: Name of the base model
        Returns:
            LLMModel instance
        """
        if base_model_name not in self.llms:
            raise ValueError(f"Base model '{base_model_name}' not found in the model dictionary.")
        return self.llms[base_model_name]

    def offload_models(self, exclude_model_name: Optional[str] = None):
        """
        Offload all models to CPU except for the specified model.

        Args:
            exclude_model_name: Name of the model to keep on the current device (optional)
        """
        for model_name, model in self.llms.items():
            if exclude_model_name is None or model_name != exclude_model_name:
                if model.device != torch.device('cpu'):
                    # logger.debug(f"Offloading model '{model_name}' to CPU.")
                    model.to(torch.device('cpu'))
        clear_gpu_memory()  # Clear GPU memory after offloading

    def prepare_model(self, base_model_name: str) -> LLMModel:
        """
        Prepare the LLMModel instance for training or evaluation.
        If the model is not on the correct device, move it (first removing other models on the device).
        Finally, disable gradients for all model parameters.
        Args:
            base_model_name: Name of the base model
        Returns:
            LLMModel instance
        """
        self.offload_models(exclude_model_name=base_model_name)  # Offload other models to CPU

        model = self.get_model(base_model_name)
        model.eval()

        # disable grad for all model parameters
        for param in model.parameters():
            param.requires_grad = False

        if model.device != self.device:
            model.to(self.device)
        return model

    def forward(
        self,
        sample: ModelSample,
        return_test_result: bool = False,
        **kwargs,
    ) -> torch.Tensor | Tuple[torch.Tensor, TestResult]:
        """
        Forward pass through the model.

        Args:
            sample: Input sample containing adapter weights and other data
            return_test_result: If True, return TestResult in addition to logits
            **kwargs: Additional keyword arguments

        Returns:
            Tensor containing model outputs, or tuple of (logits, TestResult) if return_test_result=True
        """
        if return_test_result:
            # Start timing
            self.timing_manager.start("forward_timer", sync_cuda=True, pre_sync_cuda=False)
        
        input_generation_time = 0.0
        # Check if activation dataset is cached
        cache_used = False
        activation_dataset = self.get_activation_dataset(sample.sample_path)
        
        if activation_dataset is not None:
            cache_used = True
            activation_dataset.move_to(self.device) # Move cached dataset to correct device
        else:
            if sample.peft_adapter is True:
                model = self.prepare_model(sample.base_model)
                model.remove_all_adapters()  # Ensure no adapter is loaded
                adapter_path_expanded = os.path.expanduser(sample.sample_path)
                config = PeftConfig.from_pretrained(adapter_path_expanded)
                state_dict = load_peft_weights(adapter_path_expanded, device="cpu")
                model.load_adapter_from_state_dict(
                    state_dict=state_dict,
                    adapter_config=config,
                    is_trainable=False,
                )
            else: # Loading Full Model 
                self.offload_models() # Offload all other models to CPU to free up memory
                model_config = LLMModelConfig(
                    model_name_or_path=sample.sample_path,
                    tokenizer_name_or_path=sample.sample_path,
                    apply_chat_template=True, # assume chat model   
                )
                model = LLMModel(model_config, device=self.device)
                # verify if there is a chat template
                try:
                    model.process(["Hello"], tokenizer_kwargs={"max_length": 10})
                except Exception as e:
                    logger.debug(f"Error processing input with model '{sample.base_model}': {e} with chat template. Using model without chat template.")
                    model.model_config.apply_chat_template = False

            for param in model.parameters():
                param.requires_grad = False  # Ensure model parameters are not trainable
            model.eval()

            # Generate activation dataset
            # Each item in activations_batched is a dict[str, torch.Tensor]
            # where each tensor is of shape (cls_inputs_per_sample, act_dim)
            if return_test_result:
                self.timing_manager.start("input_generation_timer", sync_cuda=True, pre_sync_cuda=False)
            activation_dataset = self.generate_activation_dataset(model, label=sample.label, base_model_name=sample.base_model)
            if return_test_result:
                input_generation_time = self.timing_manager.stop("input_generation_timer")

            # model.to('cpu')  # Move model back to CPU to free up device memory
            if sample.peft_adapter is False:
                del model  # Delete full model to free up memory
            
            # Cache the activation dataset if caching is enabled
            if self.config.cache:
                self.dataset_cache[sample.sample_path] = activation_dataset
                # logger.debug(f"Cached activation dataset for adapter: {sample.sample_path}")

        # Each batch is a Dictionary mapping layer names to activation tensors. Each tensor shape: (batch_size, cls_inputs_per_sample, feature_dim)
        activation_dataloader = DataLoader(
            activation_dataset,
            batch_size=self.config.classifier_batch_size,
            shuffle=True,
        )

        # Pass aggregated activations through the classifier
        classifier_outputs = [] # List[torch.Tensor], each tensor of shape (batch_size, 2) (except possibly last)
        for act_batch, _ in activation_dataloader:
            logits = self.classifier(act_batch)
            classifier_outputs.append(logits)

        # Combine classifier outputs
        combined_logits = torch.cat(classifier_outputs, dim=0) # Shape: (total_samples, num_classes) or (total_samples, output_dim) if latent_aggregation
        agg_pred = self.classifier.prediction_head(combined_logits)  # Shape: (1, num_classes)
        
        # Move activation dataset to CPU to free up device memory
        del activation_dataloader
        if self.config.cache:
            activation_dataset.move_to(torch.device('cpu'))
        
        clear_gpu_memory()
        
        
        if return_test_result:
            # Stop timing and create test result if requested
            elapsed_time = self.timing_manager.stop("forward_timer")
            # Get prediction from logits (sigmoid threshold for single-logit binary classification)
            pred_class = (torch.sigmoid(agg_pred) > 0.5).long().item()
            
            test_result = TestResult(
                logits=agg_pred,
                pred=pred_class,
                label=sample.label,
                time=elapsed_time,
                input_generation_time=input_generation_time,
                cache_used=cache_used,
                sample_path=sample.sample_path,
                base_model=sample.base_model,
                peft_adapter=sample.peft_adapter,
            )
            return agg_pred, test_result
        
        return agg_pred

    @torch.no_grad()
    def generate_activation_dataset(self, model: LLMModel, label:int, base_model_name: str) -> ActivationDataset:
        """
        Generate an activation dataset for a given model.
        Args:
            model: LLMModel instance to use for dataset generation
            label: Label to assign to all samples in the activation dataset
            base_model_name: Name of the base model (used to get correct activation layers)
        Returns:
            ActivationDataset containing extracted activations
        """
        # Get activation layers for this specific base model
        if base_model_name not in self.config.classifier_activation_layers:
            raise ValueError(
                f"Base model '{base_model_name}' not found in classifier_activation_layers. "
                f"Available models: {list(self.config.classifier_activation_layers.keys())}"
            )
        
        layer_names = self.config.classifier_activation_layers[base_model_name]
        
        # Generate activation dataset
        input_dataset = self.prepare_dataset(model)
        noise_dataloader = DataLoader(input_dataset, batch_size=self.config.input_ds_batch_size, shuffle=True)

        #NOTE: activations shape (without transform):
            # - after forward:  (batch_size, seq_len, act_dim) 
            # - after generate: (batch_size, 1,       act_dim)
        
        token_act_ext = TokenActivationExtraction()

        activation_extractor = ActivationExtractor(
            model=model.llm,
            layer_names=layer_names,
            hook_type="post_forward",
            verbose=False,
            transform=token_act_ext,
            # device='cpu',
        )

        # Enable hooks for activation extraction
        with activation_extractor:
            # Process noise samples to extract activations
            for noise_batch in noise_dataloader:
                # Process the input batch through the model
                processed = model.process(noise_batch, tokenizer_kwargs=self.config.input_generation_config.tokenizer_kwargs)
                
                if self.config.dynamic_input_length and processed.get("attention_mask") is None:
                    logger.warning("No attention mask found in processed input and dynamic_input_length is enabled, defaulting to capturing last token activations.")
                if self.config.dynamic_input_length and processed.get("attention_mask") is not None:
                    token_act_ext.update_mask(processed.get("attention_mask"))
                
                outputs = model.forward(
                    input_ids=processed["input_ids"],
                    attention_mask=processed.get("attention_mask"),
                )

        # Retrieve activations
        activations = activation_extractor.get_activations()
        # Aggregate activations across all input samples
        activations_batched:list[dict[str, torch.Tensor]] = []
        for activation in activations:
            # Split activations into batches
            acts_split = self.split_to_samples(activation, self.config.cls_inputs_per_sample)
            activations_batched.extend(acts_split)
        
        # Create a Dataset from aggregated_activations
        # Each item in activations_batched is a dict[str, torch.Tensor]
        # where each tensor is of shape (cls_inputs_per_sample, act_dim)
        activation_dataset = ActivationDataset(activations_batched, label)
        return activation_dataset

    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the loss between predictions and targets.

        Args:
            predictions: Tensor of shape (batch_size, num_classes)
            targets: Tensor of shape (batch_size,)
        Returns:
            Loss tensor
        """
        assert predictions.dim() == 2, f"predictions must be 2D (batch_size, 1), got shape {tuple(predictions.shape)}"
        assert predictions.size(1) == 1, f"predictions must have 1 output logit (num_classes=1), got {predictions.size(1)}"
        assert targets.dim() == 1, f"targets must be 1D (batch_size,), got shape {tuple(targets.shape)}"
        assert predictions.size(0) == targets.size(0), f"batch size mismatch: predictions {predictions.size(0)} vs targets {targets.size(0)}"
        
        loss = self.loss_fn(predictions.squeeze(-1), targets.float())
        return loss

    def forward_batch(
        self,
        batch: list[ModelSample],
        return_test_result: bool = False,
        ) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, list[TestResult]]:
        """
        Forward pass for a batch of samples.
        Args:
            batch: List of ModelSample instances
            return_test_result: If True, return list of TestResult objects in addition to logits and labels
        Returns:
            Tuple of (logits tensor: (batch_size, num_classes), labels tensor: (batch_size,))
            or (logits, labels, list[TestResult]) if return_test_result=True
        """
        logits = []
        labels = []
        test_results = [] if return_test_result else None
        
        for sample in tqdm(batch, desc="Forward batch", leave=False, disable=False):
            # Forward pass - returns shape (1, num_classes) or (logits, TestResult)
            if return_test_result:
                pred_probs, test_result = self.forward(sample, return_test_result=True)
                test_results.append(test_result)
            else:
                pred_probs = self.forward(sample, return_test_result=False)
            
            # Create target tensor
            target = torch.tensor([sample.label], dtype=torch.long, device=self.device)

            logits.append(pred_probs)
            labels.append(target)

        # Stack logits and labels
        logits_tensor = torch.cat(logits, dim=0)  # Shape: (batch_size, num_classes)
        labels_tensor = torch.cat(labels, dim=0)  # Shape: (batch_size,)

        if return_test_result:
            return logits_tensor, labels_tensor, test_results
        return logits_tensor, labels_tensor

    def training_step(self, batch: list[ModelSample], batch_idx: int) -> torch.Tensor:
        """
        Training step for a single batch.

        Args:
            batch: List of ModelSample instances
            batch_idx: Index of the batch

        Returns:
            Loss tensor
        """
        num_samples = len(batch)
        logits_tensor, labels_tensor = self.forward_batch(batch)
        # Compute loss
        loss = self.compute_loss(logits_tensor, labels_tensor)
        
        # Log metrics
        self.log("loss/train", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=num_samples)
        
        # Log current learning rate
        current_lr = self.optimizers().param_groups[0]['lr']
        self.log("lr", current_lr, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
        
        return loss

    def validation_step(self, batch: list[ModelSample], batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Validation step for a single batch.

        Args:
            batch: List of ModelSample instances
            batch_idx: Index of the batch

        Returns:
            Dictionary containing validation outputs
        """
        num_samples = len(batch)
        
        logits_tensor, labels_tensor = self.forward_batch(batch)
        
        # Compute loss
        loss = self.compute_loss(logits_tensor, labels_tensor)
        
        # Compute accuracy (sigmoid threshold for single-logit binary classification)
        pred_classes = (torch.sigmoid(logits_tensor) > 0.5).long().squeeze(-1)
        correct = (pred_classes == labels_tensor).sum().item()
        accuracy = correct / num_samples
        
        # Log metrics
        self.log("loss/val", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=num_samples)
        self.log("accuracy/val", accuracy, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=num_samples)

        return {"val_loss": loss, "val_accuracy": accuracy}

    def test_step(self, batch: list[ModelSample], batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Test step for a single batch.

        Args:
            batch: List of ModelSample instances
            batch_idx: Index of the batch

        Returns:
            Dictionary containing test outputs including TestResult objects
        """
        num_samples = len(batch)
        
        # Get logits, labels, and test results
        logits_tensor, labels_tensor, test_results = self.forward_batch(batch, return_test_result=True)
        
        # Compute loss
        loss = self.compute_loss(logits_tensor, labels_tensor)
        
        # Log loss per batch (averaging is ok for loss)
        self.log("loss/test", loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=num_samples)

        # Return logits, labels, and test results for aggregation across all batches
        # DO NOT compute metrics per batch - they must be computed globally
        return_vals = {
            "test_loss": loss,
            "logits": logits_tensor,
            "labels": labels_tensor,
            "test_results": test_results,
        }
        return return_vals

    def configure_optimizers(self) -> Dict[str, Any]:
        """
        Configure optimizers and learning rate schedulers.

        Returns:
            Dictionary containing optimizer(s) and scheduler(s)
        """
        # Only train the classifier parameters
        if self.classifier is None:
            raise ValueError("Classifier is None. Cannot configure optimizer.")
        
        optimizer = torch.optim.AdamW(
            self.classifier.parameters(),
            **self.config.optimizer_kwargs,
        )
        
        # Return optimizer only if no scheduler is configured
        scheduler_type = self.config.scheduler.lower()
        if scheduler_type == "constant":
            return {"optimizer": optimizer}
        
        # Configure scheduler based on type
        scheduler = None
        scheduler_config = {}
        
        if scheduler_type == "onecycle":
            # OneCycleLR requires max_lr and total_steps/epochs
            from torch.optim.lr_scheduler import OneCycleLR
            
            # Get max_lr from optimizer_kwargs or scheduler_kwargs
            max_lr = self.config.scheduler_kwargs.get("max_lr", self.config.optimizer_kwargs.get("lr", 1e-3))
            
            # Determine total_steps or epochs
            if "total_steps" in self.config.scheduler_kwargs:
                total_steps = self.config.scheduler_kwargs["total_steps"]
            elif "epochs" in self.config.scheduler_kwargs and self.trainer is not None:
                # Calculate total steps from epochs
                epochs = self.config.scheduler_kwargs["epochs"]
                steps_per_epoch = len(self.trainer.datamodule.train_dataloader()) if self.trainer.datamodule else 1
                total_steps = epochs * steps_per_epoch
            else:
                raise ValueError("OneCycleLR requires 'total_steps' or 'epochs' in scheduler_kwargs")
            
            # Build kwargs for OneCycleLR
            onecycle_kwargs = {
                "max_lr": max_lr,
                "total_steps": total_steps,
            }
            # Add any additional kwargs from config
            for key in ["pct_start", "anneal_strategy", "div_factor", "final_div_factor"]:
                if key in self.config.scheduler_kwargs:
                    onecycle_kwargs[key] = self.config.scheduler_kwargs[key]
            
            scheduler = OneCycleLR(optimizer, **onecycle_kwargs)
            scheduler_config = {
                "scheduler": scheduler,
                "interval": "step",  # OneCycleLR updates per step
            }
            
        elif scheduler_type == "cosine":
            # CosineAnnealingLR
            from torch.optim.lr_scheduler import CosineAnnealingLR
            
            # Require T_max (number of epochs)
            if "T_max" not in self.config.scheduler_kwargs:
                raise ValueError("CosineAnnealingLR requires 'T_max' in scheduler_kwargs")
            
            cosine_kwargs = {
                "T_max": self.config.scheduler_kwargs["T_max"],
            }
            # Add optional parameters
            if "eta_min" in self.config.scheduler_kwargs:
                cosine_kwargs["eta_min"] = self.config.scheduler_kwargs["eta_min"]
            
            scheduler = CosineAnnealingLR(optimizer, **cosine_kwargs)
            scheduler_config = {
                "scheduler": scheduler,
                "interval": "epoch",  # CosineAnnealingLR typically updates per epoch
            }
            
        elif scheduler_type == "plateau":
            # ReduceLROnPlateau
            from torch.optim.lr_scheduler import ReduceLROnPlateau
            
            # Default parameters
            plateau_kwargs = {
                "mode": "min",
                "factor": 0.1,
                "patience": 10,
            }
            # Update with user-provided kwargs
            plateau_kwargs.update(self.config.scheduler_kwargs)
            
            scheduler = ReduceLROnPlateau(optimizer, **plateau_kwargs)
            scheduler_config = {
                "scheduler": scheduler,
                "monitor": plateau_kwargs.get("monitor", "loss/val"),  # Metric to monitor
                "interval": "epoch",
            }
            
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}. "
                           f"Supported types: 'constant', 'onecycle', 'cosine', 'plateau'")
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler_config,
        }

    def on_train_epoch_start(self):
        """Hook called at the start of each training epoch."""
        cache_reset_interval = self.config.cache_reset_interval
        if cache_reset_interval > 0 and self.current_epoch % cache_reset_interval == 0:
            logger.info(f"⚠ Resetting activation dataset cache at epoch {self.current_epoch}.")
            self.dataset_cache.clear()
            clear_gpu_memory()  # Clear GPU memory after cache reset
        
        self.timing_manager.start("time/train_epoch", sync_cuda=True, pre_sync_cuda=False)

    def on_train_epoch_end(self) -> None:
        """Hook called at the end of each training epoch."""
        epoch_time = self.timing_manager.stop("time/train_epoch")
        self.log("time/train_epoch", epoch_time, sync_dist=True, logger=True, on_epoch=True)
        # Clear GPU memory at end of each epoch
        clear_gpu_memory()

    def on_validation_epoch_end(self) -> None:
        """Hook called at the end of each validation epoch."""
        # Clear GPU memory at end of validation
        clear_gpu_memory()

    def on_test_epoch_end(self) -> None:
        """
        Hook called at the end of the test epoch.
        Aggregates predictions and labels from all test batches and computes global metrics.
        """
        # Gather all logits and labels from test_step outputs
        all_logits = []
        all_labels = []
        
        # Access the test step outputs stored by Lightning
        outputs = self.test_step_outputs if hasattr(self, 'test_step_outputs') else []
        
        if not outputs:
            logger.warning("No test step outputs found. Metrics will not be computed.")
            return
        
        for output in outputs:
            all_logits.append(output['logits'])
            all_labels.append(output['labels'])
        
        # Concatenate all logits and labels
        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        # Compute global classification metrics
        metrics = self.calc_classification_metrics(all_logits, all_labels)
        # Stop timer and log test epoch time
        test_time = self.timing_manager.stop("time/test_epoch")
        metrics[f'time'] = test_time
        
        # Log global metrics
        for metric_name, metric_value in metrics.items():
            self.log(f"{metric_name}/test{self._test_name}", metric_value, sync_dist=False, logger=True)
                
        # Clear the outputs for the next epoch
        if hasattr(self, 'test_step_outputs'):
            self.test_step_outputs.clear()
        
        # Clear GPU memory after test completion
        clear_gpu_memory()
    
    def get_test_results(self) -> list[TestResult]:
        """
        Retrieve all TestResult objects collected during testing.
        
        Returns:
            List of TestResult objects from all test samples
        """
        if hasattr(self, 'all_test_results'):
            return self.all_test_results
        return []

    def on_test_start(self) -> None:
        """Initialize storage for test step outputs."""
        self.test_step_outputs = []
        self.all_test_results = []
        self.timing_manager.start("time/test_epoch", sync_cuda=True, pre_sync_cuda=False)

    def on_test_batch_end(self, outputs, batch, batch_idx) -> None:
        """Store test step outputs for later aggregation."""
        if outputs is not None:
            self.test_step_outputs.append({
                'logits': outputs['logits'].detach().cpu(),
                'labels': outputs['labels'].detach().cpu()
            })
            # Store test results (already on CPU and detached)
            if 'test_results' in outputs:
                self.all_test_results.extend(outputs['test_results'])

    @staticmethod
    def split_to_samples(activation: dict[str, torch.Tensor], inputs_per_sample: int) -> list[dict[str, torch.Tensor]]:
        """
        Split activations dictionary into a list of smaller dictionaries each containing activations for a batch.

        Args:
            activation: Dictionary mapping layer names to activation tensors of shape (total_samples, seq_len, act_dim)
            inputs_per_sample: Number of inputs (activations) to be passed together as a single sample to the classifier. 
        Returns:
            List of dictionaries, each containing activations for a sample. Shape of each activation tensor in the dictionaries: (inputs_per_sample, act_dim)
        """
        # logger.debug(f"activation {activation}")
        is_tuple = False
        sample = next(iter(activation.values()))
        if isinstance(sample, tuple):
            # logger.warning("Activation values are tuples; using the first element of each tuple.")
            sample = sample[0]
            is_tuple = True

        total_samples, seq_len, act_dim = sample.size()
        assert total_samples % inputs_per_sample == 0, "Total samples must be divisible by batch size"

        num_samples = (total_samples // inputs_per_sample) * seq_len
        split_activations = [{} for _ in range(num_samples)]
        for layer_name, acts in activation.items():
            if is_tuple:
                acts = acts[0]
            assert acts.size(0) % inputs_per_sample == 0, "Total samples must be divisible by batch size"
            assert acts.size(0) == total_samples, "Total samples mismatch"
            assert acts.size(1) == seq_len, "Sequence length mismatch"

            acts = acts.view(total_samples * seq_len, act_dim)  # Shape: (total_samples * seq_len, act_dim)
            acts_split = acts.split(inputs_per_sample, dim=0)
            for i, split in enumerate(acts_split):
                # print(f"Layer {layer_name}, batch {i}: split shape {split.shape}")
                split_activations[i][layer_name] = split
                assert split.size(0) == inputs_per_sample, "Each split must have the correct batch size"

        valid_split_activations = []
        # Filter out any splits (activation samples) that contain NaN values.
        for split_dict in split_activations:
            has_nan = False
            for acts in split_dict.values():
                if torch.isnan(acts).any().item():
                    has_nan = True
                    break
            if not has_nan:
                valid_split_activations.append(split_dict)

        return valid_split_activations


    def calc_classification_metrics(self, logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        """
        Calculate classification metrics based on logits and true labels.

        Args:
            logits: Tensor of shape (num_samples, num_classes)
            labels: Tensor of shape (num_samples,)
        Returns:
            Dictionary containing calculated metrics:
            - tpr: True Positive Rate (Recall for positive class)
            - tnr: True Negative Rate (Recall for negative class)
            - fpr: False Positive Rate
            - fnr: False Negative Rate
            - accuracy: Overall accuracy
            - precision: Precision for positive class
            - recall: Recall for positive class (same as TPR)
            - f1: F1 score for positive class
        """
        assert logits.size(0) == labels.size(0), "Logits and labels must have the same number of samples"
        assert logits.size(1) == 1, "This function expects a single logit output (num_classes=1) for binary classification"
        assert labels.dim() == 1, "Labels tensor must be 1-dimensional"
        assert logits.dim() == 2, "Logits tensor must be 2-dimensional"
        # Get predicted classes (sigmoid threshold for single-logit binary classification)
        pred_classes = (torch.sigmoid(logits) > 0.5).long().squeeze(-1)
        
        # Convert to binary format (assuming binary classification with classes 0 and 1)
        # Positive class is 1, Negative class is 0
        true_positives = ((pred_classes == 1) & (labels == 1)).sum().item()
        true_negatives = ((pred_classes == 0) & (labels == 0)).sum().item()
        false_positives = ((pred_classes == 1) & (labels == 0)).sum().item()
        false_negatives = ((pred_classes == 0) & (labels == 1)).sum().item()
        
        total_samples = labels.size(0)
        total_positives = (labels == 1).sum().item()
        total_negatives = (labels == 0).sum().item()
        
        # Calculate metrics with safe division (avoid division by zero)
        eps = 1e-10
        
        tpr = true_positives / (total_positives + eps)  # True Positive Rate (Sensitivity/Recall)
        tnr = true_negatives / (total_negatives + eps)  # True Negative Rate (Specificity)
        fpr = false_positives / (total_negatives + eps)  # False Positive Rate
        fnr = false_negatives / (total_positives + eps)  # False Negative Rate
        
        accuracy = (true_positives + true_negatives) / total_samples
        precision = true_positives / (true_positives + false_positives + eps)
        recall = tpr  # Recall is the same as TPR
        f1 = 2 * (precision * recall) / (precision + recall + eps)
        
        return {
            "tpr": tpr,
            "tnr": tnr,
            "fpr": fpr,
            "fnr": fnr,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        

class TokenActivationExtraction:
    """
    Extract activations corresponding to specific token positions relative to the unpadded sequence.
    Stateful transform that uses an attention mask to identify valid sequences.
    Assumes sequences are left-padded.
    """
    def __init__(self):
        self.attention_mask = None

    def update_mask(self, attention_mask: torch.Tensor):
        self.attention_mask = attention_mask

    def __call__(self, activation: torch.Tensor) -> torch.Tensor:
        assert activation.dim() == 3, "Expected activation to be of shape (batch_size, seq_len, act_dim)"
        
        # Fallback if no mask is available
        if self.attention_mask is None:
            return activation[:, -1, :].unsqueeze(1)  # Default to last token if no mask is provided. Shape: (batch_size, 1, act_dim)
            
        batch_size, seq_len, act_dim = activation.size()
        
        # Count the number of valid tokens per sequence in the batch
        # Since it's left padded, all valid tokens are at the end of the sequence.
        tokens_per_sample = self.attention_mask.bool().sum(dim=1)  # Shape: (batch_size,)
        
        # Start with a mask full of False (indicating everything should be NaN)
        keep_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=activation.device)
        
        for b in range(batch_size):
            valid_count = tokens_per_sample[b].item()
            if valid_count == 0:
                continue
                
            start_idx = seq_len - valid_count
            
            # The indices we want, relative to the unpadded sequence (0-indexed)
            positions = []
            
            # Add powers of 2: 0, 1, 2, 4, 8...
            pos = 1
            while pos < valid_count:
                positions.append(pos-1)
                pos *= 2
                    
            # In addition to the last token activation
            if valid_count - 1 not in positions:
                positions.append(valid_count - 1)
                
            # Convert relative unpadded indices to absolute sequence indices
            absolute_indices = [start_idx + p for p in positions]
            
            # Mark these absolute indices to be Kept
            keep_mask[b, absolute_indices] = True
            
        # Broadcast keep_mask to match activation dimensions
        keep_mask_expanded = keep_mask
        # print(f"Keep mask for the batch: \n{keep_mask.int()}")
        
        # Keep selected activations, set the rest to NaN
        # Because we only use this via ActivationExtractor which passes a cloned tensor, mutating is safe
        activation[~keep_mask_expanded].fill_(float('nan'))
        
        return activation # Shape: (batch_size, seq_len, act_dim) with unwanted positions set to NaN