"""
Run experiments using ExperimentRunner with configuration loading from files.

This script loads experiment configurations from JSON files and allows
command-line overrides for key parameters.
"""

import argparse
import json
from pathlib import Path
import os
import importlib.util

# Add project root to path
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def _parse_optional_bool(value: str) -> bool:
    """Parse boolean CLI values robustly."""
    value_normalized = value.strip().lower()
    truthy = {"1", "true", "t", "yes", "y", "on"}
    falsy = {"0", "false", "f", "no", "n", "off"}
    if value_normalized in truthy:
        return True
    if value_normalized in falsy:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value '{value}'. Expected one of: true/false, yes/no, 1/0")


def _preconfigure_cuda_environment(argv: list[str]) -> None:
    """Configure CUDA environment variables before importing torch-dependent modules."""
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config_dir", type=str, default="experiments/configs")
    pre_parser.add_argument("--gpu", type=int, default=0)
    pre_args, _ = pre_parser.parse_known_args(argv)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(pre_args.gpu)

    training_cfg_path = Path(pre_args.config_dir) / "training_config.json"
    deterministic = False
    if training_cfg_path.exists():
        try:
            with open(training_cfg_path, "r", encoding="utf-8") as f:
                deterministic = bool(json.load(f).get("deterministic", False))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            deterministic = False

    if deterministic:
        # Must be set before torch import for deterministic CuBLAS behavior.
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def parse_args():
    """Parse command line arguments."""
    from src.BackdoorID.classifiers import CLASSIFIER_CONFIGS

    parser = argparse.ArgumentParser(description="Run experiment with config loading and overrides")

    # Config directory
    parser.add_argument("--config_dir", type=str, default="experiments/configs", help="Directory containing configuration JSON files")

    # Required classifier class argument
    parser.add_argument("--classifier_class", type=str, required=True, choices=list(CLASSIFIER_CONFIGS.keys()), help="Classifier class to use")

    # Required dataset name argument
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset configuration to use (must exist in EXP_DATASET_CONFIGS)")

    # Optional input dataset name argument
    parser.add_argument("--input_ds", type=str, default=None, help="Name of the input dataset configuration to use (must exist in INPUT_DS_CONFIGS). If None, no input dataset is used.")

    # GPU selection
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID to use (default: 0)")

    # Logging toggle
    parser.add_argument("--disable_logging", action="store_true", help="Disable logging")

    # Training overrides
    parser.add_argument("--epochs", type=int, default=None, help="Override max_epochs in training config")

    parser.add_argument("--batch_size", type=int, default=None, help="Override batch_size in classifier config")

    parser.add_argument("--lr", type=float, default=None, help="Override learning rate in optimizer config")

    parser.add_argument("--weight_decay", type=float, default=None, help="Override weight_decay in optimizer config")

    parser.add_argument("--n_samples", type=int, default=None, help="Override n_samples in input_generation_config")

    parser.add_argument("--max_new_tokens", type=int, default=None, help="Override max_new_tokens in input_generation_config")

    parser.add_argument("--max_length", type=int, default=None, help="Override max_length in input_generation_config.tokenizer_kwargs")

    parser.add_argument("--val_ratio", type=float, default=None, help="Override validation ratio in training config")

    parser.add_argument("--early_stopping_patience", type=int, default=None, help="Override early stopping patience in checkpoint config")

    parser.add_argument("--dynamic_input_length", type=_parse_optional_bool, default=None, help="Override dynamic_input_length in detector config")

    parser.add_argument("--effective_n_samples", action="store_true", help="If set with dynamic_input_length=True, convert n_samples to effective n_samples based on max_length")

    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility. If None, a random seed will be sampled.")

    parser.add_argument("--prefix", type=str, default=None, help="Prefix to prepend to the experiment name in logging config")

    parser.add_argument("--log_dir", type=str, default=None, help="Override log_dir in logging config")

    parser.add_argument("--project_name", type=str, default=None, help="Override project_name in logging config")
    
    parser.add_argument("--experiment_name", type=str, default=None, help="Override experiment_name in logging config")
    
    return parser.parse_args()


def load_and_override_configs(args):
    """Load all configurations from files and apply command-line overrides."""
    from experiments.experiment_runner import (
        TrainingConfig,
        CheckpointConfig,
        LoggingConfig,
        ClassifierConfig,
    )
    from src.BackdoorID.llm_model import LLMModelConfigList
    from src.BackdoorID.base import BackdoorIDConfig
    from src.BackdoorID.classifiers import CLASSIFIER_CONFIGS, CLASSIFIER_CLASSES
    from experiments.configs.input_ds_config import get_dataset

    import numpy as np

    def fix_n_samples(max_length: int, n_samples: int) -> int:
        n_samples_per_input = int(np.ceil(np.log2(max_length)) + 1)
        # assert n_samples % n_samples_per_input == 0, f"n_samples={n_samples}, max_length={max_length}:: Modulu={n_samples % n_samples_per_input}"
        if n_samples % n_samples_per_input != 0:
            print(f"n_samples={n_samples}, max_length={max_length}:: Modulu={n_samples % n_samples_per_input}")
        return round(n_samples / n_samples_per_input)

    config_dir = Path(args.config_dir)

    if not config_dir.exists():
        raise FileNotFoundError(f"Config directory not found: {config_dir}")

    # Load regular BackdoorDetectorConfig
    backdoor_detector_config = BackdoorIDConfig.load(dir_name=config_dir, do_raise=True)

    # Apply batch_size override
    if args.batch_size is not None:
        backdoor_detector_config.batch_size = args.batch_size

    # Apply lr and weight_decay overrides
    if args.lr is not None:
        backdoor_detector_config.optimizer_kwargs["lr"] = args.lr
    if args.weight_decay is not None:
        backdoor_detector_config.optimizer_kwargs["weight_decay"] = args.weight_decay

    # Apply noise generation config overrides
    if args.n_samples is not None:
        backdoor_detector_config.input_generation_config.n_samples = args.n_samples
    if args.max_new_tokens is not None:
        backdoor_detector_config.input_generation_config.generation_kwargs["max_new_tokens"] = args.max_new_tokens
    if args.max_length is not None:
        backdoor_detector_config.input_generation_config.tokenizer_kwargs["max_length"] = args.max_length
    if args.dynamic_input_length is not None:
        backdoor_detector_config.dynamic_input_length = args.dynamic_input_length
    if args.effective_n_samples and backdoor_detector_config.dynamic_input_length:
        max_length = backdoor_detector_config.input_generation_config.tokenizer_kwargs.get("max_length")
        if max_length is None:
            raise ValueError("Cannot compute effective n_samples: tokenizer_kwargs['max_length'] is missing")
        backdoor_detector_config.input_generation_config.n_samples = fix_n_samples(
            backdoor_detector_config.input_generation_config.tokenizer_kwargs["max_length"],
            backdoor_detector_config.input_generation_config.n_samples,
        )

    # Load the specific classifier config class and create ClassifierConfig
    classifier_config_class = CLASSIFIER_CONFIGS[args.classifier_class]
    specific_classifier_config = classifier_config_class.load(dir_name=config_dir, do_raise=True)
    # Get the classifier class
    classifier_class = CLASSIFIER_CLASSES[args.classifier_class]
    # Create the ClassifierConfig with classifier_class and classifier_config
    classifier_config = ClassifierConfig(classifier_class=classifier_class, classifier_config=specific_classifier_config)

    # Load training config
    training_config: TrainingConfig = TrainingConfig.load(dir_name=config_dir, do_raise=True)

    # Apply training overrides
    if args.epochs is not None:
        training_config.max_epochs = args.epochs
    if args.val_ratio is not None:
        training_config.val_ratio = args.val_ratio
    if args.seed is not None:
        training_config.seed = args.seed

    # Load checkpoint config
    checkpoint_config: CheckpointConfig = CheckpointConfig.load(dir_name=config_dir, do_raise=True)

    # Apply early stopping override
    if args.early_stopping_patience is not None:
        checkpoint_config.early_stopping_patience = args.early_stopping_patience

    # Load logging config
    logging_config: LoggingConfig = LoggingConfig.load(dir_name=config_dir, do_raise=True)
    if args.experiment_name is not None:
        logging_config.experiment_name = args.experiment_name
        logging_config.model_post_init(None)  # Update experiment name in logging config
    logging_config.experiment_name = f"{args.dataset_name}_{args.classifier_class}_{logging_config.experiment_name}"
    if args.prefix is not None:
        logging_config.experiment_name = f"{args.prefix}_{logging_config.experiment_name}"

    # Apply logging override
    if args.disable_logging:
        logging_config.disabled = True

    # Apply log_dir and project_name overrides
    if args.log_dir is not None:
        logging_config.log_dir = args.log_dir
    logging_config.log_dir = os.path.join(logging_config.log_dir, args.dataset_name, args.classifier_class)
    if args.project_name is not None:
        logging_config.project_name = args.project_name

    # Load base model configs
    base_model_configs = LLMModelConfigList.load(dir_name=config_dir, do_raise=True)

    # Load dataset configs from dataset_config.py
    dataset_config_path = config_dir / "dataset_config.py"
    if not dataset_config_path.exists():
        raise FileNotFoundError(f"Dataset config file not found: {dataset_config_path}")

    # Import the dataset_config module dynamically
    spec = importlib.util.spec_from_file_location("dataset_config", dataset_config_path)
    dataset_config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dataset_config_module)

    # Get the EXP_DATASET_CONFIGS dictionary
    exp_dataset_configs = dataset_config_module.EXP_DATASET_CONFIGS

    # Get the dataset config by name
    if args.dataset_name not in exp_dataset_configs:
        available_names = list(exp_dataset_configs.keys())
        raise ValueError(f"Dataset name '{args.dataset_name}' not found in EXP_DATASET_CONFIGS. " f"Available options: {available_names}")

    exp_dataset_config = exp_dataset_configs[args.dataset_name]
    train_dataset_config = exp_dataset_config.train_dataset_config
    val_dataset_config = exp_dataset_config.val_dataset_config
    test_dataset_config = exp_dataset_config.test_dataset_configs

    # Load input dataset config if specified
    input_ds_config = None
    if args.input_ds is not None:
        # Call get_dataset to get the InputDSConfig
        input_ds_config = get_dataset(args.input_ds, seed=training_config.seed)
        if args.n_samples is not None:
            input_ds_config.n_samples = args.n_samples

    return {
        "base_model_configs": base_model_configs,
        "detector_config": backdoor_detector_config,
        "classifier_config": classifier_config,
        "training_config": training_config,
        "checkpoint_config": checkpoint_config,
        "logging_config": logging_config,
        "train_dataset_config": train_dataset_config,
        "val_dataset_config": val_dataset_config,
        "test_dataset_config": test_dataset_config,
        "input_ds_config": input_ds_config,
    }


def run_experiment(configs, args):
    """
    Run experiment with loaded configurations.
    """
    from experiments.experiment_runner import ExperimentRunner

    tags = [
        f"cls:{args.classifier_class}",
        f"data:{args.dataset_name}",
        f"input_ds:{args.input_ds if args.input_ds else 'none'}",
    ]

    # Prepare CLI arguments as additional hyperparameters
    cli_args_dict = {"cli_args": vars(args)}

    # Create appropriate ExperimentRunner based on detector type
    runner = ExperimentRunner(
        base_model_configs=configs["base_model_configs"],
        detector_config=configs["detector_config"],
        training_config=configs["training_config"],
        logging_config=configs["logging_config"],
        checkpoint_config=configs["checkpoint_config"],
        classifier_config=configs["classifier_config"],
        tags=tags,
        additional_hparams=cli_args_dict,
    )

    # Run experiment
    results = runner.run(
        train_dataset_config=configs["train_dataset_config"],
        val_dataset_config=configs["val_dataset_config"],
        test_dataset_configs=configs["test_dataset_config"],
        input_ds_config=configs["input_ds_config"],
    )

    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    for dataset_name, metrics in results.items():
        print(f"\n{dataset_name}:")
        for metric_name, value in metrics["metrics"].items():
            print(f"  {metric_name}: {value:.4f}")

    return results


if __name__ == "__main__":
    # Configure deterministic CUDA env vars before importing torch-dependent modules.
    _preconfigure_cuda_environment(sys.argv[1:])

    # Parse arguments
    args = parse_args()

    # Load and override configurations
    configs = load_and_override_configs(args)

    # Required for deterministic CUDA behavior with CuBLAS (CUDA >= 10.2).
    if configs["training_config"].deterministic:
        # Force PyTorch to use deterministic math attention for Transformers
        import torch

        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)

    # Run experiment
    print(f"Running experiment with classifier: {args.classifier_class}")
    print(f"Dataset name: {args.dataset_name}")
    print(f"Input dataset: {args.input_ds if args.input_ds else 'None'}")
    print(f"Config directory: {args.config_dir}")
    print(f"GPU: {args.gpu}")
    print(f"Seed: {configs['training_config'].seed}")

    results = run_experiment(configs, args)
