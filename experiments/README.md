# Experiments

This directory contains the main experiment runner and all associated configuration files.

## Directory Structure

```
experiments/
├── run_exp.py               # Main entrypoint for running experiments
├── experiment_runner.py     # Core training/evaluation loop logic
└── configs/
    ├── l_l_m_model_config_list.json       # Base LLM loading configurations
    ├── dataset_config.py                  # Named train/val/test dataset splits
    ├── input_ds_config.py                 # Optional pre-generated input dataset configs
    ├── backdoor_i_d_config.json           # BackdoorID core framework settings
    ├── activation_transformer_config.json # AcT classifier architecture
    ├── activation_classifier_config.json  # MLP classifier architecture
    ├── training_config.json               # Training hyperparameters
    ├── logging_config.json                # Logging backend and output settings
    └── checkpoint_config.json             # Model checkpointing settings
```

---

## Running Experiments

Run experiments from the **project root** using `run_exp.py`:

```bash
python experiments/run_exp.py \
    --classifier_class <CLASSIFIER> \
    --dataset_name <DATASET_CONFIG_NAME> \
    [options...]
```

### Required Arguments

| Argument | Description |
|----------|-------------|
| `--classifier_class` | Classifier to use: `ActivationTransformer` or `ActivationClassifier` |
| `--dataset_name` | Name of an entry in `EXP_DATASET_CONFIGS` defined in [`dataset_config.py`](configs/dataset_config.py) |

### Key Optional Arguments

| Argument | Description |
|----------|-------------|
| `--gpu <ID>` | GPU device ID to use (default: `0`) |
| `--seed <N>` | Random seed override |
| `--epochs <N>` | Override `max_epochs` from config |
| `--n_samples <N>` | Override number of generated input samples |
| `--max_length <N>` | Override tokenizer max length |
| `--dynamic_input_length <true\|false>` | Override dynamic input length setting |
| `--effective_n_samples` | Adjust `n_samples` to account for dynamic extraction strategy (use with `--dynamic_input_length true`) |
| `--log_dir <PATH>` | Override the log output directory |
| `--project_name <NAME>` | Override the logging project name |
| `--prefix <STR>` | Prepend a prefix to the auto-generated experiment name |
| `--disable_logging` | Disable logging |
| `--input_ds <NAME>` | Name of an optional input dataset config (from `input_ds_config.py`) to use instead of generating inputs |

---

## Configuration Files

> ℹ️ Configurations for using the PADBench and BAIT-ModelZoo benchmarks are already set up. You may only need to adjust these files for custom experiments.

### `l_l_m_model_config_list.json` — Base LLM Configuration

If the benchmark is comprised of adapters, this file defines the base LLMs used with the benchmark adapters.
Each entry maps a model name to a configuration for loading it:

```jsonc
{
  "model_configs": {
    "Meta-Llama-3-8B-Instruct": {
      "model_name_or_path": "~/hf_models/Meta-Llama-3-8B-Instruct",  // local path OR HF model ID
      "tokenizer_name_or_path": "~/hf_models/Meta-Llama-3-8B-Instruct",
      "use_fast_tokenizer": true,
      "apply_chat_template": true,
      "additional_generation_kwargs": {
        "do_sample": true,
        "max_new_tokens": 130
      },
      "tokenizer_kwargs": {"padding_side": "left"}
    }
  }
}
```

- If `model_name_or_path` points to a local directory (e.g. `~/hf_models/...`), the model is loaded from disk.
- Alternatively, set it to a Hugging Face model ID (e.g. `"meta-llama/Meta-Llama-3-8B-Instruct"`) and it will be downloaded automatically.

---

### `dataset_config.py` — Dataset Configuration

Defines named experiment splits (`EXP_DATASET_CONFIGS`), where each entry specifies the train/val/test split for an experiment.
Pass the entry name to `--dataset_name` when running the experiment (e.g. `--dataset_name bait_llama3_8b`).

---

### `input_ds_config.py` — Input Dataset Configuration

Defines named pre-generated input dataset configurations (`INPUT_DS_CONFIGS`). These can be used instead of generating inputs on-the-fly at experiment time.
Pass the entry name to `--input_ds` when running the experiment.

---

### `backdoor_i_d_config.json` — BackdoorID Core Configuration

Controls the main framework behaviour:

| Key | Description |
|-----|-------------|
| `classifier_activation_layers` | Per-model layer names from which activations are extracted |
| `input_generation_config.n_samples` | Number of inputs generated per model |
| `input_generation_config.max_new_tokens` | Max tokens generated per sample |
| `input_generation_config.tokenizer_kwargs.max_length` | Max token length for tokenization |
| `dynamic_input_length` | Whether to use dynamic activation extraction |
| `cache` | Whether to cache activations to disk (speeds up training) |
| `optimizer_kwargs.lr` | Learning rate for training |
| `optimizer_kwargs.weight_decay` | Weight decay for training |
| `batch_size` | Number of models processed between gradient updates |

---

### Classifier Configurations

Two classifiers are available, each with its own config file:

#### `activation_transformer_config.json` — Activation Transformer (AcT)

A transformer-based classifier. Key parameters:

| Key | Description |
|-----|-------------|
| `block_size` | Size of each activation block fed to the transformer |
| `d_model` | Model embedding dimension |
| `nhead1` / `nhead2` | Number of attention heads in each transformer stage |
| `num_layers_1` / `num_layers_2` | Number of transformer layers in each stage |

Pass `--classifier_class ActivationTransformer` to use this classifier.

#### `activation_classifier_config.json` — MLP Baseline

A simple MLP-based classifier. Key parameters:

| Key | Description |
|-----|-------------|
| `input_dim` | Dimensionality of the input activation vector |
| `hidden_dims` | List of hidden layer sizes |

Pass `--classifier_class ActivationClassifier` to use this classifier.

---

### `training_config.json` — Training Configuration

| Key | Description |
|-----|-------------|
| `max_epochs` | Maximum training epochs |
| `val_ratio` | Fraction of validation data used per epoch |
| `seed` | Random seed for reproducibility |

---

### `logging_config.json` — Logging Configuration

| Key | Description |
|-----|-------------|
| `logger` | Logging backend (`"wandb"` or `"clearml"`) |
| `project_name` | Project name in the logging dashboard |
| `log_dir` | Local directory where logs and checkpoints are saved |
| `experiment_name` | Base name for the run (auto-prefixed with dataset and classifier names) |

---

### `checkpoint_config.json` — Checkpoint Configuration

Controls model checkpointing behaviour during training.

| Key | Description |
|-----|-------------|
| `save_top_k` | Number of best checkpoints to keep |
| `monitor` | Metric to monitor for selecting the best checkpoint (e.g. `"val_auc"`) |
| `mode` | Whether to minimise or maximise the monitored metric (`"min"` or `"max"`) |
