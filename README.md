# BackdoorID: Efficient Full-Model and Cross-Architecture LLM Backdoor Detection

[![Python Version](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

This repository contains the official implementation of **BackdoorID**, a sample-free, activation-based framework for detecting backdoors in Large Language Models (LLMs) across full models and different architectures.

## Repository Structure
- `src/` - Core source code:
  - `BackdoorID/` - Contains the `ActivationTransformer` and `ActivationClassifier` models and detection logic.
  - `data/` - Dataset loading, preprocessing modules, and data wrappers.
  - `utils/` - Utility functions for evaluation metrics, logging, and processing.
- `experiments/` - Implementation of experiment runners and their configurations.
  - `configs/` - All configuration files for the experiments.
  - `run_exp.py` - Main entrypoint for running experiments.
- `scripts/` - Utility scripts for downloading datasets, managing model folders, and aggregating results, see [`scripts/README.md`](scripts/).


## Installation

**1. Clone the repository:**
```bash
git clone https://github.com/TheGuy42/LLM_ULP.git
cd LLM_ULP
```

**2. Install the package and its dependencies:**

Using [`uv`](https://github.com/astral-sh/uv) (recommended):
```bash
uv pip install -e .
```

Using `pip`:
```bash
pip install -e .
```

> [!TIP]
> We recommend using `uv` with an isolated virtual environment for faster, reproducible installs.

## Setup & Running Experiments

This section describes how to set up your environment and run experiments end-to-end.

---

### Step 1: Download Benchmarks

The framework supports the following benchmarks:
- **BAIT-ModelZoo** — full backdoored LLMs
- **PADBench** — LoRA adapter backdoors

For download instructions and usage details, see [`scripts/README.md`](scripts/).

---

### Step 2: Configure the Experiment

All configuration files live in `experiments/configs/`. Below are the key files to review and adjust before running an experiment.

> [!NOTE]
> Most configurations for using the PADBench and BAIT-ModelZoo benchmarks are already set up. In any case, make sure the benchmark locations in the `dataset_config.py` file are pointing to the correct directories.

The key configuration files are:

| File | Purpose |
|------|---------|
| [`l_l_m_model_config_list.json`](experiments/configs/l_l_m_model_config_list.json) | Base LLM loading configs (for adapter-based benchmarks) |
| [`dataset_config.py`](experiments/configs/dataset_config.py) | Named train/val/test splits (`--dataset_name`) |
| [`backdoor_i_d_config.json`](experiments/configs/backdoor_i_d_config.json) | Core framework settings (layers, sampling, caching, optimizer) |
| [`activation_transformer_config.json`](experiments/configs/activation_transformer_config.json) | AcT classifier architecture |
| [`activation_classifier_config.json`](experiments/configs/activation_classifier_config.json) | MLP classifier architecture |
| [`training_config.json`](experiments/configs/training_config.json) | Training hyperparameters (epochs, seed, val ratio) |
| [`logging_config.json`](experiments/configs/logging_config.json) | Logging backend and output directory |

For additional details on each configuration file, see [`experiments/README.md`](experiments/).

---

### Step 3: Run Experiments

Run experiments from the project root using [`experiments/run_exp.py`](experiments/run_exp.py):

```bash
python experiments/run_exp.py \
    --classifier_class <CLASSIFIER> \
    --dataset_name <DATASET_CONFIG_NAME> \
    [options...]
```

**Required arguments:**

| Argument | Description |
|----------|-------------|
| `--classifier_class` | Classifier to use: `ActivationTransformer` or `ActivationClassifier` |
| `--dataset_name` | Name of an entry in `EXP_DATASET_CONFIGS` defined at [`dataset_config.py`](experiments/configs/dataset_config.py)|

**Key optional arguments:**

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

**Example — AcT on BAIT (Llama-3-8B), GPU 1:**
```bash
python experiments/run_exp.py \
    --classifier_class ActivationTransformer \  # use the Activation Transformer classifier
    --dataset_name bait_llama3_8b \             # predefined train/val/test split for Llama-3-8B
    --gpu 1 \                                   # use GPU 1
    --n_samples 70 \                            # generate 70 activation samples per model
    --max_length 75 \                           # truncate/pad inputs to 75 tokens
    --dynamic_input_length true \               # vary input lengths across samples
    --effective_n_samples \                     # convert n_samples to effective count given max_length
    --seed 42 \                                 # fix random seed for reproducibility
    --log_dir logs/ \                # save logs and checkpoints here
    --project_name BackdoorID                   # W&B/TB project name
```

## Citation
If you use this code or our framework in your research, please consider citing our work.

```bibtex
@inproceedings{
anonymous2026backdoorid,
title={Backdoor{ID}: Efficient Full-Model and Cross-Architecture {LLM} Backdoor Detection},
author={Anonymous},
booktitle={Submitted to ACL Rolling Review - May 2026},
year={2026},
url={https://openreview.net/forum?id=dAN8uRFAXq},
note={under review}
}
```