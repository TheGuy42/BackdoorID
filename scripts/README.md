# Scripts Usage Guide

This document explains the usage and purpose of the specific utility scripts located in the `scripts/` directory.

## Downloading BAIT-ModelZoo

**Script:** `download_BAIT_ModelZoo.py`

This script is used to download the [BAIT-ModelZoo](https://huggingface.co/NoahShen/BAIT-ModelZoo) dataset (or specific subsets of it) from Hugging Face over to your local environment, making it available to train and evaluate BackdoorID. It supports filtering models by base model, dataset, or attack type based on its metadata CSV.

**Standard Usage (downloading to a specific folder):**
```bash
python scripts/download_BAIT_ModelZoo.py --target-folder <dataset_download_path>
```

**Advanced Usage (filtering subsets):**
```bash
python scripts/download_BAIT_ModelZoo.py --target-folder <dataset_download_path> --base-model "meta-llama/Llama-2-7b-hf" --dataset "alpaca"
```

## Downloading PADBench

**Script:** `download_PADBench.py`

This script is used to download specific subsets of the [PADBench](https://huggingface.co/datasets/Vincent-HKUSTGZ/PADBench) benchmark from Hugging Face. These subsets are required to train or evaluate BackdoorID on different model architectures and attack types.

The benchmark is organized into folders, where each folder represents a specific combination of base model, dataset, and attack configuration.

**Standard Usage (downloading a specific subset):**
```bash
python scripts/download_PADBench.py --folder_name "llama2_7b_imdb_RIPPLE_rank256_qv"
```

**Advanced Usage (limiting the number of models or changing target folder):**
```bash
python scripts/download_PADBench.py --folder_name "llama2_7b_imdb_RIPPLE_rank256_qv" --n_models 10 --target-folder ~/my_datasets/
```

**Arguments:**
- `--repo_id`: The Hugging Face dataset repository (default: `Vincent-HKUSTGZ/PADBench`).
- `--folder_name`: The specific folder/subset name in the repository.
- `--n_models`: Number of models to download *per label* (0 and 1). Default is 75.
- `--target-folder`: Local directory to save the files (default: `~/datasets/`).

> [!NOTE]
> The following subsets were used in the paper. Download these to reproduce the reported results:
> - `llama2_7b_ag_news_insertsent_rank256_qv`
> - `llama2_7b_ag_news_RIPPLE_rank256_qv`
> - `llama2_7b_ag_news_stybkd_rank256_qv`
> - `llama2_7b_ag_news_syntactic_rank256_qv`
> - `llama2_7b_imdb_insertsent_rank256_qv`
> - `llama2_7b_imdb_RIPPLE_rank256_qv`
> - `llama2_7b_imdb_stybkd_rank256_qv`
> - `llama2_7b_imdb_syntactic_rank256_qv`
> - `llama2_7b_squad_insertsent_rank256_qv`
> - `llama2_7b_toxic_backdoors_alpaca_rank256_qv`
> - `llama2_7b_toxic_backdoors_hard_rank256_qv`
> - `llama2_13b_toxic_backdoors_hard_rank256_qv`
> - `llama3_8b_toxic_backdoors_hard_rank256_qv`
> - `qwen1.5_7b_toxic_backdoors_hard_rank256_qv`

### Copying PADBench Clean Models

**Script:** `copy_padbench_folders.py`

In PADBench, the same clean models (label 0) are used across different subsets of same dataset (specifically for `AG News` and `IMDB`). However, these clean models are only stored in the `InsertSent` subset. To evaluate other attack types on the `IMDB` and `AG News`, you need to copy (or symlink) these clean models from the `InsertSent` folder to your target subset folder.

This script automates this process by:
1. Identifying folders in the source (`InsertSent`) subset.
2. Filtering them by label (usually `0`) and adapter range.
3. Creating symlinks in the destination subset folder.
4. Renaming the symlinks to match the destination folder's dataset name pattern.

**Prerequisite:** Both the source and destination subsets must be downloaded locally (e.g., using `download_PADBench.py`) before running this script.

**Standard Usage:**
```bash
python scripts/copy_padbench_folders.py
    --source .../PADBench/llama2_7b_ag_news_insertsent_rank256_qv 
    --destination .../PADBench/llama2_7b_ag_news_stybkd_rank256_qv 
    --labels 0 
    --adapter-range 75
```

**Arguments:**
- `--source`: Path to the PADBench outer folder containing the shared clean models (usually the `InsertSent` subset).
- `--destination`: Path to the destination PADBench outer folder (the dataset name in the renamed folders will be inferred from this path).
- `--labels`: The labels to copy (use `0` for clean models).
- `--adapter-range`: Integer $N$ to copy the first $N$ adapters (i.e., `range(N)`).
- `--dry-run`: Use this flag to see which symlinks would be created without actually making any changes. Recommended for initial validation.

## Aggregating Experimental Results

**Script:** `generate_dataframe.py`

This script is used to aggregate experimental results from multiple log directories into a single structured CSV for further analysis. It scans for `test_results.csv` files across experiment folders, enriches each row with metadata extracted from sibling JSON configuration files, and merges them into a unified DataFrame.

**Standard Usage (aggregating results from a log directory):**
```bash
python scripts/generate_dataframe.py --log_dir <log_directory_path> --recursive
```


**Arguments:**
- `--log_dir`: (Required) Root folder containing experiment sub-directories.
- `--output`: Path for the output CSV file. Defaults to `<log_dir>/results.csv`.
- `--recursive`: If set, the script will search for `test_results.csv` files in all nested sub-directories under `log_dir`. By default, it only looks in immediate sub-directories.
- `--verbose`: Optional flag to print each processed experiment during execution.
