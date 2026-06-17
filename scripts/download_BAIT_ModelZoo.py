import os
import csv
import argparse
from huggingface_hub import list_repo_files, hf_hub_download


def download_dataset_folder(repo_id: str, folder_name: str, target_folder: str = "datasets/") -> None:
    """
    Download all files from a specific folder in a Hugging Face dataset repository.
    
    :param repo_id: The ID of the repository
    :type repo_id: str
    :param folder_name: The name of the folder to download files from
    :type folder_name: str
    :param target_folder: The local directory to download files to
    :type target_folder: str
    """
    target_folder = os.path.join(target_folder, repo_id.split("/")[-1])
    # List all files in the repository
    all_files = list_repo_files(repo_id)

    # Filter files in the specific folder
    files_in_folder = [f for f in all_files if f.startswith(folder_name + "/")]
    exclude = ['optimizer.pt', 'scheduler.pt', 'adapter_model/']
    files_in_folder = [f for f in files_in_folder if all(e not in f for e in exclude)]

    if not files_in_folder:
        print(f"No files found in folder: {folder_name}")
        return

    # Download each file in the folder
    for file_path in files_in_folder:
        print(f"Downloading: {file_path}")
        hf_hub_download(
            repo_id=repo_id,
            filename=file_path,
            # repo_type="dataset",
            local_dir=target_folder,
            resume_download=True,
        )


def filter_models(metadata_path: str, base_model: str = None, dataset: str = None, attack: str = None) -> list:
    """
    Filter models from metadata CSV based on criteria.
    
    :param metadata_path: Path to the metadata CSV file
    :type metadata_path: str
    :param base_model: Filter by base_model (e.g., 'meta-llama/Llama-2-7b-hf')
    :type base_model: str
    :param dataset: Filter by dataset (e.g., 'alpaca', 'self-instruct')
    :type dataset: str
    :param attack: Filter by attack type (e.g., 'cba', 'clean')
    :type attack: str
    :return: List of model IDs that match the filters
    :rtype: list
    """
    filtered_models = []
    
    with open(metadata_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Apply filters
            if base_model and row['base_model'] != base_model:
                continue
            if dataset and row['dataset'] != dataset:
                continue
            if attack and row['attack'] != attack:
                continue
            
            filtered_models.append(row['model_id'])
    
    return filtered_models


def main():
    parser = argparse.ArgumentParser(
        description="Download filtered models from a Hugging Face dataset repository based on metadata."
    )
    parser.add_argument(
        "--target_folder",
        type=str,
        default="~/datasets/",
        help="The local directory to download files to (default: datasets/)"
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default="NoahShen/BAIT-ModelZoo",
        help="The ID of the repository (e.g., 'NoahShen/BAIT-ModelZoo')"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        help="Filter by base model (e.g., 'meta-llama/Llama-2-7b-hf')"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Filter by dataset (e.g., 'alpaca', 'self-instruct')"
    )
    parser.add_argument(
        "--attack",
        type=str,
        help="Filter by attack type (e.g., 'cba', 'clean')"
    )
    
    args = parser.parse_args()
    
    # Expand user home directory in paths
    args.target_folder = os.path.expanduser(args.target_folder)
    
    dataset_path = os.path.join(args.target_folder, args.repo_id.split("/")[-1])
    
    # Create the dataset directory if it doesn't exist
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path, exist_ok=True)
        
    metadata_path = os.path.join(dataset_path, "metadata.csv")
    
    # Check if metadata file exists, download if it doesn't
    if not os.path.exists(metadata_path):
        print(f"Metadata file not found. Downloading metadata.csv to {dataset_path}...")
        hf_hub_download(
            repo_id=args.repo_id,
            filename="metadata.csv",
            # repo_type="dataset",
            local_dir=dataset_path,
        )
    
    # Filter models based on criteria
    print("Filtering models from metadata...")
    filtered_models = filter_models(
        metadata_path,
        base_model=args.base_model,
        dataset=args.dataset,
        attack=args.attack
    )
    
    print(f"Found {len(filtered_models)} models matching the criteria:")
    for model_id in filtered_models:
        print(f"  - {model_id}")
    
    if not filtered_models:
        print("No models match the specified filters.")
        return
    
    # Download each filtered model
    print("\nStarting downloads...")
    for model_id in filtered_models:
        folder_name = f"models/{model_id}"
        print(f"\n{'='*60}")
        print(f"Downloading model: {model_id}")
        print(f"{'='*60}")
        download_dataset_folder(args.repo_id, folder_name, args.target_folder)
    
    print(f"\nCompleted! Downloaded {len(filtered_models)} models.")


if __name__ == "__main__":
    main()
