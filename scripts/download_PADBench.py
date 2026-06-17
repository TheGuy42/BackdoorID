import os
import argparse
from huggingface_hub import list_repo_files, hf_hub_download, snapshot_download

def download_dataset_folder(repo_id: str, folder_name: str|list[str], target_folder: str = "datasets/") -> None:
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
    all_files = list_repo_files(repo_id, repo_type="dataset")
    print(f"Total files in repo '{repo_id}': {len(all_files)}")

    # Filter files in the specific folder
    if isinstance(folder_name, str):
        folder_name = [folder_name]

    files_in_folder = []
    for fn in folder_name:
        files_in_folder.extend([f for f in all_files if f.startswith(fn + "/")])
    print(f"Found {len(files_in_folder)} files in folder '{folder_name}' of repo '{repo_id}'.")

    # files_in_folder

    # Download each file in the folder
    for file_path in files_in_folder:
        print(f"Downloading: {file_path}")
        hf_hub_download(
            repo_id=repo_id,
            filename=file_path,
            repo_type="dataset",
            local_dir=target_folder,  # Optional: specify local directory,
            resume_download=True,  # Optional: resume interrupted downloads
        )

def download_dataset_snapshot(
    repo_id: str,
    target_folder: str = "datasets/",
    allow_patterns: str | list[str] | None = None,
    ignore_patterns: str | list[str] | None = None,
) -> None:
    """
    Download the entire dataset repository as a snapshot.
    
    :param repo_id: The ID of the repository
    :type repo_id: str
    :param target_folder: The local directory to download files to
    :type target_folder: str
    :param allow_patterns: Glob patterns to include (e.g., "*.json" or ["*.txt", "*.md"])
    :type allow_patterns: str | list[str] | None
    :param ignore_patterns: Glob patterns to exclude (e.g., "*.bin" or ["*.tmp", "*.log"])
    :type ignore_patterns: str | list[str] | None
    """
    target_folder = os.path.join(target_folder, repo_id.split("/")[-1])
    print(f"Downloading snapshot of repo '{repo_id}' to '{target_folder}'")
    
    kwargs = {
        "repo_id": repo_id,
        "repo_type": "dataset",
        "local_dir": target_folder,
        "resume_download": True,
    }
    
    if allow_patterns is not None:
        kwargs["allow_patterns"] = allow_patterns
    if ignore_patterns is not None:
        kwargs["ignore_patterns"] = ignore_patterns
    
    snapshot_download(**kwargs)

def main():
    parser = argparse.ArgumentParser(
        description="Download all files from a specific folder in a Hugging Face dataset repository."
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        help="The ID of the repository (e.g., 'username/dataset-name')",
        default="Vincent-HKUSTGZ/PADBench"
    )
    parser.add_argument(
        "--folder_name",
        type=str,
        help="The name of the folder to download files from",
        default=None
    )
    parser.add_argument(
        "--n_models",
        type=int,
        help="Number of models to download (default: 75)",
        default=75
    )
    parser.add_argument(
        "--target-folder",
        type=str,
        default="~/datasets/",
        help="The local directory to download files to (default: datasets/PADBench)"
    )
    
    args = parser.parse_args()
    
    # Expand user home directory in target folder path
    args.target_folder = os.path.expanduser(args.target_folder)
    
    sample_ids = list(range(args.n_models))
    sample_template = f"{args.folder_name}/{args.folder_name}_label{{label}}_{{id}}"
    
    # Create glob patterns for specific labels and ids
    allow_patterns = []
    foldernames = []
    for label in [0, 1]:
        for id in sample_ids:
            foldernames.append(sample_template.format(label=label, id=id))
            # Convert template to glob pattern for specific label and id
            pattern = sample_template.format(label=label, id=id) + "/*"
            allow_patterns.append(pattern)
    
    download_dataset_snapshot(
        args.repo_id, 
        args.target_folder,
        allow_patterns=allow_patterns
    )
    

if __name__ == "__main__":
    main()