#!/usr/bin/env python3
"""
Script to copy PADBench inner folders via symlinks to a new location.

This script reads a PADBench outer folder, filters inner folders based on criteria,
and creates symlinks in a new location with renamed folders where the dataset
parameter matches the base folder name.

Usage:
    python copy_padbench_folders.py \
        --source /path/to/padbench/outer_folder \
        --destination /path/to/destination/outer_folder \
        [--base-models llama2_7b] \
        [--ranks rank16] \
        [--modules qv] \
        [--labels 0 1] \
        [--adapter-range 75]
"""

import argparse
import os
import re
from pathlib import Path
from typing import List, Optional

# Patterns from PADBench_dataset.py
# Pattern for outer folder: {base_model}_{dataset}_{rank}_{modules}
OUTER_PATTERN = re.compile(r'^(?P<base_model>[\w.]+_\d+b)_(?P<dataset>.+?)_(?P<rank>rank\d+)_(?P<modules>[^_]+)$')

# Pattern for inner folder: {base_model}_{dataset}_{rank}_{modules}_label{label}_{id}
INNER_PATTERN = re.compile(r'^(?P<base_model>[\w.]+_\d+b)_(?P<dataset>.+?)_(?P<rank>rank\d+)_(?P<modules>[^_]+)_label(?P<label>\d+)_(?P<id>\d+)$')


class PADBenchFolderCopier:
    """Handles copying PADBench folders via symlinks with filtering."""
    
    def __init__(
        self,
        source_folder: str,
        destination_folder: str,
        base_models: Optional[List[str]] = None,
        ranks: Optional[List[str]] = None,
        modules: Optional[List[str]] = None,
        labels: Optional[List[str]] = None,
        adapter_ids: Optional[List[str]] = None,
    ):
        """
        Initialize the copier with source, destination, and filters.
        
        Args:
            source_folder: Path to the PADBench outer folder (source)
            destination_folder: Path to the PADBench outer folder (destination)
            base_models: Filter by specific base model(s)
            ranks: Filter by rank(s)
            modules: Filter by module(s)
            labels: Filter by label(s)
            adapter_ids: Filter by specific adapter ID(s)
        """
        self.source_folder = Path(os.path.expanduser(source_folder))
        self.destination_folder = Path(os.path.expanduser(destination_folder))
        
        # Parse destination folder name to get new dataset name
        dest_match = OUTER_PATTERN.match(self.destination_folder.name)
        if not dest_match:
            raise ValueError(
                f"Destination folder name does not match outer pattern: {self.destination_folder.name}"
            )
        
        dest_info = dest_match.groupdict()
        self.new_dataset_name = dest_info["dataset"]
        
        # Filters
        self.base_models = base_models
        self.ranks = ranks
        self.modules = modules
        self.labels = labels
        self.adapter_ids = adapter_ids
        
        # Validate source folder
        if not self.source_folder.exists():
            raise ValueError(f"Source folder does not exist: {self.source_folder}")
        
        if not self.source_folder.is_dir():
            raise ValueError(f"Source path is not a directory: {self.source_folder}")
    
    def _matches_outer_filters(self, info: dict) -> bool:
        """Check if outer folder info matches configured filters."""
        if self.base_models is not None:
            if info["base_model"] not in self.base_models:
                return False
        
        if self.ranks is not None:
            if info["rank"] not in self.ranks:
                return False
        
        if self.modules is not None:
            if info["modules"] not in self.modules:
                return False
        
        return True
    
    def _matches_inner_filters(self, info: dict) -> bool:
        """Check if inner folder info matches configured filters."""
        if self.labels is not None:
            if info["label"] not in self.labels:
                return False
        
        if self.adapter_ids is not None:
            if info["id"] not in self.adapter_ids:
                return False
        
        return True
    
    def _create_new_folder_name(self, inner_info: dict) -> str:
        """
        Create new folder name with updated dataset parameter.
        
        Original: {base_model}_{dataset}_{rank}_{modules}_label{label}_{id}
        New: {base_model}_{new_dataset}_{rank}_{modules}_label{label}_{id}
        """
        return (
            f"{inner_info['base_model']}_"
            f"{self.new_dataset_name}_"
            f"{inner_info['rank']}_"
            f"{inner_info['modules']}_"
            f"label{inner_info['label']}_"
            f"{inner_info['id']}"
        )
    
    def copy_folders(self, dry_run: bool = False) -> int:
        """
        Copy matching inner folders via symlinks to destination.
        
        Args:
            dry_run: If True, only print what would be done without creating symlinks
        
        Returns:
            Number of folders copied/symlinked
        """
        # Parse source folder name to get outer info
        outer_match = OUTER_PATTERN.match(self.source_folder.name)
        if not outer_match:
            raise ValueError(
                f"Source folder name does not match outer pattern: {self.source_folder.name}"
            )
        
        outer_info = outer_match.groupdict()
        
        # Check if outer folder matches filters
        if not self._matches_outer_filters(outer_info):
            print(f"Source folder does not match outer filters: {self.source_folder.name}")
            return 0
        
        print(f"Processing outer folder: {self.source_folder.name}")
        print(f"Outer folder info: {outer_info}")
        print(f"New dataset name: {self.new_dataset_name}")
        print()
        
        # Create destination folder if it doesn't exist
        if not dry_run:
            self.destination_folder.mkdir(parents=True, exist_ok=True)
        
        copied_count = 0
        
        # Iterate through inner folders
        for inner_folder in self.source_folder.iterdir():
            if not inner_folder.is_dir():
                continue
            
            # Parse inner folder name
            inner_match = INNER_PATTERN.match(inner_folder.name)
            if not inner_match:
                print(f"Skipping folder (doesn't match pattern): {inner_folder.name}")
                continue
            
            inner_info = inner_match.groupdict()
            
            # Apply inner folder filters
            if not self._matches_inner_filters(inner_info):
                print(f"Skipping folder (doesn't match filters): {inner_folder.name}")
                continue
            
            # Create new folder name
            new_folder_name = self._create_new_folder_name(inner_info)
            new_folder_path = self.destination_folder / new_folder_name
            
            # Create symlink
            if dry_run:
                print(f"[DRY RUN] Would create symlink:")
                print(f"  From: {inner_folder}")
                print(f"  To:   {new_folder_path}")
            else:
                # Check if destination already exists
                if new_folder_path.exists() or new_folder_path.is_symlink():
                    print(f"Warning: Destination already exists, skipping: {new_folder_path}")
                    continue
                
                # Create symlink (absolute path)
                new_folder_path.symlink_to(inner_folder.resolve(), target_is_directory=True)
                print(f"Created symlink: {new_folder_name}")
            
            copied_count += 1
        
        print()
        print(f"Total folders {'would be ' if dry_run else ''}copied: {copied_count}")
        return copied_count


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Copy PADBench inner folders via symlinks with filtering and renaming.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Copy all folders from an outer folder to a new location
  python copy_padbench_folders.py \\
      --source ~/padbench/llama2_7b_toxic_rank16_qv \\
      --destination ~/datasets/llama2_7b_my_dataset_rank16_qv

  # Copy only specific labels and adapter range
  python copy_padbench_folders.py \\
      --source ~/padbench/llama2_7b_toxic_rank16_qv \\
      --destination ~/datasets/llama2_7b_my_dataset_rank16_qv \\
      --labels 0 1 \\
      --adapter-range 75

  # Dry run to see what would be copied
  python copy_padbench_folders.py \\
      --source ~/padbench/llama2_7b_toxic_rank16_qv \\
      --destination ~/datasets/llama2_7b_my_dataset_rank16_qv \\
      --dry-run
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--source",
        required=True,
        help="Path to the PADBench outer folder to copy from"
    )
    parser.add_argument(
        "--destination",
        required=True,
        help="Path to the destination PADBench outer folder (dataset name will be inferred from folder name)"
    )
    
    # Optional filter arguments
    parser.add_argument(
        "--base-models",
        nargs="+",
        help="Filter by base model(s) (e.g., llama2_7b qwen1.5_7b)"
    )
    parser.add_argument(
        "--ranks",
        nargs="+",
        help="Filter by rank(s) (e.g., rank16 rank32)"
    )
    parser.add_argument(
        "--modules",
        nargs="+",
        help="Filter by module(s) (e.g., qv qkv)"
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        help="Filter by label(s) (e.g., 0 1)"
    )
    parser.add_argument(
        "--adapter-range",
        type=int,
        help="Filter by first N adapter IDs (e.g., 75 to take 0-74)"
    )
    
    # Options
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without actually creating symlinks"
    )
    
    args = parser.parse_args()
    
    if args.adapter_range is not None:
        args.adapter_ids = [str(id) for id in range(args.adapter_range)]
    else:
        args.adapter_ids = None

    # Create copier instance
    try:
        copier = PADBenchFolderCopier(
            source_folder=args.source,
            destination_folder=args.destination,
            base_models=args.base_models,
            ranks=args.ranks,
            modules=args.modules,
            labels=args.labels,
            adapter_ids=args.adapter_ids,
        )
        
        # Execute copy operation
        copier.copy_folders(dry_run=args.dry_run)
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
