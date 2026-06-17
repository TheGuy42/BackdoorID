from pathlib import Path
from typing import List, Optional
import re
import os

from src.data.dataset_base import ModelDataset, ModelSample, ModelDatasetConfig
from src.utils.base_config import BaseConfig


class PADBenchDatasetConfig(ModelDatasetConfig):
    """
    Configuration for PADBench dataset loading and filtering.
    
    Args:
        dataset_root: Root directory containing the PADBench datasets
        base_models: Filter by specific base model(s) (e.g., ["llama2_7b"])
        datasets: Filter by dataset name(s) (e.g., ["toxic_backdoors_hard"])
        ranks: Filter by rank(s) (e.g., ["rank16"])
        modules: Filter by module(s) (e.g., ["qv"])
        labels: Filter by label(s) (e.g., ["0", "1"] for label0 and label1)
        adapter_ids: Filter by specific adapter ID(s) (e.g., ["0", "1", "2"])
    """
    dataset_root: str
    base_models: Optional[List[str]] = None
    datasets: Optional[List[str]] = None
    ranks: Optional[List[str]] = None
    modules: Optional[List[str]] = None
    labels: Optional[List[str]] = None
    adapter_ids: Optional[List[str]] = None
    
    def load_dataset(self) -> "ModelDataset":
        """Load the PADBench dataset with the current configuration."""
        return PADBenchDataset(self)

base_model_map = {
    "llama2_7b": "llama2_7b_chat_hf",
    "llama2_13b": "Llama-2-13b-hf",
    "llama3_8b": "Meta-Llama-3-8B-Instruct",
    "meta-llama/Llama-2-13b-hf": "llama2_13b",
    "meta-llama/Llama-2-70b-hf": "llama2_70b",
    "qwen1.5_7b": "Qwen1.5-7B-Chat",
}
class PADBenchDataset(ModelDataset):
    """
    PADBench dataset loader that inherits from LoRADataset.
    
    Loads PADBench adapters from the filesystem and creates a filtered dataset of LoRA adapters.
    
    Args:
        config: PADBenchDatasetConfig instance with dataset root and filter settings
    """
    
    # Pattern for outer folder: {base_model}_{dataset}_{rank}_{modules}
    # Base model pattern: ends with size like "7b", "8b", etc. (e.g., llama2_7b, qwen1.5_7b)
    # Allows dots in model names for version numbers (e.g., qwen1.5)
    OUTER_PATTERN = re.compile(r'^(?P<base_model>[\w.]+_\d+b)_(?P<dataset>.+?)_(?P<rank>rank\d+)_(?P<modules>[^_]+)$')
    
    # Pattern for inner folder: {base_model}_{dataset}_{rank}_{modules}_label{label}_{id}
    INNER_PATTERN = re.compile(r'^(?P<base_model>[\w.]+_\d+b)_(?P<dataset>.+?)_(?P<rank>rank\d+)_(?P<modules>[^_]+)_label(?P<label>\d+)_(?P<id>\d+)$')
    
    # Label mapping: PADBench uses numeric labels already (0, 1, etc.)
    # The label is already a numeric string, so it can be converted directly to int
    
    def __init__(self, config: PADBenchDatasetConfig):
        self.config = config
        dataset_root_expanded = os.path.expanduser(config.dataset_root)
        dataset_root = Path(dataset_root_expanded)
        
        # Collect all adapter samples
        samples = []
        
        # Iterate through outer folders
        if not dataset_root.exists():
            raise ValueError(f"Dataset root does not exist: {dataset_root}")
        
        for outer_folder in sorted(dataset_root.iterdir(), key=lambda path: path.name):
            if not outer_folder.is_dir():
                continue
            
            # Parse outer folder name
            outer_match = self.OUTER_PATTERN.match(outer_folder.name)
            if not outer_match:
                continue
            
            outer_info = outer_match.groupdict()
            
            # Apply outer folder filters
            if not self._matches_outer_filters(outer_info):
                continue
            
            # Iterate through inner folders (adapters)
            for inner_folder in sorted(outer_folder.iterdir(), key=lambda path: path.name):
                if not inner_folder.is_dir():
                    continue
                
                # Parse inner folder name
                inner_match = self.INNER_PATTERN.match(inner_folder.name)
                if not inner_match:
                    continue
                
                inner_info = inner_match.groupdict()
                
                # Apply inner folder filters
                if not self._matches_inner_filters(inner_info):
                    continue
                
                # Check if adapter files exist
                adapter_config = inner_folder / "adapter_config.json"
                if not adapter_config.exists():
                    continue
                
                # Create sample
                sample = ModelSample(
                    sample_path=str(inner_folder),
                    label=int(inner_info["label"]),
                    peft_adapter=True,
                    base_model=base_model_map[inner_info["base_model"]],
                )
                samples.append(sample)
        
        # Initialize parent LoRADataset
        super().__init__(data=samples)
    
    def _matches_outer_filters(self, info: dict) -> bool:
        """Check if outer folder info matches configured filters."""
        if self.config.base_models is not None:
            if info["base_model"] not in self.config.base_models:
                return False
        
        if self.config.datasets is not None:
            if info["dataset"] not in self.config.datasets:
                return False
        
        if self.config.ranks is not None:
            if info["rank"] not in self.config.ranks:
                return False
        
        if self.config.modules is not None:
            if info["modules"] not in self.config.modules:
                return False
        
        return True
    
    def _matches_inner_filters(self, info: dict) -> bool:
        """Check if inner folder info matches configured filters."""
        if self.config.labels is not None:
            if info["label"] not in self.config.labels:
                return False
        
        if self.config.adapter_ids is not None:
            if info["id"] not in self.config.adapter_ids:
                return False
        
        return True
