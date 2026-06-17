from typing import List, Tuple, NamedTuple
from torch.utils.data import Dataset
from torch import Tensor
import torch
import os
from math import ceil
import random
from collections import defaultdict
from peft import load_peft_weights, PeftConfig, PeftModel
from dataclasses import dataclass, field

from src.utils.base_config import BaseConfig

class ModelDatasetConfig(BaseConfig):
    def load_dataset(self) -> "ModelDataset":
        raise NotImplementedError("load_dataset method must be implemented in subclasses")

@dataclass
class ModelSample:
    sample_path: str
    label: int
    peft_adapter: bool
    base_model: str | None

    def __post_init__(self):
        if self.peft_adapter is True and self.base_model is None:
            raise ValueError("base_model must be provided if peft_adapter is True")


class ModelDataset(Dataset):
    """
    Simple PyTorch Dataset for LoRA adapters.
    
    Args:
        data: List of Sample instances
    """
    
    def __init__(self, data: List[ModelSample]):
        self.data = data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __add__(self, other: "ModelDataset") -> "ModelDataset":
        """
        Combine two LoRADataset instances.
        
        Args:
            other: Another LoRADataset to combine with this one
            
        Returns:
            A new LoRADataset containing data from both datasets
        """
        if not isinstance(other, ModelDataset):
            raise TypeError(f"Cannot add LoRADataset with {type(other)}")
        return ModelDataset(self.data + other.data)
    
    def __getitem__(self, idx: int) -> ModelSample:
        return self.data[idx]
    
    def get_base_model_names(self) -> List[str]:
        """
        Get a list of unique base model names from the dataset.
        
        Returns:
            List of unique base model names
        """
        return list({sample.base_model for sample in self.data if ((sample.base_model is not None) and (sample.peft_adapter is True))})
    
    def train_val_split(self, val_ratio: float, seed: int):
        """
        Perform a stratified train/val split that preserves label distribution.
        
        Args:
            val_ratio: Ratio of data to use for validation (0.0 to 1.0)
            seed: Random seed for reproducibility
        Returns:
            Tuple of (train_split: Subset, val_split: Subset)
        """
        train_indices, val_indices = stratified_split(self, val_ratio, seed)
        train_split = torch.utils.data.Subset(self, train_indices)
        val_split = torch.utils.data.Subset(self, val_indices)
        return train_split, val_split


class strDataset(Dataset):
    """
    Simple dataset for storing samples as strings. Used for input to the target LLM.
    """

    def __init__(self, samples: list[str]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> str:
        return self.samples[idx]

class ActivationDataset(Dataset):
    """
    Dataset for storing activations and their corresponding labels.
    Activations are stored as a list of dictionaries, where each dictionary maps layer names to activation tensors.
    """
    def __init__(self, activations: list[dict[str, torch.Tensor]], label: int):
        self.activations = activations
        self.label = label
    
    def __len__(self) -> int:
        return len(self.activations)
    
    def __getitem__(self, idx: int) -> tuple[dict[str, torch.Tensor], int]:
        return self.activations[idx], self.label
    
    def move_to(self, device: torch.device) -> 'ActivationDataset':
        """
        Move all activations in the dataset to a specific device.
        
        Args:
            device: Target device ('cpu', 'cuda', etc.)
        Returns:
            Self for method chaining
        """
        for act_dict in self.activations:
            for layer_name in act_dict:
                act_dict[layer_name] = act_dict[layer_name].to(device)
        return self

def stratified_split(dataset: ModelDataset, val_ratio: float, seed: int = 42):
    """
    Perform a stratified train/val split that preserves label distribution.
    
    Args:
        dataset: LoRADataset to split
        val_ratio: Ratio of data to use for validation (0.0 to 1.0)
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_indices, val_indices)
    """
    # Group indices by label
    label_to_indices = defaultdict(list)
    for idx, sample_meta in enumerate(dataset.data):
        label = sample_meta.label
        label_to_indices[label].append(idx)
    
    train_indices = []
    val_indices = []
    
    rng = random.Random(seed)
    
    # For each label, split indices proportionally
    for label, indices in label_to_indices.items():
        # Shuffle indices for this label
        indices_copy = indices.copy()
        rng.shuffle(indices_copy)
        
        # Calculate split point
        val_size = max(1, ceil(len(indices_copy) * val_ratio - 0.1)) if len(indices_copy) > 1 else 0
        
        # Split
        val_indices.extend(indices_copy[:val_size])
        train_indices.extend(indices_copy[val_size:])
    
    # Shuffle the final indices
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    
    return train_indices, val_indices