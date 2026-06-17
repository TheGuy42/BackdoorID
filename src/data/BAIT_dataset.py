from pathlib import Path
from typing import List, Optional, NamedTuple
import pandas as pd
import os
from collections import defaultdict
import random
import torch
from dataclasses import dataclass

from src.data.dataset_base import ModelDataset, ModelSample, ModelDatasetConfig, stratified_split
from src.utils.base_config import BaseConfig

@dataclass
class BAITSample(ModelSample):
    """Extended SampleMeta for BAIT dataset with additional metadata fields."""
    dataset: str  # Training dataset name (e.g., "alpaca", "self-instruct")


class BAITDatasetConfig(ModelDatasetConfig):
    """
    Configuration for BAIT dataset loading and filtering.
    
    Args:
        metadata_path: Path to the BAIT metadata CSV file
        dataset_root: Root directory containing the model adapters
        base_models: Filter by specific base model(s) (e.g., ["meta-llama/Llama-2-7b-hf"])
        labels: Filter by label(s) (e.g., ["clean", "poison"])
        attacks: Filter by attack type(s) (e.g., ["clean", "cba"])
        datasets: Filter by training dataset(s) (e.g., ["alpaca", "self-instruct"])
        min_poison_rate: Minimum poison rate (inclusive)
        max_poison_rate: Maximum poison rate (inclusive)
        model_ids: Filter by specific model ID(s) (e.g., ["id-0001", "id-0005"])
    """
    dataset_root: str
    metadata_filename: str = "metadata.csv"
    base_models: Optional[List[str]] = None
    labels: Optional[List[str]] = None
    attacks: Optional[List[str]] = None
    datasets: Optional[List[str]] = None
    min_poison_rate: Optional[float] = None
    max_poison_rate: Optional[float] = None
    model_ids: Optional[List[str]] = None
    
    def load_dataset(self) -> "ModelDataset":
        """Load the BAIT dataset with the current configuration."""
        return BAITDataset(self)

base_model_map = {
    "meta-llama/Llama-2-7b-hf": "llama2_7b_chat_hf",
    "meta-llama/Meta-Llama-3-8B-Instruct": "Meta-Llama-3-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.2": "Mistral-7B-Instruct-v0.2",
}
class BAITDataset(ModelDataset):
    """
    BAIT dataset loader that inherits from LoRADataset.
    
    Loads BAIT model zoo metadata and creates a filtered dataset of LoRA adapters.
    
    Args:
        config: BAITDatasetConfig instance with metadata path and filter settings
    """
    
    # Label mapping for BAIT dataset: convert string labels to integers
    LABEL_MAP = {"clean": 0, "poison": 1}
    
    def __init__(self, config: BAITDatasetConfig):
        self.config = config
        
        # Load metadata
        dataset_root_expanded = os.path.expanduser(config.dataset_root)
        metadata_path = Path(dataset_root_expanded) / config.metadata_filename
        metadata_df = pd.read_csv(metadata_path)
        
        # Apply filters
        filtered_df = self._apply_filters(metadata_df)
        
        # Create SampleMeta list
        dataset_root = Path(dataset_root_expanded)
        samples = []
        for _, row in filtered_df.iterrows():
            adapter_path = dataset_root / "models" / row["model_id"] / "model"
            
            # Only include if adapter path exists
            if adapter_path.exists():
                # Convert string label to mapped string number for consistency
                label_str = str(self.LABEL_MAP.get(row["label"], row["label"]))
                sample = BAITSample(
                    sample_path=str(adapter_path),
                    label=int(label_str),
                    peft_adapter=True,
                    base_model=base_model_map[row["base_model"]],
                    dataset=row["dataset"]
                )
                samples.append(sample)
        
        # Initialize parent LoRADataset
        super().__init__(data=samples)
    
    def _apply_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all configured filters to the metadata dataframe."""
        filtered_df = df.copy()
        
        # Filter by model IDs
        if self.config.model_ids is not None:
            filtered_df = filtered_df[filtered_df["model_id"].isin(self.config.model_ids)]
        
        # Filter by base models
        if self.config.base_models is not None:
            filtered_df = filtered_df[filtered_df["base_model"].isin(self.config.base_models)]
        
        # Filter by labels
        if self.config.labels is not None:
            filtered_df = filtered_df[filtered_df["label"].isin(self.config.labels)]
        
        # Filter by attacks
        if self.config.attacks is not None:
            filtered_df = filtered_df[filtered_df["attack"].isin(self.config.attacks)]
        
        # Filter by datasets
        if self.config.datasets is not None:
            filtered_df = filtered_df[filtered_df["dataset"].isin(self.config.datasets)]
        
        # Filter by poison rate range
        if self.config.min_poison_rate is not None:
            filtered_df = filtered_df[filtered_df["poison_rate"] >= self.config.min_poison_rate]
        
        if self.config.max_poison_rate is not None:
            filtered_df = filtered_df[filtered_df["poison_rate"] <= self.config.max_poison_rate]
        
        return filtered_df
    
    def train_val_split(self, val_ratio: float, seed: int):
        """
        Perform a stratified train/val split grouped by (base_model, dataset) pairs.
        
        For each unique (base_model, dataset) combination, performs a stratified split
        preserving label distribution, then combines all train and validation splits
        respectively to form the final train and validation subsets.
        
        Args:
            val_ratio: Ratio of data to use for validation (0.0 to 1.0)
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (train_split: Subset, val_split: Subset)
        """
        # Group indices by (base_model, dataset) pair
        group_to_indices = defaultdict(list)
        for idx, sample_meta in enumerate(self.data):
            group_key = (sample_meta.base_model, sample_meta.dataset)
            group_to_indices[group_key].append(idx)
        
        all_train_indices = []
        all_val_indices = []
        
        # For each (base_model, dataset) group, perform stratified split
        for group_key, group_indices in group_to_indices.items():
            # Create a temporary subset dataset for this group
            group_data = [self.data[idx] for idx in group_indices]
            temp_dataset = ModelDataset(data=group_data)
            
            # Use stratified_split from base class
            # print(f"Splitting group {group_key} with {len(group_data)} and val_ratio={val_ratio}")
            group_train_indices, group_val_indices = stratified_split(
                temp_dataset, val_ratio, seed
            )
            
            # Map back to original dataset indices
            all_train_indices.extend([group_indices[i] for i in group_train_indices])
            all_val_indices.extend([group_indices[i] for i in group_val_indices])
        
        train_split = torch.utils.data.Subset(self, all_train_indices)
        val_split = torch.utils.data.Subset(self, all_val_indices)
        
        return train_split, val_split

