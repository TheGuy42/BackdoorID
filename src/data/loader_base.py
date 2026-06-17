from typing import Optional, List
from collections import defaultdict
import lightning as L
import torch
from torch.utils.data import DataLoader, Subset
import random
from src.data.dataset_base import ModelDataset, ModelSample


def collate_samples(batch: List[ModelSample]) -> List[ModelSample]:
    """
    Custom collate function that returns a list of ModelSample objects.
    
    Args:
        batch: List of ModelSample objects from the dataset
        
    Returns:
        List of ModelSample objects (no batching)
    """
    return batch


class ModelDataModule(L.LightningDataModule):
    """
    PyTorch Lightning DataModule for ModelDataset.
    
    Args:
        train_dataset: Training dataset (will be split into train/val if val_dataset is not provided)
        val_dataset: Validation dataset (optional). If provided, no splitting is done.
        val_ratio: Ratio of data to use for validation (0.0 to 1.0). Only used if val_dataset is None.
        test_dataset: Test dataset (optional)
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes for data loading
        seed: Random seed for reproducible train/val split. Only used if val_dataset is None.
    """
    
    def __init__(
        self,
        train_dataset: ModelDataset,
        val_dataset: Optional[ModelDataset] = None,
        val_ratio: float = 0.1,
        test_dataset: Optional[ModelDataset] = None,
        batch_size: int = 32,
        num_workers: int = 4,
        seed: int = 42,
    ):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.val_ratio = val_ratio
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        
        # If validation dataset is provided, use it directly without splitting
        if val_dataset is not None:
            self.train_split = train_dataset
            self.val_split = val_dataset
        else:
            # Perform stratified split to preserve label distribution
            self.train_split, self.val_split = self.train_dataset.train_val_split(val_ratio, seed)
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_split,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
            collate_fn=collate_samples,
        )
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_split,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
            collate_fn=collate_samples,
        )
    
    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            raise ValueError("Test dataset not provided")
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
            collate_fn=collate_samples,
        )
