"""
Pytorch Lightning DataModule for training prototype segmentation model on Cityscapes and SUN datasets
"""
import multiprocessing
import os

import gin
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from segmentation.dataset import PatchClassificationDataset
from settings import data_path


# Try this out in case of high RAM usage:
# import torch.multiprocessing
# torch.multiprocessing.set_sharing_strategy('file_system')


# noinspection PyAbstractClass
@gin.configurable(denylist=['batch_size'])
class PatchClassificationDataModule(LightningDataModule):
    """
    =============================================================================
    PATCHCLASSIFICATIONDATAMODULE - PYTORCH LIGHTNING DATAMODULE
    =============================================================================
    
    PURPOSE:
        Manages data loading for prototype-based semantic segmentation training.
        This is the main data management class that handles all data operations.
    
    KEY RESPONSIBILITIES:
        1. Data Loading: Manages train/val/test data splits
        2. Data Processing: Handles augmentation, normalization, and preprocessing
        3. Batch Management: Creates DataLoaders with proper configuration
        4. Multiprocessing: Manages parallel data loading for efficiency
        5. Special Modes: Provides prototype pushing dataloader
    
    PATCH CLASSIFICATION APPROACH:
        - Treats semantic segmentation as patch classification
        - Each image is divided into patches/regions
        - Each patch gets classified into a semantic class
        - Model learns prototypes that represent characteristic features of each class
        - Prototypes are matched against image patches for classification
    
    DATASET SUPPORT:
        - Cityscapes: Urban scene segmentation (19 classes)
        - PASCAL VOC: Object segmentation (21 classes)
        - Custom datasets with proper preprocessing
    
    =============================================================================
    """
    def __init__(self, batch_size: int, dataloader_n_jobs: int = gin.REQUIRED,
                 train_key: str = 'train'):
        super().__init__()
        
        # Configure multiprocessing for data loading
        # Use all CPU cores if dataloader_n_jobs = -1, otherwise use specified number
        self.dataloader_n_jobs = dataloader_n_jobs if dataloader_n_jobs != -1 else multiprocessing.cpu_count()
        
        # Batch size for training (can be different for different phases)
        self.batch_size = batch_size
        
        # Which split to use for training (usually 'train')
        self.train_key = train_key

    def prepare_data(self):
        """
        =============================================================================
        DATA PREPARATION CHECK
        =============================================================================
        Purpose: Verify that the dataset has been downloaded and preprocessed
        Checks: Whether annotation files exist in the expected location
        Error: Raises ValueError if data is not properly prepared
        =============================================================================
        """
        if not os.path.exists(os.path.join(data_path, 'annotations')):
            raise ValueError("Please download dataset and preprocess it using 'preprocess.py' script")

    def get_data_loader(self, dataset: PatchClassificationDataset, **kwargs) -> DataLoader:
        """
        =============================================================================
        DATALOADER CREATION
        =============================================================================
        Purpose: Create PyTorch DataLoader with proper configuration
        Features:
            - Automatic shuffling for training (not for evaluation)
            - Multiprocessing for efficient data loading
            - Flexible batch size configuration
            - Proper dataset handling
        =============================================================================
        """
        if 'batch_size' in kwargs:
            # Use custom batch size if provided
            return DataLoader(
                dataset=dataset,
                shuffle=not dataset.is_eval,  # Shuffle only for training
                num_workers=self.dataloader_n_jobs,  # Parallel data loading
                **kwargs
            )
        # Use default batch size
        return DataLoader(
            dataset=dataset,
            shuffle=not dataset.is_eval,  # Shuffle only for training
            num_workers=self.dataloader_n_jobs,  # Parallel data loading
            batch_size=self.batch_size,  # Default batch size
            **kwargs
        )

    def train_dataloader(self, **kwargs):
        """
        =============================================================================
        TRAINING DATALOADER
        =============================================================================
        Purpose: Create dataloader for training phase
        Features:
            - Uses training split of the dataset
            - Enables data augmentation (is_eval=False)
            - Shuffles data for better training
            - Applies normalization and preprocessing
        =============================================================================
        """
        train_split = PatchClassificationDataset(
            split_key=self.train_key,  # Usually 'train'
            is_eval=False,             # Enable data augmentation
        )
        return self.get_data_loader(train_split, **kwargs)

    def val_dataloader(self, **kwargs):
        """
        =============================================================================
        VALIDATION DATALOADER
        =============================================================================
        Purpose: Create dataloader for validation phase
        Features:
            - Uses validation split of the dataset
            - Disables data augmentation (is_eval=True)
            - No shuffling for consistent evaluation
            - Only applies normalization (no augmentation)
        =============================================================================
        """
        val_split = PatchClassificationDataset(
            split_key='val',    # Validation split
            is_eval=True,       # Disable data augmentation
        )
        return self.get_data_loader(val_split, **kwargs)

    def test_dataloader(self, **kwargs):
        """
        =============================================================================
        TEST DATALOADER
        =============================================================================
        Purpose: Create dataloader for testing phase
        Note: Uses validation split since Cityscapes doesn't have public test labels
        Features:
            - Uses validation split (no public test set for Cityscapes)
            - Disables data augmentation (is_eval=True)
            - No shuffling for consistent evaluation
            - Only applies normalization
        =============================================================================
        """
        test_split = PatchClassificationDataset(
            split_key='val',    # Use val split (no public test set for Cityscapes)
            is_eval=True,       # Disable data augmentation
        )
        return self.get_data_loader(test_split, **kwargs)

    def train_push_dataloader(self, **kwargs):
        """
        =============================================================================
        PROTOTYPE PUSHING DATALOADER
        =============================================================================
        Purpose: Create special dataloader for prototype pushing phase
        Features:
            - Uses training data to find best prototype matches
            - Disables data augmentation (is_eval=True) for consistent matching
            - Enables prototype pushing mode (push_prototypes=True)
            - No shuffling for deterministic prototype finding
        =============================================================================
        """
        train_split = PatchClassificationDataset(
            split_key='train',      # Use training data
            is_eval=True,           # Disable augmentation for consistent matching
            push_prototypes=True    # Enable prototype pushing mode
        )
        return self.get_data_loader(train_split, **kwargs)
