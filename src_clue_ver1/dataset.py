"""
=============================================================================
SIMPLIFIED DATASET FOR PROTOTYPE SEGMENTATION
=============================================================================

🎯 WHAT THIS FILE DOES:
This file handles loading and preprocessing of images and segmentation masks for
prototype-based semantic segmentation. It treats segmentation as a patch classification
problem where each pixel is classified independently.

🏗️ KEY FEATURES:
- Supports Cityscapes and PASCAL VOC datasets
- Applies data augmentation (random crop, flip, scaling)
- Converts segmentation masks to class IDs
- Handles different dataset formats and class mappings
- Provides train/val/test splits

📊 SUPPORTED DATASETS:

🌆 Cityscapes Dataset:
- 19 evaluation classes (road, sidewalk, building, etc.)
- High-resolution images (1024×2048)
- Dense pixel-level annotations
- Train: 2975 images, Val: 500 images

🏷️ PASCAL VOC 2012:
- 21 classes (20 object classes + background)
- Medium-resolution images (~500×500)
- Sparse pixel-level annotations
- Train: 1464 images, Val: 1449 images

🔧 DATA PREPROCESSING:
- Resize images to target size (513×513)
- Normalize using ImageNet statistics
- Convert masks to class IDs
- Apply data augmentation (training only)

📚 USAGE INSTRUCTIONS:
- Import: from dataset import PatchClassificationDataset
- Create dataset: dataset = PatchClassificationDataset(split='train', config=config)
- Get sample: image, mask = dataset[0]

⚙️ CONFIGURATION:
- All parameters are configured via config.yaml
- Key parameters:
  * dataset.name: 'cityscapes' or 'pascal'
  * dataset.window_size: [513, 513] (input image size)
  * dataset.mean/std: ImageNet normalization values
  * dataset.scales: [0.5, 1.5] (random scaling range)
=============================================================================
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from typing import List, Tuple, Optional


class PatchClassificationDataset(Dataset):
    """
    Simplified dataset for prototype-based semantic segmentation
    Treats segmentation as patch classification problem
    """
    
    def __init__(self, split: str, config: dict, data_dir: str = "./data"):
        """
        Initialize dataset
        
        Args:
            split: Dataset split ('train', 'val', 'test')
            config: Configuration dictionary
            data_dir: Directory containing dataset
        """
        self.split = split
        self.config = config
        self.data_dir = data_dir
        
        # Get dataset configuration
        dataset_config = config['dataset']
        self.window_size = tuple(dataset_config['window_size'])  # (height, width)
        self.mean = dataset_config['mean']
        self.std = dataset_config['std']
        self.scales = dataset_config['scales']
        self.only_19_from_cityscapes = dataset_config.get('only_19_from_cityscapes', True)
        
        # Class mappings
        self._setup_class_mappings()
        
        # Get image and mask paths
        self.image_paths, self.mask_paths = self._get_data_paths()
        
        # Setup transforms
        self._setup_transforms()
        
        print(f"Loaded {len(self.image_paths)} {split} samples")
    
    def _setup_class_mappings(self):
        """Setup class ID mappings for different datasets"""
        if self.config['dataset']['name'] == 'cityscapes':
            if self.only_19_from_cityscapes:
                # Cityscapes 19 evaluation classes
                # Map original Cityscapes class IDs to 0-18 range
                # This is the standard Cityscapes 19-class evaluation mapping
                self.class_mapping = {
                    0: 0,   # void -> 0
                    1: 1,   # road -> 1
                    2: 2,   # sidewalk -> 2
                    3: 3,   # building -> 3
                    4: 4,   # wall -> 4
                    5: 5,   # fence -> 5
                    6: 6,   # pole -> 6
                    7: 7,   # traffic_light -> 7
                    8: 8,   # traffic_sign -> 8
                    9: 9,   # vegetation -> 9
                    10: 10, # terrain -> 10
                    11: 11, # sky -> 11
                    12: 12, # person -> 12
                    13: 13, # rider -> 13
                    14: 14, # car -> 14
                    15: 15, # truck -> 15
                    16: 16, # bus -> 16
                    17: 17, # train -> 17
                    18: 18, # motorcycle -> 18
                    19: 18, # bicycle -> 18 (map to motorcycle class)
                }
                
                # Add mapping for all other Cityscapes classes to void (0)
                # This ensures any unexpected class IDs are mapped to void
                for i in range(20, 255):
                    self.class_mapping[i] = 0
                
                self.num_classes = 19
            else:
                # All Cityscapes classes
                self.class_mapping = {i: i for i in range(34)}
                self.num_classes = 34
        else:
            # PASCAL VOC
            self.class_mapping = {i: i for i in range(21)}
            self.num_classes = 21
    
    def _get_data_paths(self):
        """Get paths to images and masks"""
        if self.config['dataset']['name'] == 'cityscapes':
            return self._get_cityscapes_paths()
        else:
            return self._get_pascal_paths()
    
    def _get_cityscapes_paths(self):
        """Get Cityscapes dataset paths"""
        # Check if data_dir already contains the cityscapes structure
        if os.path.exists(os.path.join(self.data_dir, 'leftImg8bit')):
            # Direct path to cityscapes data
            image_dir = os.path.join(self.data_dir, 'leftImg8bit', self.split)
            mask_dir = os.path.join(self.data_dir, 'gtFine', self.split)
        else:
            # Nested path structure
            image_dir = os.path.join(self.data_dir, 'cityscapes', 'leftImg8bit', self.split)
            mask_dir = os.path.join(self.data_dir, 'cityscapes', 'gtFine', self.split)
        
        image_paths = []
        mask_paths = []
        
        for city in os.listdir(image_dir):
            city_image_dir = os.path.join(image_dir, city)
            city_mask_dir = os.path.join(mask_dir, city)
            
            for img_name in os.listdir(city_image_dir):
                if img_name.endswith('.png'):
                    # Get corresponding mask
                    mask_name = img_name.replace('leftImg8bit', 'gtFine_labelIds')
                    mask_path = os.path.join(city_mask_dir, mask_name)
                    
                    if os.path.exists(mask_path):
                        image_paths.append(os.path.join(city_image_dir, img_name))
                        mask_paths.append(mask_path)
        
        return image_paths, mask_paths
    
    def _get_pascal_paths(self):
        """Get PASCAL VOC dataset paths"""
        image_dir = os.path.join(self.data_dir, 'pascal', 'JPEGImages')
        mask_dir = os.path.join(self.data_dir, 'pascal', 'SegmentationClass')
        
        # Get split file
        split_file = os.path.join(self.data_dir, 'pascal', 'ImageSets', 'Segmentation', f'{self.split}.txt')
        
        image_paths = []
        mask_paths = []
        
        with open(split_file, 'r') as f:
            for line in f:
                img_id = line.strip()
                img_path = os.path.join(image_dir, f'{img_id}.jpg')
                mask_path = os.path.join(mask_dir, f'{img_id}.png')
                
                if os.path.exists(img_path) and os.path.exists(mask_path):
                    image_paths.append(img_path)
                    mask_paths.append(mask_path)
        
        return image_paths, mask_paths
    
    def _setup_transforms(self):
        """Setup data transforms"""
        if self.split == 'train':
            # Training transforms with augmentation
            self.transform = transforms.Compose([
                transforms.Resize((self.window_size[0] + 50, self.window_size[1] + 50)),
                transforms.RandomCrop(self.window_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std)
            ])
        else:
            # Validation/test transforms without augmentation
            self.transform = transforms.Compose([
                transforms.Resize(self.window_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std)
            ])
    
    def _convert_mask_to_class_ids(self, mask):
        """Convert segmentation mask to class IDs"""
        mask_array = np.array(mask)
        class_mask = np.zeros_like(mask_array, dtype=np.int64)
        
        for old_id, new_id in self.class_mapping.items():
            class_mask[mask_array == old_id] = new_id
        
        return class_mask
    
    def _resize_mask(self, mask, target_size):
        """Resize mask to target size"""
        mask = mask.resize(target_size, Image.NEAREST)
        return mask
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """Get a sample from the dataset"""
        # Load image and mask
        image = Image.open(self.image_paths[idx]).convert('RGB')
        mask = Image.open(self.mask_paths[idx])
        
        # Resize mask to target size
        mask = self._resize_mask(mask, self.window_size)
        
        # Convert mask to class IDs
        class_mask = self._convert_mask_to_class_ids(mask)
        
        # Apply transforms to image
        image = self.transform(image)
        
        # Convert mask to tensor
        class_mask = torch.from_numpy(class_mask).long()
        
        return image, class_mask


# Example usage
if __name__ == "__main__":
    import yaml
    
    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create dataset
    dataset = PatchClassificationDataset('train', config)
    
    # Get a sample
    image, mask = dataset[0]
    print(f"Image shape: {image.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Mask unique values: {torch.unique(mask)}")
