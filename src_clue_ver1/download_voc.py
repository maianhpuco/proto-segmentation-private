"""
=============================================================================
PASCAL VOC 2012 DATASET DOWNLOADER
=============================================================================

This script downloads and sets up the PASCAL VOC 2012 dataset for prototype segmentation.
It downloads both training and validation splits and organizes them properly.

USAGE:
    python download_voc.py

The dataset will be downloaded to ./data/VOCdevkit/VOC2012/
=============================================================================
"""

import os
import sys
from torchvision.datasets import VOCSegmentation
from torchvision import transforms
import torch
from PIL import Image
import numpy as np


def download_voc_dataset(data_dir="./data"):
    """
    Download PASCAL VOC 2012 dataset
    
    Args:
        data_dir: Directory to download dataset to
    """
    print("🚀 Starting PASCAL VOC 2012 dataset download...")
    print("=" * 60)
    
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    try:
        # Download training set
        print("📥 Downloading training set...")
        train_dataset = VOCSegmentation(
            root=data_dir, 
            year="2012", 
            image_set="train", 
            download=True
        )
        print(f"✅ Training set downloaded: {len(train_dataset)} samples")
        
        # Download validation set
        print("📥 Downloading validation set...")
        val_dataset = VOCSegmentation(
            root=data_dir, 
            year="2012", 
            image_set="val", 
            download=True
        )
        print(f"✅ Validation set downloaded: {len(val_dataset)} samples")
        
        # Print summary
        print("\n📊 Dataset Summary:")
        print(f"   - Training samples: {len(train_dataset)}")
        print(f"   - Validation samples: {len(val_dataset)}")
        print(f"   - Total samples: {len(train_dataset) + len(val_dataset)}")
        print(f"   - Dataset location: {os.path.abspath(data_dir)}/VOCdevkit/VOC2012/")
        
        return train_dataset, val_dataset
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return None, None


def test_dataset_loading(train_dataset, val_dataset):
    """
    Test loading a sample from the dataset
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
    """
    print("\n🧪 Testing dataset loading...")
    
    try:
        # Test training sample
        if train_dataset:
            image, mask = train_dataset[0]
            print(f"✅ Training sample loaded:")
            print(f"   - Image size: {image.size}")
            print(f"   - Mask size: {mask.size}")
            print(f"   - Image mode: {image.mode}")
            print(f"   - Mask mode: {mask.mode}")
        
        # Test validation sample
        if val_dataset:
            image, mask = val_dataset[0]
            print(f"✅ Validation sample loaded:")
            print(f"   - Image size: {image.size}")
            print(f"   - Mask size: {mask.size}")
            
        return True
        
    except Exception as e:
        print(f"❌ Dataset loading test failed: {e}")
        return False


def check_dataset_structure(data_dir="./data"):
    """
    Check the downloaded dataset structure
    
    Args:
        data_dir: Directory containing the dataset
    """
    print("\n📁 Checking dataset structure...")
    
    voc_path = os.path.join(data_dir, "VOCdevkit", "VOC2012")
    
    if not os.path.exists(voc_path):
        print(f"❌ VOC dataset not found at {voc_path}")
        return False
    
    # Check key directories
    key_dirs = [
        "JPEGImages",
        "SegmentationClass", 
        "ImageSets/Segmentation"
    ]
    
    for dir_name in key_dirs:
        dir_path = os.path.join(voc_path, dir_name)
        if os.path.exists(dir_path):
            file_count = len(os.listdir(dir_path))
            print(f"✅ {dir_name}: {file_count} files")
        else:
            print(f"❌ {dir_name}: Not found")
            return False
    
    return True


def create_sample_visualization(train_dataset, val_dataset, output_dir="./samples"):
    """
    Create sample visualizations of the dataset
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        output_dir: Directory to save sample images
    """
    print("\n🎨 Creating sample visualizations...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Save training sample
        if train_dataset:
            image, mask = train_dataset[0]
            image.save(os.path.join(output_dir, "train_sample_image.jpg"))
            mask.save(os.path.join(output_dir, "train_sample_mask.png"))
            print(f"✅ Training sample saved to {output_dir}/")
        
        # Save validation sample
        if val_dataset:
            image, mask = val_dataset[0]
            image.save(os.path.join(output_dir, "val_sample_image.jpg"))
            mask.save(os.path.join(output_dir, "val_sample_mask.png"))
            print(f"✅ Validation sample saved to {output_dir}/")
            
        return True
        
    except Exception as e:
        print(f"❌ Sample visualization failed: {e}")
        return False


def main():
    """Main function"""
    print("🔽 PASCAL VOC 2012 Dataset Downloader")
    print("=" * 60)
    
    # Download dataset
    train_dataset, val_dataset = download_voc_dataset()
    
    if train_dataset is None or val_dataset is None:
        print("❌ Dataset download failed. Exiting.")
        return
    
    # Test dataset loading
    if not test_dataset_loading(train_dataset, val_dataset):
        print("❌ Dataset loading test failed. Exiting.")
        return
    
    # Check dataset structure
    if not check_dataset_structure():
        print("❌ Dataset structure check failed. Exiting.")
        return
    
    # Create sample visualizations
    create_sample_visualization(train_dataset, val_dataset)
    
    print("\n🎉 Dataset download and setup completed successfully!")
    print("=" * 60)
    print("📚 Next steps:")
    print("   1. Update config.yaml to use 'pascal' dataset")
    print("   2. Run: python train.py")
    print("   3. Or test with: python example_usage.py")
    print("\n💡 The dataset is now ready for prototype segmentation training!")


if __name__ == "__main__":
    main()
