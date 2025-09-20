#!/usr/bin/env python3
"""
=============================================================================
QUICK DATASET DOWNLOADER
=============================================================================

Simple script to download datasets for prototype segmentation.
Supports PASCAL VOC 2012 and Cityscapes (if available).

USAGE:
    python download_datasets.py [dataset_name]
    
Examples:
    python download_datasets.py voc        # Download PASCAL VOC 2012
    python download_datasets.py cityscapes # Download Cityscapes (if available)
    python download_datasets.py            # Download VOC by default
=============================================================================
"""

import sys
import os
from torchvision.datasets import VOCSegmentation


def download_voc():
    """Download PASCAL VOC 2012 dataset"""
    print("🔽 Downloading PASCAL VOC 2012 dataset...")
    
    try:
        # Download training set
        print("📥 Downloading training set...")
        train = VOCSegmentation(root="./data", year="2012", image_set="train", download=True)
        print(f"✅ Training set: {len(train)} samples")
        
        # Download validation set  
        print("📥 Downloading validation set...")
        val = VOCSegmentation(root="./data", year="2012", image_set="val", download=True)
        print(f"✅ Validation set: {len(val)} samples")
        
        print(f"\n🎉 VOC 2012 downloaded successfully!")
        print(f"📁 Location: ./data/VOCdevkit/VOC2012/")
        print(f"📊 Total samples: {len(train) + len(val)}")
        
        return True
        
    except Exception as e:
        print(f"❌ VOC download failed: {e}")
        return False


def download_cityscapes():
    """Download Cityscapes dataset (if available)"""
    print("🔽 Cityscapes dataset download not implemented yet.")
    print("💡 You need to manually download Cityscapes from:")
    print("   https://www.cityscapes-dataset.com/")
    print("   Then place it in ./data/cityscapes/")
    return False


def main():
    """Main function"""
    dataset = sys.argv[1] if len(sys.argv) > 1 else "voc"
    
    print(f"🚀 Downloading {dataset.upper()} dataset...")
    print("=" * 50)
    
    if dataset.lower() in ["voc", "pascal", "voc2012"]:
        success = download_voc()
    elif dataset.lower() in ["cityscapes", "city"]:
        success = download_cityscapes()
    else:
        print(f"❌ Unknown dataset: {dataset}")
        print("💡 Supported datasets: voc, cityscapes")
        return
    
    if success:
        print("\n✅ Dataset download completed!")
        print("🔧 Next: Update config.yaml and run python train.py")
    else:
        print("\n❌ Dataset download failed!")


if __name__ == "__main__":
    main()
