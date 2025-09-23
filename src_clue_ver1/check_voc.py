#!/usr/bin/env python3
"""
=============================================================================
VOC DATASET VERIFICATION SCRIPT
=============================================================================

This script checks if PASCAL VOC 2012 dataset has been downloaded successfully
and provides detailed information about the dataset structure and contents.

🎯 WHAT IT CHECKS:
- Dataset directory structure
- Number of images and annotations
- File integrity and accessibility
- Class distribution
- Image dimensions and formats

🚀 USAGE:
    python check_voc.py

=============================================================================
"""

import os
import sys
from pathlib import Path
import json
from PIL import Image
import numpy as np


def check_voc_structure():
    """Check if VOC dataset has the correct directory structure"""
    print("🔍 Checking VOC dataset structure...")
    print("="*60)
    
    # Expected VOC directory structure
    voc_base = Path("./data/VOCdevkit/VOC2012")
    
    if not voc_base.exists():
        print(f"❌ VOC base directory not found: {voc_base}")
        print("💡 Expected location: ./data/VOCdevkit/VOC2012/")
        return False
    
    print(f"✅ VOC base directory found: {voc_base}")
    
    # Check required subdirectories
    required_dirs = [
        "JPEGImages",      # Training and validation images
        "SegmentationClass",  # Segmentation masks
        "ImageSets/Segmentation"  # Train/val split files
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        full_path = voc_base / dir_path
        if full_path.exists():
            print(f"✅ Found: {dir_path}")
        else:
            print(f"❌ Missing: {dir_path}")
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        print(f"\n❌ Missing directories: {missing_dirs}")
        return False
    
    return True


def count_voc_files():
    """Count the number of images and annotations in VOC dataset"""
    print("\n📊 Counting VOC dataset files...")
    print("="*60)
    
    voc_base = Path("./data/VOCdevkit/VOC2012")
    
    # Count images
    jpeg_dir = voc_base / "JPEGImages"
    if jpeg_dir.exists():
        image_files = list(jpeg_dir.glob("*.jpg"))
        print(f"📸 Total images: {len(image_files)}")
    else:
        print("❌ JPEGImages directory not found")
        return False
    
    # Count segmentation masks
    seg_dir = voc_base / "SegmentationClass"
    if seg_dir.exists():
        mask_files = list(seg_dir.glob("*.png"))
        print(f"🎭 Total segmentation masks: {len(mask_files)}")
    else:
        print("❌ SegmentationClass directory not found")
        return False
    
    # Check train/val splits
    split_dir = voc_base / "ImageSets" / "Segmentation"
    if split_dir.exists():
        train_file = split_dir / "train.txt"
        val_file = split_dir / "val.txt"
        
        if train_file.exists():
            with open(train_file, 'r') as f:
                train_samples = len(f.read().strip().split('\n'))
            print(f"🚂 Training samples: {train_samples}")
        
        if val_file.exists():
            with open(val_file, 'r') as f:
                val_samples = len(f.read().strip().split('\n'))
            print(f"✅ Validation samples: {val_samples}")
    
    return True


def check_voc_classes():
    """Check VOC class information and distribution"""
    print("\n🏷️ Checking VOC classes...")
    print("="*60)
    
    # VOC 2012 classes
    voc_classes = [
        'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
        'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
        'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]
    
    print(f"📋 VOC 2012 has {len(voc_classes)} classes:")
    for i, class_name in enumerate(voc_classes):
        print(f"   {i:2d}: {class_name}")
    
    return voc_classes


def check_sample_images():
    """Check a few sample images to verify they can be loaded"""
    print("\n🖼️ Checking sample images...")
    print("="*60)
    
    voc_base = Path("./data/VOCdevkit/VOC2012")
    jpeg_dir = voc_base / "JPEGImages"
    seg_dir = voc_base / "SegmentationClass"
    
    if not jpeg_dir.exists() or not seg_dir.exists():
        print("❌ Image directories not found")
        return False
    
    # Get first few image files
    image_files = list(jpeg_dir.glob("*.jpg"))[:3]
    
    if not image_files:
        print("❌ No image files found")
        return False
    
    for i, img_file in enumerate(image_files):
        try:
            # Check image
            img = Image.open(img_file)
            print(f"✅ Sample {i+1}: {img_file.name}")
            print(f"   Size: {img.size} (W×H)")
            print(f"   Mode: {img.mode}")
            
            # Check corresponding mask
            mask_name = img_file.stem + ".png"
            mask_file = seg_dir / mask_name
            
            if mask_file.exists():
                mask = Image.open(mask_file)
                print(f"   Mask: {mask_name} - Size: {mask.size}, Mode: {mask.mode}")
                
                # Check unique values in mask (should be class indices)
                mask_array = np.array(mask)
                unique_values = np.unique(mask_array)
                print(f"   Classes in mask: {len(unique_values)} unique values")
                print(f"   Value range: {unique_values.min()} - {unique_values.max()}")
            else:
                print(f"   ⚠️ No mask found: {mask_name}")
            
            print()
            
        except Exception as e:
            print(f"❌ Error loading {img_file.name}: {e}")
            return False
    
    return True


def check_dataset_compatibility():
    """Check if the dataset is compatible with the prototype segmentation code"""
    print("\n🔧 Checking dataset compatibility...")
    print("="*60)
    
    # Check if we can import the dataset class
    try:
        from dataset import PatchClassificationDataset
        print("✅ Dataset class imported successfully")
    except ImportError as e:
        print(f"❌ Cannot import dataset class: {e}")
        return False
    
    # Try to create a small dataset instance
    try:
        # Create a minimal config for testing
        test_config = {
            'dataset': {
                'name': 'pascal',
                'root': './data',
                'window_size': [64, 64],
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225]
            }
        }
        
        # Try to create dataset (this will test if files are accessible)
        dataset = PatchClassificationDataset(test_config, split='train')
        print(f"✅ Dataset created successfully")
        print(f"   Dataset size: {len(dataset)} samples")
        
        # Try to load one sample
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"✅ Sample loaded successfully")
            print(f"   Image shape: {sample['image'].shape}")
            print(f"   Mask shape: {sample['mask'].shape}")
        
    except Exception as e:
        print(f"❌ Error creating dataset: {e}")
        return False
    
    return True


def generate_voc_report():
    """Generate a comprehensive VOC dataset report"""
    print("\n📋 VOC Dataset Report")
    print("="*60)
    
    voc_base = Path("./data/VOCdevkit/VOC2012")
    
    if not voc_base.exists():
        print("❌ VOC dataset not found!")
        return
    
    # Calculate total size
    total_size = 0
    for file_path in voc_base.rglob("*"):
        if file_path.is_file():
            total_size += file_path.stat().st_size
    
    total_size_mb = total_size / (1024 * 1024)
    print(f"📦 Total dataset size: {total_size_mb:.1f} MB")
    
    # Count files by type
    jpeg_count = len(list((voc_base / "JPEGImages").glob("*.jpg")))
    png_count = len(list((voc_base / "SegmentationClass").glob("*.png")))
    
    print(f"📸 JPEG images: {jpeg_count}")
    print(f"🎭 PNG masks: {png_count}")
    
    # Check if ready for training
    print(f"\n🚀 Ready for training: {'✅ YES' if jpeg_count > 0 and png_count > 0 else '❌ NO'}")


def main():
    """Main function to run all VOC checks"""
    print("🔍 PASCAL VOC 2012 Dataset Verification")
    print("="*60)
    
    # Run all checks
    checks = [
        ("Directory Structure", check_voc_structure),
        ("File Counts", count_voc_files),
        ("Class Information", check_voc_classes),
        ("Sample Images", check_sample_images),
        ("Dataset Compatibility", check_dataset_compatibility),
    ]
    
    results = {}
    for check_name, check_func in checks:
        try:
            result = check_func()
            results[check_name] = result
        except Exception as e:
            print(f"❌ Error in {check_name}: {e}")
            results[check_name] = False
    
    # Generate final report
    generate_voc_report()
    
    # Summary
    print("\n📊 VERIFICATION SUMMARY")
    print("="*60)
    
    all_passed = True
    for check_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{check_name:20s}: {status}")
        if not result:
            all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("🎉 VOC dataset verification PASSED!")
        print("✅ Your VOC dataset is ready for prototype segmentation training!")
        print("\n🚀 Next steps:")
        print("   1. Update config.yaml: dataset.name = 'pascal'")
        print("   2. Run: python train.py")
    else:
        print("❌ VOC dataset verification FAILED!")
        print("💡 Please check the errors above and re-download if necessary.")
        print("\n🔧 Troubleshooting:")
        print("   1. Check if download completed successfully")
        print("   2. Verify file permissions")
        print("   3. Try re-downloading the dataset")


if __name__ == "__main__":
    main()
