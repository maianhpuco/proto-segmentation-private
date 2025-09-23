#!/usr/bin/env python3
"""
Simple test script to check if dataset can find Cityscapes data
"""

import yaml
import os
from dataset import PatchClassificationDataset

def main():
    print("=" * 60)
    print("🧪 TESTING DATASET LOADING")
    print("=" * 60)
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    data_dir = config['paths']['data_dir']
    print(f"Data directory: {data_dir}")
    
    # Check if directories exist
    leftImg8bit_dir = os.path.join(data_dir, 'leftImg8bit')
    gtFine_dir = os.path.join(data_dir, 'gtFine')
    
    print(f"leftImg8bit exists: {os.path.exists(leftImg8bit_dir)}")
    print(f"gtFine exists: {os.path.exists(gtFine_dir)}")
    
    if os.path.exists(leftImg8bit_dir):
        train_dir = os.path.join(leftImg8bit_dir, 'train')
        val_dir = os.path.join(leftImg8bit_dir, 'val')
        print(f"train dir exists: {os.path.exists(train_dir)}")
        print(f"val dir exists: {os.path.exists(val_dir)}")
        
        if os.path.exists(train_dir):
            cities = os.listdir(train_dir)
            print(f"Train cities: {len(cities)} - {cities[:3]}...")
    
    # Test dataset creation
    try:
        print("\nTesting train dataset...")
        train_dataset = PatchClassificationDataset('train', config, data_dir)
        print(f"✅ Train dataset: {len(train_dataset)} samples")
    except Exception as e:
        print(f"❌ Train dataset error: {e}")
    
    try:
        print("\nTesting val dataset...")
        val_dataset = PatchClassificationDataset('val', config, data_dir)
        print(f"✅ Val dataset: {len(val_dataset)} samples")
    except Exception as e:
        print(f"❌ Val dataset error: {e}")

if __name__ == "__main__":
    main()
