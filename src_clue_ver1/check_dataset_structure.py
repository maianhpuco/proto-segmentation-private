#!/usr/bin/env python3
"""
Direct check of dataset structure without interactive input
"""

import os
import yaml

def main():
    print("=" * 60)
    print("🔍 CHECKING DATASET STRUCTURE")
    print("=" * 60)
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    data_dir = config['paths']['data_dir']
    print(f"Data directory: {data_dir}")
    
    # Check if directories exist
    leftImg8bit_dir = os.path.join(data_dir, 'leftImg8bit')
    gtFine_dir = os.path.join(data_dir, 'gtFine')
    
    print(f"\n📁 Directory Structure:")
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
            
            # Count images in first city
            if cities:
                first_city = os.path.join(train_dir, cities[0])
                images = [f for f in os.listdir(first_city) if f.endswith('.png')]
                print(f"Images in {cities[0]}: {len(images)}")
    
    if os.path.exists(gtFine_dir):
        train_gt_dir = os.path.join(gtFine_dir, 'train')
        val_gt_dir = os.path.join(gtFine_dir, 'val')
        print(f"gtFine train dir exists: {os.path.exists(train_gt_dir)}")
        print(f"gtFine val dir exists: {os.path.exists(val_gt_dir)}")
        
        if os.path.exists(train_gt_dir):
            cities = os.listdir(train_gt_dir)
            print(f"GT Train cities: {len(cities)} - {cities[:3]}...")
    
    print("\n" + "=" * 60)
    print("✅ Dataset structure check complete!")

if __name__ == "__main__":
    main()
