#!/usr/bin/env python3
"""
Simple test script for prototype segmentation model with Cityscapes dataset
"""

import os
import sys
import torch
import yaml

# Add current directory to path
sys.path.append('.')

def main():
    print("=" * 60)
    print("🧪 SIMPLE CITYSCAPES MODEL TEST")
    print("=" * 60)
    
    # Step 1: Load config
    print("📋 Loading configuration...")
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        print(f"✅ Config loaded successfully!")
        print(f"   - Dataset: {config['dataset']['name']}")
        print(f"   - Data path: {config['paths']['data_dir']}")
        print(f"   - Classes: {config['model']['num_classes']}")
        print(f"   - Prototypes: {config['model']['prototype_shape'][0]}")
    except Exception as e:
        print(f"❌ Failed to load config: {e}")
        return
    
    # Step 2: Test model creation
    print("\n🏗️ Testing model creation...")
    try:
        from model import construct_PPNet
        model = construct_PPNet(config)
        print("✅ Model created successfully!")
    except Exception as e:
        print(f"❌ Failed to create model: {e}")
        return
    
    # Step 3: Test forward pass
    print("\n🚀 Testing forward pass...")
    try:
        # Create dummy input
        dummy_input = torch.randn(1, 3, 256, 512)
        print(f"   - Input shape: {dummy_input.shape}")
        
        # Test forward pass
        model.eval()
        with torch.no_grad():
            logits, activations = model.forward_segmentation(dummy_input)
            print(f"   - Output logits shape: {logits.shape}")
            print(f"   - Prototype activations shape: {activations.shape}")
        
        print("✅ Forward pass successful!")
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        return
    
    # Step 4: Check dataset path
    print("\n📁 Checking dataset path...")
    dataset_path = config['paths']['data_dir']
    if os.path.exists(dataset_path):
        print(f"✅ Dataset path exists: {dataset_path}")
        
        # Check subdirectories
        leftImg8bit = os.path.join(dataset_path, 'leftImg8bit')
        gtFine = os.path.join(dataset_path, 'gtFine')
        
        if os.path.exists(leftImg8bit):
            print(f"✅ Images directory found: {leftImg8bit}")
        else:
            print(f"❌ Images directory missing: {leftImg8bit}")
            
        if os.path.exists(gtFine):
            print(f"✅ Annotations directory found: {gtFine}")
        else:
            print(f"❌ Annotations directory missing: {gtFine}")
    else:
        print(f"❌ Dataset path does not exist: {dataset_path}")
    
    print("\n" + "=" * 60)
    print("🎉 TEST COMPLETED!")
    print("=" * 60)
    print("\n📋 SUMMARY:")
    print("✅ Configuration: OK")
    print("✅ Model creation: OK") 
    print("✅ Forward pass: OK")
    print("✅ Ready for training!")
    
    print(f"\n📁 Your dataset is at: {dataset_path}")
    print("\n🚀 To start training, run: python train.py")

if __name__ == "__main__":
    main()
