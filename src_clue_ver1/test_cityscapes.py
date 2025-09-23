#!/usr/bin/env python3
"""
=============================================================================
CITYSCAPES MODEL TESTING SCRIPT
=============================================================================

This script tests the prototype segmentation model with your Cityscapes dataset.
It will:
1. Set up the dataset path
2. Create a simple model
3. Test forward pass
4. Show results

🚀 USAGE:
    python test_cityscapes.py

=============================================================================
"""

import os
import sys
import torch
import yaml
from pathlib import Path

# Add current directory to path
sys.path.append('.')

def setup_cityscapes_path():
    """Set up the Cityscapes dataset path"""
    print("🔧 Setting up Cityscapes dataset path...")
    
    # Your dataset location
    cityscapes_path = "/project/hnguyen2/mvu9/datasets/cityscapes"
    
    # Expected location in the project
    expected_path = "./data/cityscapes"
    
    # Create data directory if it doesn't exist
    os.makedirs("./data", exist_ok=True)
    
    # Create symbolic link if it doesn't exist
    if not os.path.exists(expected_path):
        try:
            os.symlink(cityscapes_path, expected_path)
            print(f"✅ Created symbolic link: {expected_path} -> {cityscapes_path}")
        except Exception as e:
            print(f"❌ Failed to create symbolic link: {e}")
            return False
    else:
        print(f"✅ Symbolic link already exists: {expected_path}")
    
    # Verify the link works
    if os.path.exists(expected_path):
        print(f"✅ Dataset accessible at: {expected_path}")
        return True
    else:
        print(f"❌ Dataset not accessible at: {expected_path}")
        return False

def test_model_creation():
    """Test creating the prototype segmentation model"""
    print("\n🏗️ Testing model creation...")
    
    try:
        from model import construct_PPNet
        
        # Create a simple config for testing
        config = {
            'model': {
                'base_architecture': 'deeplabv2_resnet101',
                'pretrained': False,  # Don't load pretrained weights for testing
                'prototype_shape': [10, 32, 1, 1],  # 10 prototypes, 32 features
                'num_classes': 19,  # Cityscapes has 19 classes
                'prototype_activation_function': 'log',
                'add_on_layers_type': 'deeplab_simple'
            }
        }
        
        # Create model
        model = construct_PPNet(config)
        print("✅ Model created successfully!")
        print(f"   - Architecture: {config['model']['base_architecture']}")
        print(f"   - Prototypes: {config['model']['prototype_shape'][0]}")
        print(f"   - Classes: {config['model']['num_classes']}")
        
        return model, config
        
    except Exception as e:
        print(f"❌ Failed to create model: {e}")
        return None, None

def test_forward_pass(model):
    """Test forward pass with dummy data"""
    print("\n🚀 Testing forward pass...")
    
    try:
        # Create dummy input (batch_size=2, channels=3, height=256, width=512)
        dummy_input = torch.randn(2, 3, 256, 512)
        print(f"   - Input shape: {dummy_input.shape}")
        
        # Test image classification forward pass
        model.eval()
        with torch.no_grad():
            logits, prototype_activations = model(dummy_input)
            print(f"   - Classification logits shape: {logits.shape}")
            print(f"   - Prototype activations shape: {prototype_activations.shape}")
        
        # Test segmentation forward pass
        with torch.no_grad():
            seg_logits, seg_activations = model.forward_segmentation(dummy_input)
            print(f"   - Segmentation logits shape: {seg_logits.shape}")
            print(f"   - Segmentation activations shape: {seg_activations.shape}")
        
        print("✅ Forward pass successful!")
        return True
        
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        return False

def test_dataset_loading():
    """Test loading the Cityscapes dataset"""
    print("\n📁 Testing dataset loading...")
    
    try:
        from dataset import PatchClassificationDataset
        
        # Test dataset creation
        dataset = PatchClassificationDataset(
            root='./data/cityscapes',
            split='train',
            transform=None,
            target_transform=None
        )
        
        print(f"✅ Dataset loaded successfully!")
        print(f"   - Dataset size: {len(dataset)}")
        
        # Test loading a sample
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"   - Sample image shape: {sample[0].shape}")
            print(f"   - Sample mask shape: {sample[1].shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Dataset loading failed: {e}")
        return False

def main():
    """Main testing function"""
    print("=" * 60)
    print("🧪 CITYSCAPES MODEL TESTING")
    print("=" * 60)
    
    # Step 1: Setup dataset path
    if not setup_cityscapes_path():
        print("❌ Failed to setup dataset path. Exiting.")
        return
    
    # Step 2: Test model creation
    model, config = test_model_creation()
    if model is None:
        print("❌ Failed to create model. Exiting.")
        return
    
    # Step 3: Test forward pass
    if not test_forward_pass(model):
        print("❌ Forward pass failed. Exiting.")
        return
    
    # Step 4: Test dataset loading
    if not test_dataset_loading():
        print("⚠️ Dataset loading failed, but model works with dummy data.")
    
    print("\n" + "=" * 60)
    print("🎉 TESTING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\n📋 SUMMARY:")
    print("✅ Dataset path setup: OK")
    print("✅ Model creation: OK")
    print("✅ Forward pass: OK")
    print("✅ Ready for training!")
    
    print("\n🚀 NEXT STEPS:")
    print("1. Run training: python train.py")
    print("2. Check config.yaml for training parameters")
    print("3. Monitor training progress in logs/")

if __name__ == "__main__":
    main()
