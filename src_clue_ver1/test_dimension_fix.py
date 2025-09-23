#!/usr/bin/env python3
"""
Test script to verify the dimension fix for the L2 convolution
"""

import yaml
import torch
from model import construct_PPNet

def main():
    print("=" * 60)
    print("🧪 TESTING DIMENSION FIX")
    print("=" * 60)
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    print('✅ Config loaded successfully!')
    print(f'   Dataset: {config["dataset"]["name"]}')
    print(f'   Data path: {config["paths"]["data_dir"]}')
    print(f'   Classes: {config["model"]["num_classes"]}')
    print(f'   Prototypes: {config["model"]["prototype_shape"][0]}')
    print(f'   Feature dim: {config["model"]["prototype_shape"][1]}')
    
    # Create model
    model = construct_PPNet(config)
    print('✅ Model created successfully!')
    
    # Test forward pass with different input sizes
    test_sizes = [
        (1, 3, 256, 512),  # Small test
        (2, 3, 513, 513),  # Standard size
    ]
    
    for i, size in enumerate(test_sizes):
        print(f'\n🧪 Test {i+1}: Input size {size}')
        dummy_input = torch.randn(size)
        model.eval()
        
        try:
            with torch.no_grad():
                logits, activations = model.forward_segmentation(dummy_input)
                print(f'   ✅ Forward pass successful!')
                print(f'   📊 Logits shape: {logits.shape}')
                print(f'   📊 Activations shape: {activations.shape}')
        except Exception as e:
            print(f'   ❌ Forward pass failed: {e}')
            return False
    
    print('\n🎉 ALL TESTS PASSED!')
    print('🚀 The dimension fix is working correctly!')
    print('🚀 Ready for training with: python train.py')
    return True

if __name__ == "__main__":
    main()
