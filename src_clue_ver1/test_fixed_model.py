#!/usr/bin/env python3
"""
Simple test script for the fixed prototype segmentation model
"""

import yaml
import torch
from model import construct_PPNet

def main():
    print("=" * 60)
    print("🧪 TESTING FIXED MODEL")
    print("=" * 60)
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    print('✅ Config loaded successfully!')
    print(f'   Dataset: {config["dataset"]["name"]}')
    print(f'   Data path: {config["paths"]["data_dir"]}')
    print(f'   Classes: {config["model"]["num_classes"]}')
    
    # Create model
    model = construct_PPNet(config)
    print('✅ Model created successfully!')
    
    # Test forward pass
    dummy_input = torch.randn(1, 3, 256, 512)
    model.eval()
    with torch.no_grad():
        logits, activations = model.forward_segmentation(dummy_input)
        print(f'✅ Forward pass successful: {logits.shape} -> {activations.shape}')
    
    print('🎉 All tests passed! Ready for training.')
    print('🚀 To start training, run: python train.py')

if __name__ == "__main__":
    main()
