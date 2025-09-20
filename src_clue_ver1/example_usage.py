"""
=============================================================================
EXAMPLE USAGE OF SIMPLIFIED PROTOTYPE SEGMENTATION
=============================================================================

This script demonstrates how to use the simplified prototype segmentation code.
It shows how to:
1. Load configuration
2. Create model and dataset
3. Run training
4. Test inference

HOW TO RUN:
python example_usage.py

This will create a small example with dummy data to demonstrate the pipeline.
=============================================================================
"""

import torch
import yaml
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from model import construct_PPNet
from module import PatchClassificationModule
from dataset import PatchClassificationDataset


def create_dummy_config():
    """Create a dummy configuration for testing"""
    config = {
        'model': {
            'base_architecture': 'deeplabv2_resnet101',
            'pretrained': False,  # Use False for faster testing
            'prototype_shape': [10, 32, 1, 1],  # Smaller for testing
            'num_classes': 3,
            'prototype_activation_function': 'log',
            'add_on_layers_type': 'deeplab_simple'
        },
        'training': {
            'warmup_steps': 10,
            'joint_steps': 20,
            'finetune_steps': 10,
            'warmup_batch_size': 2,
            'joint_batch_size': 2,
            'warmup_lr_add_on_layers': 0.001,
            'warmup_lr_prototype_vectors': 0.001,
            'warmup_weight_decay': 0.0001,
            'joint_lr_features': 0.0001,
            'joint_lr_add_on_layers': 0.001,
            'joint_lr_prototype_vectors': 0.001,
            'joint_weight_decay': 0.0001,
            'last_layer_lr': 0.0001,
            'loss_weight_cross_entropy': 1.0,
            'loss_weight_l1': 0.0001,
            'loss_weight_kld': 0.1,
            'poly_lr_power': 0.9,
            'iter_size': 1,
            'ignore_void_class': True,
            'early_stopping_patience': 5
        },
        'dataset': {
            'name': 'cityscapes',
            'window_size': [64, 64],  # Smaller for testing
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'scales': [0.8, 1.2],
            'image_margin_size': 0,
            'only_19_from_cityscapes': True,
            'dataloader_n_jobs': 0,  # Use 0 for testing
            'train_key': 'train'
        },
        'system': {
            'random_seed': 42,
            'gpus': 0,  # Use CPU for testing
            'load_coco': False,
            'start_checkpoint': ''
        },
        'paths': {
            'data_dir': './data',
            'results_dir': './results',
            'model_dir': './models'
        }
    }
    return config


def create_dummy_dataset(config, num_samples=20):
    """Create dummy dataset for testing"""
    batch_size = config['training']['warmup_batch_size']
    window_size = config['dataset']['window_size']
    num_classes = config['model']['num_classes']
    
    # Create dummy images and masks
    images = torch.randn(num_samples, 3, window_size[0], window_size[1])
    masks = torch.randint(0, num_classes, (num_samples, window_size[0], window_size[1]))
    
    # Create dataset
    dataset = TensorDataset(images, masks)
    return dataset


def test_model_creation(config):
    """Test model creation"""
    print("🧪 Testing model creation...")
    
    try:
        model = construct_PPNet(config)
        print(f"✅ Model created successfully!")
        print(f"   - Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   - Prototypes: {model.num_prototypes}")
        print(f"   - Classes: {model.num_classes}")
        
        # Test forward pass
        dummy_input = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            logits, activations = model.forward_segmentation(dummy_input)
        
        print(f"   - Input shape: {dummy_input.shape}")
        print(f"   - Output logits shape: {logits.shape}")
        print(f"   - Prototype activations shape: {activations.shape}")
        
        return model
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return None


def test_training_module(config, model):
    """Test training module creation"""
    print("\n🧪 Testing training module creation...")
    
    try:
        # Test warmup phase
        module = PatchClassificationModule(config, model, training_phase=0)
        print("✅ Warmup training module created successfully!")
        
        # Test joint phase
        module = PatchClassificationModule(config, model, training_phase=1)
        print("✅ Joint training module created successfully!")
        
        # Test fine-tuning phase
        module = PatchClassificationModule(config, model, training_phase=2)
        print("✅ Fine-tuning training module created successfully!")
        
        return module
        
    except Exception as e:
        print(f"❌ Training module creation failed: {e}")
        return None


def test_training_step(config, model, module):
    """Test a single training step"""
    print("\n🧪 Testing training step...")
    
    try:
        # Create dummy batch
        batch_size = config['training']['warmup_batch_size']
        window_size = config['dataset']['window_size']
        num_classes = config['model']['num_classes']
        
        images = torch.randn(batch_size, 3, window_size[0], window_size[1])
        targets = torch.randint(0, num_classes, (batch_size, window_size[0], window_size[1]))
        
        # Training step
        loss_dict, accuracy = module.training_step((images, targets))
        
        print("✅ Training step completed successfully!")
        print(f"   - Loss: {loss_dict['total_loss']:.4f}")
        print(f"   - Accuracy: {accuracy:.4f}")
        print(f"   - CE Loss: {loss_dict['ce_loss']:.4f}")
        print(f"   - L1 Loss: {loss_dict['l1_loss']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Training step failed: {e}")
        return False


def test_validation_step(config, model, module):
    """Test a single validation step"""
    print("\n🧪 Testing validation step...")
    
    try:
        # Create dummy batch
        batch_size = config['training']['warmup_batch_size']
        window_size = config['dataset']['window_size']
        num_classes = config['model']['num_classes']
        
        images = torch.randn(batch_size, 3, window_size[0], window_size[1])
        targets = torch.randint(0, num_classes, (batch_size, window_size[0], window_size[1]))
        
        # Validation step
        loss_dict, accuracy = module.validation_step((images, targets))
        
        print("✅ Validation step completed successfully!")
        print(f"   - Loss: {loss_dict['total_loss']:.4f}")
        print(f"   - Accuracy: {accuracy:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Validation step failed: {e}")
        return False


def test_checkpoint_saving(module):
    """Test checkpoint saving and loading"""
    print("\n🧪 Testing checkpoint saving...")
    
    try:
        # Save checkpoint
        checkpoint_path = "test_checkpoint.pth"
        module.save_checkpoint(checkpoint_path)
        print("✅ Checkpoint saved successfully!")
        
        # Load checkpoint
        module.load_checkpoint(checkpoint_path)
        print("✅ Checkpoint loaded successfully!")
        
        # Clean up
        import os
        os.remove(checkpoint_path)
        print("✅ Test checkpoint cleaned up!")
        
        return True
        
    except Exception as e:
        print(f"❌ Checkpoint test failed: {e}")
        return False


def main():
    """Main example function"""
    print("🚀 Starting Simplified Prototype Segmentation Example")
    print("=" * 60)
    
    # Create dummy configuration
    config = create_dummy_config()
    print("📋 Created dummy configuration for testing")
    
    # Test model creation
    model = test_model_creation(config)
    if model is None:
        print("❌ Cannot continue without model")
        return
    
    # Test training module
    module = test_training_module(config, model)
    if module is None:
        print("❌ Cannot continue without training module")
        return
    
    # Test training step
    if not test_training_step(config, model, module):
        print("❌ Training step test failed")
        return
    
    # Test validation step
    if not test_validation_step(config, model, module):
        print("❌ Validation step test failed")
        return
    
    # Test checkpoint saving
    if not test_checkpoint_saving(module):
        print("❌ Checkpoint test failed")
        return
    
    print("\n🎉 All tests passed successfully!")
    print("=" * 60)
    print("✅ The simplified prototype segmentation code is working correctly!")
    print("📚 You can now use this code for your own experiments.")
    print("🔧 Modify config.yaml to customize the training parameters.")
    print("📖 Check README.md for detailed usage instructions.")


if __name__ == "__main__":
    main()
