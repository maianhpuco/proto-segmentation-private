"""
=============================================================================
PIPELINE TEST WITH RANDOM INPUT DATA
=============================================================================

This script tests the complete prototype segmentation pipeline using random input data.
It verifies that all components work correctly without requiring real dataset files.

HOW TO RUN:
python test_pipeline.py

This will test:
1. Model creation and forward pass
2. Training module initialization
3. Training and validation steps
4. Checkpoint saving/loading
5. Complete training pipeline simulation
=============================================================================
"""

import torch
import torch.nn as nn
import numpy as np
import os
import sys
from torch.utils.data import DataLoader, TensorDataset
import yaml

# Add current directory to path for imports
sys.path.append('.')

from model import construct_PPNet
from module import PatchClassificationModule


def create_test_config():
    """Create a minimal test configuration"""
    config = {
        'model': {
            'base_architecture': 'deeplabv2_resnet101',
            'pretrained': False,  # Use False for faster testing
            'prototype_shape': [6, 16, 1, 1],  # Very small for testing
            'num_classes': 3,
            'prototype_activation_function': 'log',
            'add_on_layers_type': 'deeplab_simple'
        },
        'training': {
            'warmup_steps': 5,
            'joint_steps': 10,
            'finetune_steps': 5,
            'warmup_batch_size': 2,
            'joint_batch_size': 2,
            'warmup_lr_add_on_layers': 0.01,
            'warmup_lr_prototype_vectors': 0.01,
            'warmup_weight_decay': 0.0001,
            'joint_lr_features': 0.001,
            'joint_lr_add_on_layers': 0.01,
            'joint_lr_prototype_vectors': 0.01,
            'joint_weight_decay': 0.0001,
            'last_layer_lr': 0.001,
            'loss_weight_cross_entropy': 1.0,
            'loss_weight_l1': 0.0001,
            'loss_weight_kld': 0.1,
            'poly_lr_power': 0.9,
            'iter_size': 1,
            'ignore_void_class': True,
            'early_stopping_patience': 3
        },
        'dataset': {
            'name': 'test',
            'window_size': [32, 32],  # Very small for testing
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'scales': [0.8, 1.2],
            'image_margin_size': 0,
            'only_19_from_cityscapes': True,
            'dataloader_n_jobs': 0,
            'train_key': 'train'
        },
        'system': {
            'random_seed': 42,
            'gpus': 0,  # Use CPU for testing
            'load_coco': False,
            'start_checkpoint': ''
        },
        'paths': {
            'data_dir': './test_data',
            'results_dir': './test_results',
            'model_dir': './test_models'
        }
    }
    return config


def create_random_dataset(config, num_samples=20):
    """Create random dataset for testing"""
    batch_size = config['training']['warmup_batch_size']
    window_size = config['dataset']['window_size']
    num_classes = config['model']['num_classes']
    
    print(f"📊 Creating random dataset:")
    print(f"   - Samples: {num_samples}")
    print(f"   - Image size: {window_size}")
    print(f"   - Classes: {num_classes}")
    
    # Create random images (normalized)
    images = torch.randn(num_samples, 3, window_size[0], window_size[1])
    images = images * 0.5 + 0.5  # Scale to [0, 1] range
    
    # Create random masks with class IDs
    masks = torch.randint(0, num_classes, (num_samples, window_size[0], window_size[1]))
    
    return images, masks


def test_model_creation(config):
    """Test 1: Model creation and basic forward pass"""
    print("\n🧪 TEST 1: Model Creation and Forward Pass")
    print("-" * 50)
    
    try:
        # Create model
        print("Creating model...")
        model = construct_PPNet(config)
        print(f"✅ Model created successfully!")
        print(f"   - Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   - Prototypes: {model.num_prototypes}")
        print(f"   - Classes: {model.num_classes}")
        
        # Test forward pass
        print("\nTesting forward pass...")
        batch_size = config['training']['warmup_batch_size']
        window_size = config['dataset']['window_size']
        
        dummy_input = torch.randn(batch_size, 3, window_size[0], window_size[1])
        print(f"   - Input shape: {dummy_input.shape}")
        
        with torch.no_grad():
            logits, activations = model.forward_segmentation(dummy_input)
        
        print(f"   - Output logits shape: {logits.shape}")
        print(f"   - Prototype activations shape: {activations.shape}")
        print(f"   - Logits range: [{logits.min():.3f}, {logits.max():.3f}]")
        print(f"   - Activations range: [{activations.min():.3f}, {activations.max():.3f}]")
        
        return model
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_training_module(config, model):
    """Test 2: Training module creation and parameter setup"""
    print("\n🧪 TEST 2: Training Module Creation")
    print("-" * 50)
    
    try:
        # Test all three phases
        phases = [0, 1, 2]
        phase_names = ["Warmup", "Joint", "Fine-tuning"]
        
        for phase, name in zip(phases, phase_names):
            print(f"\nTesting {name} phase (phase {phase})...")
            
            module = PatchClassificationModule(config, model, training_phase=phase)
            
            # Count trainable parameters
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())
            
            print(f"   ✅ {name} module created successfully!")
            print(f"   - Trainable parameters: {trainable_params:,} / {total_params:,}")
            print(f"   - Trainable ratio: {trainable_params/total_params:.2%}")
        
        return module
        
    except Exception as e:
        print(f"❌ Training module creation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_training_step(config, model, module):
    """Test 3: Training step execution"""
    print("\n🧪 TEST 3: Training Step Execution")
    print("-" * 50)
    
    try:
        # Create random batch
        batch_size = config['training']['warmup_batch_size']
        window_size = config['dataset']['window_size']
        num_classes = config['model']['num_classes']
        
        images = torch.randn(batch_size, 3, window_size[0], window_size[1])
        targets = torch.randint(0, num_classes, (batch_size, window_size[0], window_size[1]))
        
        print(f"Created random batch:")
        print(f"   - Images shape: {images.shape}")
        print(f"   - Targets shape: {targets.shape}")
        print(f"   - Target classes: {torch.unique(targets).tolist()}")
        
        # Training step
        print("\nExecuting training step...")
        loss_dict, accuracy = module.training_step((images, targets))
        
        print(f"✅ Training step completed successfully!")
        print(f"   - Total loss: {loss_dict['total_loss']:.4f}")
        print(f"   - CE loss: {loss_dict['ce_loss']:.4f}")
        print(f"   - L1 loss: {loss_dict['l1_loss']:.4f}")
        print(f"   - KLD loss: {loss_dict['kld_loss']:.4f}")
        print(f"   - Accuracy: {accuracy:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Training step failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_validation_step(config, model, module):
    """Test 4: Validation step execution"""
    print("\n🧪 TEST 4: Validation Step Execution")
    print("-" * 50)
    
    try:
        # Create random batch
        batch_size = config['training']['warmup_batch_size']
        window_size = config['dataset']['window_size']
        num_classes = config['model']['num_classes']
        
        images = torch.randn(batch_size, 3, window_size[0], window_size[1])
        targets = torch.randint(0, num_classes, (batch_size, window_size[0], window_size[1]))
        
        print(f"Created random validation batch:")
        print(f"   - Images shape: {images.shape}")
        print(f"   - Targets shape: {targets.shape}")
        
        # Validation step
        print("\nExecuting validation step...")
        loss_dict, accuracy = module.validation_step((images, targets))
        
        print(f"✅ Validation step completed successfully!")
        print(f"   - Total loss: {loss_dict['total_loss']:.4f}")
        print(f"   - CE loss: {loss_dict['ce_loss']:.4f}")
        print(f"   - L1 loss: {loss_dict['l1_loss']:.4f}")
        print(f"   - KLD loss: {loss_dict['kld_loss']:.4f}")
        print(f"   - Accuracy: {accuracy:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Validation step failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_checkpoint_operations(module):
    """Test 5: Checkpoint saving and loading"""
    print("\n🧪 TEST 5: Checkpoint Operations")
    print("-" * 50)
    
    try:
        # Create test directory
        os.makedirs("test_models", exist_ok=True)
        
        # Save checkpoint
        checkpoint_path = "test_models/test_checkpoint.pth"
        print(f"Saving checkpoint to {checkpoint_path}...")
        module.save_checkpoint(checkpoint_path)
        print("✅ Checkpoint saved successfully!")
        
        # Check file exists and size
        if os.path.exists(checkpoint_path):
            file_size = os.path.getsize(checkpoint_path) / (1024 * 1024)  # MB
            print(f"   - File size: {file_size:.2f} MB")
        
        # Load checkpoint
        print(f"Loading checkpoint from {checkpoint_path}...")
        module.load_checkpoint(checkpoint_path)
        print("✅ Checkpoint loaded successfully!")
        
        # Clean up
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
            print("✅ Test checkpoint cleaned up!")
        
        return True
        
    except Exception as e:
        print(f"❌ Checkpoint operations failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_complete_training_loop(config, model):
    """Test 6: Complete training loop simulation"""
    print("\n🧪 TEST 6: Complete Training Loop Simulation")
    print("-" * 50)
    
    try:
        # Create random dataset
        images, masks = create_random_dataset(config, num_samples=10)
        
        # Create data loader
        dataset = TensorDataset(images, masks)
        dataloader = DataLoader(dataset, batch_size=config['training']['warmup_batch_size'], shuffle=True)
        
        print(f"Created data loader with {len(dataloader)} batches")
        
        # Test all three training phases
        phases = [0, 1, 2]
        phase_names = ["Warmup", "Joint", "Fine-tuning"]
        phase_steps = [2, 3, 2]  # Very few steps for testing
        
        for phase, name, steps in zip(phases, phase_names, phase_steps):
            print(f"\n--- {name} Phase (Phase {phase}) ---")
            
            # Create module for this phase
            module = PatchClassificationModule(config, model, training_phase=phase)
            
            # Run a few training steps
            for step in range(steps):
                # Get batch
                batch = next(iter(dataloader))
                
                # Training step
                loss_dict, train_acc = module.training_step(batch)
                
                # Validation step
                val_batch = next(iter(dataloader))
                val_loss_dict, val_acc = module.validation_step(val_batch)
                
                print(f"   Step {step+1}/{steps}: Train Loss={loss_dict['total_loss']:.4f}, "
                      f"Train Acc={train_acc:.4f}, Val Loss={val_loss_dict['total_loss']:.4f}, "
                      f"Val Acc={val_acc:.4f}")
            
            print(f"✅ {name} phase completed successfully!")
        
        return True
        
    except Exception as e:
        print(f"❌ Complete training loop failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_prototype_operations(config, model):
    """Test 7: Prototype operations"""
    print("\n🧪 TEST 7: Prototype Operations")
    print("-" * 50)
    
    try:
        print(f"Testing prototype operations:")
        print(f"   - Number of prototypes: {model.num_prototypes}")
        print(f"   - Prototype shape: {model.prototype_vectors.shape}")
        print(f"   - Prototype class identity shape: {model.prototype_class_identity.shape}")
        
        # Test prototype class distribution
        prototypes_per_class = model.num_prototypes // model.num_classes
        print(f"   - Prototypes per class: {prototypes_per_class}")
        
        # Check class identity matrix
        for class_id in range(model.num_classes):
            class_prototypes = torch.sum(model.prototype_class_identity[:, class_id]).item()
            print(f"   - Class {class_id}: {int(class_prototypes)} prototypes")
        
        # Test prototype activations
        batch_size = config['training']['warmup_batch_size']
        window_size = config['dataset']['window_size']
        dummy_input = torch.randn(batch_size, 3, window_size[0], window_size[1])
        
        with torch.no_grad():
            conv_features = model.conv_features(dummy_input)
            distances = model._l2_convolution(conv_features, model.prototype_vectors)
            activations = model._prototype_activations(distances)
        
        print(f"   - Feature shape: {conv_features.shape}")
        print(f"   - Distance shape: {distances.shape}")
        print(f"   - Activation shape: {activations.shape}")
        print(f"   - Distance range: [{distances.min():.3f}, {distances.max():.3f}]")
        print(f"   - Activation range: [{activations.min():.3f}, {activations.max():.3f}]")
        
        print("✅ Prototype operations completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Prototype operations failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def cleanup_test_files():
    """Clean up test files and directories"""
    print("\n🧹 Cleaning up test files...")
    
    test_dirs = ["test_models", "test_results", "test_data"]
    for dir_name in test_dirs:
        if os.path.exists(dir_name):
            import shutil
            shutil.rmtree(dir_name)
            print(f"   - Removed {dir_name}/")


def main():
    """Main test function"""
    print("🚀 PROTOTYPE SEGMENTATION PIPELINE TEST")
    print("=" * 60)
    print("Testing the complete pipeline with random input data...")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create test configuration
    config = create_test_config()
    print(f"📋 Created test configuration")
    
    # Run all tests
    tests_passed = 0
    total_tests = 7
    
    # Test 1: Model creation
    model = test_model_creation(config)
    if model is not None:
        tests_passed += 1
    
    if model is None:
        print("\n❌ Cannot continue without model. Stopping tests.")
        return
    
    # Test 2: Training module
    module = test_training_module(config, model)
    if module is not None:
        tests_passed += 1
    
    # Test 3: Training step
    if test_training_step(config, model, module):
        tests_passed += 1
    
    # Test 4: Validation step
    if test_validation_step(config, model, module):
        tests_passed += 1
    
    # Test 5: Checkpoint operations
    if test_checkpoint_operations(module):
        tests_passed += 1
    
    # Test 6: Complete training loop
    if test_complete_training_loop(config, model):
        tests_passed += 1
    
    # Test 7: Prototype operations
    if test_prototype_operations(config, model):
        tests_passed += 1
    
    # Clean up
    cleanup_test_files()
    
    # Final results
    print("\n" + "=" * 60)
    print("🎯 TEST RESULTS SUMMARY")
    print("=" * 60)
    print(f"Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("🎉 ALL TESTS PASSED! The pipeline is working correctly!")
        print("\n✅ You can now use this code for real training:")
        print("   1. Prepare your dataset")
        print("   2. Modify config.yaml with your parameters")
        print("   3. Run: python train.py")
    else:
        print(f"❌ {total_tests - tests_passed} tests failed. Please check the errors above.")
    
    print("\n📚 For detailed usage instructions, see README.md")


if __name__ == "__main__":
    main()
