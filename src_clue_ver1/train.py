"""
=============================================================================
SIMPLIFIED TRAINING SCRIPT FOR PROTOTYPE SEGMENTATION
=============================================================================

USAGE INSTRUCTIONS:
1. This script implements the complete 3-phase training pipeline
2. Phase 0: Warmup - Train prototypes and add-on layers only
3. Phase 1: Joint - Train all components with different learning rates
4. Phase 2: Fine-tuning - Train only the last classification layer
5. Includes prototype pushing after joint training

HOW TO RUN:
1. Prepare your dataset in the data directory
2. Modify config.yaml for your specific setup
3. Run: python train.py

COMMAND LINE ARGUMENTS:
- --config: Path to config file (default: config.yaml)
- --resume: Path to checkpoint to resume from (optional)
- --phase: Specific training phase to run (0, 1, or 2)

EXAMPLE USAGE:
python train.py --config config.yaml
python train.py --resume checkpoints/warmup_checkpoint.pth
python train.py --phase 1  # Run only joint training
=============================================================================
"""

import os
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import time

from model import construct_PPNet
from module import PatchClassificationModule
from dataset import PatchClassificationDataset


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_data_loaders(config: dict):
    """Create data loaders for training and validation"""
    dataset_config = config['dataset']
    training_config = config['training']
    
    # Create datasets
    train_dataset = PatchClassificationDataset('train', config)
    val_dataset = PatchClassificationDataset('val', config)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config['warmup_batch_size'],
        shuffle=True,
        num_workers=dataset_config['dataloader_n_jobs'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config['warmup_batch_size'],
        shuffle=False,
        num_workers=dataset_config['dataloader_n_jobs'],
        pin_memory=True
    )
    
    return train_loader, val_loader


def train_phase(module: PatchClassificationModule, train_loader: DataLoader, 
                val_loader: DataLoader, phase_name: str, max_steps: int):
    """Train a specific phase"""
    print(f"\n🚀 Starting {phase_name} training for {max_steps} steps")
    
    best_val_acc = 0.0
    patience_counter = 0
    patience = 50  # Early stopping patience
    
    for step in tqdm(range(max_steps), desc=f"{phase_name} Training"):
        # Training step
        module.model.train()
        train_batch = next(iter(train_loader))
        train_batch = [x.cuda() if torch.cuda.is_available() else x for x in train_batch]
        
        loss_dict, train_acc = module.training_step(train_batch)
        
        # Validation step (every 100 steps)
        if step % 100 == 0:
            module.model.eval()
            val_batch = next(iter(val_loader))
            val_batch = [x.cuda() if torch.cuda.is_available() else x for x in val_batch]
            
            val_loss_dict, val_acc = module.validation_step(val_batch)
            
            # Print progress
            print(f"\nStep {step}/{max_steps}")
            print(f"Train Loss: {loss_dict['total_loss']:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss_dict['total_loss']:.4f}, Val Acc: {val_acc:.4f}")
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"Early stopping at step {step}")
                break
    
    print(f"✅ {phase_name} training completed. Best val acc: {best_val_acc:.4f}")
    return best_val_acc


def push_prototypes(model: nn.Module, train_loader: DataLoader, config: dict):
    """Simplified prototype pushing"""
    print("\n🔄 Starting prototype pushing...")
    
    model.eval()
    device = next(model.parameters()).device
    
    # Get prototype shape
    prototype_shape = model.prototype_vectors.shape
    num_prototypes = prototype_shape[0]
    
    # Initialize tracking variables
    global_min_distances = torch.full((num_prototypes,), float('inf'), device=device)
    global_min_patches = torch.zeros(prototype_shape, device=device)
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(tqdm(train_loader, desc="Pushing prototypes")):
            images = images.to(device)
            
            # Get features
            conv_features = model.conv_features(images)
            
            # Compute distances
            distances = model._l2_convolution(conv_features, model.prototype_vectors)
            
            # Find minimum distances for each prototype
            for proto_idx in range(num_prototypes):
                proto_distances = distances[:, proto_idx, :, :]
                min_dist, min_idx = torch.min(proto_distances.view(-1), dim=0)
                
                if min_dist < global_min_distances[proto_idx]:
                    global_min_distances[proto_idx] = min_dist
                    # Store the corresponding feature patch
                    batch_idx_min = min_idx // (proto_distances.shape[1] * proto_distances.shape[2])
                    h_idx = (min_idx % (proto_distances.shape[1] * proto_distances.shape[2])) // proto_distances.shape[2]
                    w_idx = min_idx % proto_distances.shape[2]
                    
                    global_min_patches[proto_idx] = conv_features[batch_idx_min, :, h_idx, w_idx]
    
    # Update prototype vectors with best matches
    model.prototype_vectors.data = global_min_patches
    print("✅ Prototype pushing completed")


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train Prototype Segmentation Model')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--phase', type=int, default=None, help='Specific phase to run (0, 1, or 2)')
    parser.add_argument('--skip_push', action='store_true', help='Skip prototype pushing')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    print(f"📋 Loaded configuration from {args.config}")
    
    # Set random seed
    torch.manual_seed(config['system']['random_seed'])
    np.random.seed(config['system']['random_seed'])
    
    # Create directories
    os.makedirs(config['paths']['results_dir'], exist_ok=True)
    os.makedirs(config['paths']['model_dir'], exist_ok=True)
    
    # Create model
    model = construct_PPNet(config)
    if torch.cuda.is_available():
        model = model.cuda()
    print(f"🏗️ Created model with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(config)
    print(f"📊 Created data loaders: {len(train_loader)} train batches, {len(val_loader)} val batches")
    
    # Training phases
    training_config = config['training']
    
    if args.phase is not None:
        # Run specific phase
        phases_to_run = [args.phase]
    else:
        # Run all phases
        phases_to_run = [0, 1, 2]
    
    for phase in phases_to_run:
        if phase == 0:  # Warmup
            module = PatchClassificationModule(config, model, training_phase=0)
            if args.resume and 'warmup' in args.resume:
                module.load_checkpoint(args.resume)
            
            best_acc = train_phase(module, train_loader, val_loader, "Warmup", 
                                 training_config['warmup_steps'])
            
            # Save checkpoint
            checkpoint_path = os.path.join(config['paths']['model_dir'], 'warmup_checkpoint.pth')
            module.save_checkpoint(checkpoint_path)
        
        elif phase == 1:  # Joint training
            module = PatchClassificationModule(config, model, training_phase=1)
            if args.resume and 'joint' in args.resume:
                module.load_checkpoint(args.resume)
            elif not args.resume:
                # Load warmup checkpoint
                warmup_path = os.path.join(config['paths']['model_dir'], 'warmup_checkpoint.pth')
                if os.path.exists(warmup_path):
                    module.load_checkpoint(warmup_path)
            
            best_acc = train_phase(module, train_loader, val_loader, "Joint", 
                                 training_config['joint_steps'])
            
            # Save checkpoint
            checkpoint_path = os.path.join(config['paths']['model_dir'], 'joint_checkpoint.pth')
            module.save_checkpoint(checkpoint_path)
            
            # Prototype pushing
            if not args.skip_push:
                push_prototypes(model, train_loader, config)
                push_checkpoint_path = os.path.join(config['paths']['model_dir'], 'push_checkpoint.pth')
                module.save_checkpoint(push_checkpoint_path)
        
        elif phase == 2:  # Fine-tuning
            module = PatchClassificationModule(config, model, training_phase=2)
            if args.resume and 'finetune' in args.resume:
                module.load_checkpoint(args.resume)
            else:
                # Load push checkpoint
                push_path = os.path.join(config['paths']['model_dir'], 'push_checkpoint.pth')
                if os.path.exists(push_path):
                    module.load_checkpoint(push_path)
            
            best_acc = train_phase(module, train_loader, val_loader, "Fine-tuning", 
                                 training_config['finetune_steps'])
            
            # Save final checkpoint
            final_checkpoint_path = os.path.join(config['paths']['model_dir'], 'final_checkpoint.pth')
            module.save_checkpoint(final_checkpoint_path)
    
    print("\n🎉 Training completed successfully!")
    print(f"📁 Checkpoints saved in: {config['paths']['model_dir']}")


if __name__ == "__main__":
    main()
