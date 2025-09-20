"""
=============================================================================
SIMPLIFIED TRAINING MODULE FOR PROTOTYPE SEGMENTATION
=============================================================================

USAGE INSTRUCTIONS:
1. This file contains the PyTorch Lightning module for training
2. Handles the 3-phase training: Warmup -> Joint -> Fine-tuning
3. Manages loss computation, optimization, and metrics
4. Supports different training phases with different parameter configurations

HOW TO USE:
- Import: from module import PatchClassificationModule
- Create module: module = PatchClassificationModule(config, model, training_phase)
- Use with PyTorch Lightning trainer

TRAINING PHASES:
- Phase 0 (Warmup): Train prototypes and add-on layers only
- Phase 1 (Joint): Train all components with different learning rates
- Phase 2 (Fine-tuning): Train only the last classification layer

CONFIGURATION:
- All parameters are configured via config.yaml
- Modify learning rates and loss weights in the config file
=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim import Adam
from torch.optim.lr_scheduler import PolynomialLR
from typing import Dict, Optional
import yaml


class PatchClassificationModule:
    """
    Simplified training module for prototype-based segmentation
    Handles 3-phase training without PyTorch Lightning complexity
    """
    
    def __init__(self, config: dict, model: nn.Module, training_phase: int):
        """
        Initialize training module
        
        Args:
            config: Configuration dictionary
            model: PPNet model
            training_phase: Training phase (0=warmup, 1=joint, 2=finetune)
        """
        self.config = config
        self.model = model
        self.training_phase = training_phase
        
        # Get training configuration
        self.training_config = config['training']
        
        # Setup training phase
        self._setup_training_phase()
        
        # Initialize optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Initialize metrics
        self.metrics = {
            'train_loss': 0.0,
            'train_acc': 0.0,
            'val_loss': 0.0,
            'val_acc': 0.0,
            'step': 0
        }
        
        print(f"Initialized training module for phase {training_phase}")
        print(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}")
    
    def _setup_training_phase(self):
        """Setup which parameters are trainable based on training phase"""
        if self.training_phase == 0:  # Warmup
            self._warmup_phase()
        elif self.training_phase == 1:  # Joint
            self._joint_phase()
        else:  # Fine-tuning
            self._finetune_phase()
    
    def _warmup_phase(self):
        """Phase 0: Train prototypes and add-on layers only"""
        print("🔥 WARMUP PHASE: Training prototypes and add-on layers")
        
        # Freeze backbone
        for param in self.model.features.parameters():
            param.requires_grad = False
        
        # Train add-on layers and prototypes
        for param in self.model.add_on_layers.parameters():
            param.requires_grad = True
        self.model.prototype_vectors.requires_grad = True
        
        # Train last layer
        for param in self.model.last_layer.parameters():
            param.requires_grad = True
    
    def _joint_phase(self):
        """Phase 1: Train all components with different learning rates"""
        print("🔄 JOINT PHASE: Training all components")
        
        # Train all parameters
        for param in self.model.parameters():
            param.requires_grad = True
    
    def _finetune_phase(self):
        """Phase 2: Train only the last layer"""
        print("🎯 FINE-TUNING PHASE: Training only last layer")
        
        # Freeze everything except last layer
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Train only last layer
        for param in self.model.last_layer.parameters():
            param.requires_grad = True
    
    def _create_optimizer(self):
        """Create optimizer based on training phase"""
        if self.training_phase == 0:  # Warmup
            return Adam([
                {'params': self.model.add_on_layers.parameters(), 
                 'lr': self.training_config['warmup_lr_add_on_layers']},
                {'params': [self.model.prototype_vectors], 
                 'lr': self.training_config['warmup_lr_prototype_vectors']},
                {'params': self.model.last_layer.parameters(), 
                 'lr': self.training_config['warmup_lr_add_on_layers']}
            ], weight_decay=self.training_config['warmup_weight_decay'])
        
        elif self.training_phase == 1:  # Joint
            return Adam([
                {'params': self.model.features.parameters(), 
                 'lr': self.training_config['joint_lr_features']},
                {'params': self.model.add_on_layers.parameters(), 
                 'lr': self.training_config['joint_lr_add_on_layers']},
                {'params': [self.model.prototype_vectors], 
                 'lr': self.training_config['joint_lr_prototype_vectors']},
                {'params': self.model.last_layer.parameters(), 
                 'lr': self.training_config['joint_lr_add_on_layers']}
            ], weight_decay=self.training_config['joint_weight_decay'])
        
        else:  # Fine-tuning
            return Adam([
                {'params': self.model.last_layer.parameters(), 
                 'lr': self.training_config['last_layer_lr']}
            ])
    
    def _create_scheduler(self):
        """Create learning rate scheduler"""
        if self.training_phase == 1:  # Only use scheduler for joint training
            max_steps = self.training_config['joint_steps']
            return PolynomialLR(self.optimizer, 
                              power=self.training_config['poly_lr_power'],
                              total_iters=max_steps)
        return None
    
    def compute_loss(self, logits, targets, prototype_activations):
        """
        Compute total loss
        
        Args:
            logits: Model predictions [B, C, H, W]
            targets: Ground truth masks [B, H, W]
            prototype_activations: Prototype activations [B, P, H, W]
        
        Returns:
            Total loss and loss components
        """
        # Cross-entropy loss
        ce_loss = F.cross_entropy(logits, targets, ignore_index=0 if self.training_config['ignore_void_class'] else -100)
        
        # L1 regularization on prototype activations
        l1_loss = torch.mean(torch.abs(prototype_activations))
        
        # KLD loss for prototype diversity (simplified)
        kld_loss = 0.0
        if self.training_config['loss_weight_kld'] > 0:
            # Simple KLD loss to encourage prototype diversity
            prototype_means = torch.mean(prototype_activations, dim=(0, 2, 3))
            kld_loss = torch.mean(prototype_means)
        
        # Total loss
        total_loss = (self.training_config['loss_weight_cross_entropy'] * ce_loss +
                     self.training_config['loss_weight_l1'] * l1_loss +
                     self.training_config['loss_weight_kld'] * kld_loss)
        
        return total_loss, {
            'ce_loss': ce_loss.item(),
            'l1_loss': l1_loss.item(),
            'kld_loss': kld_loss.item(),
            'total_loss': total_loss.item()
        }
    
    def compute_accuracy(self, logits, targets):
        """Compute pixel-wise accuracy"""
        predictions = torch.argmax(logits, dim=1)
        if self.training_config['ignore_void_class']:
            mask = targets != 0
            correct = (predictions == targets) & mask
            total = mask.sum()
        else:
            correct = (predictions == targets)
            total = targets.numel()
        
        accuracy = correct.sum().float() / total if total > 0 else 0.0
        return accuracy.item()
    
    def training_step(self, batch):
        """Single training step"""
        images, targets = batch
        
        # Forward pass
        self.model.train()
        logits, prototype_activations = self.model.forward_segmentation(images)
        
        # Compute loss
        loss, loss_dict = self.compute_loss(logits, targets, prototype_activations)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update scheduler
        if self.scheduler is not None:
            self.scheduler.step()
        
        # Compute accuracy
        accuracy = self.compute_accuracy(logits, targets)
        
        # Update metrics
        self.metrics['train_loss'] = loss_dict['total_loss']
        self.metrics['train_acc'] = accuracy
        self.metrics['step'] += 1
        
        return loss_dict, accuracy
    
    def validation_step(self, batch):
        """Single validation step"""
        images, targets = batch
        
        # Forward pass
        self.model.eval()
        with torch.no_grad():
            logits, prototype_activations = self.model.forward_segmentation(images)
            
            # Compute loss
            loss, loss_dict = self.compute_loss(logits, targets, prototype_activations)
            
            # Compute accuracy
            accuracy = self.compute_accuracy(logits, targets)
        
        # Update metrics
        self.metrics['val_loss'] = loss_dict['total_loss']
        self.metrics['val_acc'] = accuracy
        
        return loss_dict, accuracy
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_phase': self.training_phase,
            'metrics': self.metrics,
            'config': self.config
        }
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.metrics = checkpoint['metrics']
        print(f"Checkpoint loaded from {path}")


# Example usage
if __name__ == "__main__":
    import yaml
    from model import construct_PPNet
    
    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create model
    model = construct_PPNet(config)
    
    # Create training module
    module = PatchClassificationModule(config, model, training_phase=0)
    
    print("Training module created successfully!")
