"""
Pytorch Lightning Module for training prototype segmentation model on Cityscapes and SUN datasets
"""
import os
from collections import defaultdict
from typing import Dict, Optional

import gin  # Google's gin-config: A lightweight configuration framework for Python
import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
import numpy as np

from deeplab_pytorch.libs.utils import PolynomialLR
from segmentation.utils import get_params
from helpers import list_of_distances
from model import PPNet
from segmentation.dataset import resize_label
from settings import log
from train_and_test import warm_only, joint, last_only


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def reset_metrics() -> Dict:
    return {
        'n_correct': 0,
        'n_batches': 0,
        'n_patches': 0,
        'cross_entropy': 0,
        'kld_loss': 0,
        'loss': 0
    }


# noinspection PyAbstractClass
@gin.configurable(denylist=['model_dir', 'ppnet', 'training_phase', 'max_steps'])
class PatchClassificationModule(LightningModule):
    """
    =============================================================================
    GIN CONFIGURATION SYSTEM EXPLANATION
    =============================================================================
    
    GIN (Google's gin-config) is a lightweight configuration framework that allows:
    
    1. DECLARATIVE CONFIGURATION: Define hyperparameters in .gin files
    2. AUTOMATIC INJECTION: Automatically inject config values into functions/classes
    3. EXPERIMENT MANAGEMENT: Easy switching between different configurations
    4. REPRODUCIBILITY: Save and load exact configurations for experiments
    
    HOW IT WORKS:
    - @gin.configurable decorator makes this class configurable
    - denylist=['model_dir', 'ppnet', 'training_phase', 'max_steps'] excludes these from config
    - gin.REQUIRED parameters must be provided in .gin config files
    - gin.parse_config_file() loads configuration from .gin files
    
    EXAMPLE CONFIG FILE (cityscapes_kld_imnet.gin):
    PatchClassificationModule.loss_weight_crs_ent = 1.0
    PatchClassificationModule.joint_optimizer_lr_features = 2.5e-5
    PatchClassificationModule.warm_optimizer_lr_add_on_layers = 2.5e-4
    
    =============================================================================
    """
    """
    =============================================================================
    PATCHCLASSIFICATIONMODULE - PYTORCH LIGHTNING MODULE
    =============================================================================
    
    PURPOSE:
        Main training module for prototype-based semantic segmentation.
        This is the core PyTorch Lightning module that handles the complete
        training pipeline for ProtoSegmentation.
    
    KEY RESPONSIBILITIES:
        1. Training Loop Management: Handles train/val/test steps
        2. Loss Computation: Implements prototype-based segmentation losses
        3. Optimizer Configuration: Sets up different optimizers for each phase
        4. Metrics Tracking: Computes accuracy, loss, and prototype statistics
        5. Phase Management: Controls which parameters are trainable
        6. Checkpointing: Saves models at different training stages
    
    TRAINING PHASES:
        - Phase 0 (Warmup): Train prototypes and add-on layers only
          → Handled by: warm_only() function in train_and_test.py
          → Configured in: __init__() method (lines 175-180)
          → Optimizer: configure_optimizers() method (warmup section)
        
        - Phase 1 (Joint): Train all components with different learning rates
          → Handled by: joint() function in train_and_test.py
          → Configured in: __init__() method (lines 175-180)
          → Optimizer: configure_optimizers() method (joint section)
        
        - Phase 2 (Fine-tuning): Train only the last classification layer
          → Handled by: last_only() function in train_and_test.py
          → Configured in: __init__() method (lines 175-180)
          → Optimizer: configure_optimizers() method (last layer section)
    
    PATCH CLASSIFICATION APPROACH:
        - Treats semantic segmentation as pixel-wise classification
        - Each pixel is classified using prototype matching
        - Prototypes represent characteristic features of each class
        - Model learns to match image patches to learned prototypes
    
    =============================================================================
    """
    def __init__(
            self,
            # =============================================================================
            # PATCHCLASSIFICATIONMODULE INITIALIZATION PARAMETERS
            # =============================================================================
            
            # === CORE COMPONENTS ===
            model_dir: str,                    # Directory to save models and results
            ppnet: PPNet,                      # The prototype network model
            training_phase: int,               # Current training phase (0=warmup, 1=joint, 2=finetune)
            max_steps: Optional[int] = None,   # Maximum training steps for this phase
            
            # === LEARNING RATE SCHEDULER ===
            poly_lr_power: float = gin.REQUIRED,  # Power for polynomial LR decay (MUST be in .gin file)
            
            # === LOSS WEIGHTS ===
            loss_weight_crs_ent: float = gin.REQUIRED,  # Cross-entropy loss weight (MUST be in .gin file)
            loss_weight_l1: float = gin.REQUIRED,       # L1 regularization weight (MUST be in .gin file)
            loss_weight_kld: float = 0.0,               # KLD loss weight for prototype diversity (optional)
            
            # === JOINT TRAINING OPTIMIZER (Phase 1) ===
            joint_optimizer_lr_features: float = gin.REQUIRED,        # LR for backbone features (MUST be in .gin file)
            joint_optimizer_lr_add_on_layers: float = gin.REQUIRED,   # LR for add-on layers (MUST be in .gin file)
            joint_optimizer_lr_prototype_vectors: float = gin.REQUIRED, # LR for prototypes (MUST be in .gin file)
            joint_optimizer_weight_decay: float = gin.REQUIRED,       # Weight decay (MUST be in .gin file)
            
            # === WARMUP OPTIMIZER (Phase 0) ===
            warm_optimizer_lr_add_on_layers: float = gin.REQUIRED,    # LR for add-on layers (MUST be in .gin file)
            warm_optimizer_lr_prototype_vectors: float = gin.REQUIRED, # LR for prototypes (MUST be in .gin file)
            warm_optimizer_weight_decay: float = gin.REQUIRED,        # Weight decay (MUST be in .gin file)
            
            # === LAST LAYER OPTIMIZER (Phase 2) ===
            last_layer_optimizer_lr: float = gin.REQUIRED,  # LR for classification head (MUST be in .gin file)
            
            # === TRAINING OPTIONS ===
            ignore_void_class: bool = False,   # Whether to ignore void class (0) in loss
            iter_size: int = 1,                # Gradient accumulation steps
    ):
        super().__init__()
        
        # =============================================================================
        # INITIALIZATION - STORE ALL CONFIGURATION PARAMETERS
        # =============================================================================
        
        # Core components
        self.model_dir = model_dir                                    # Results directory
        self.prototypes_dir = os.path.join(model_dir, 'prototypes')   # Prototype images directory
        self.checkpoints_dir = os.path.join(model_dir, 'checkpoints') # Model checkpoints directory
        self.ppnet = ppnet                                            # The prototype network model
        self.training_phase = training_phase                          # Current training phase
        self.max_steps = max_steps                                    # Max steps for this phase
        
        # Learning rate scheduler
        self.poly_lr_power = poly_lr_power                            # Polynomial LR decay power
        
        # Loss weights
        self.loss_weight_crs_ent = loss_weight_crs_ent                # Cross-entropy loss weight
        self.loss_weight_l1 = loss_weight_l1                          # L1 regularization weight
        self.loss_weight_kld = loss_weight_kld                        # KLD loss weight
        
        # Joint training optimizer parameters (Phase 1)
        self.joint_optimizer_lr_features = joint_optimizer_lr_features
        self.joint_optimizer_lr_add_on_layers = joint_optimizer_lr_add_on_layers
        self.joint_optimizer_lr_prototype_vectors = joint_optimizer_lr_prototype_vectors
        self.joint_optimizer_weight_decay = joint_optimizer_weight_decay
        
        # Warmup optimizer parameters (Phase 0)
        self.warm_optimizer_lr_add_on_layers = warm_optimizer_lr_add_on_layers
        self.warm_optimizer_lr_prototype_vectors = warm_optimizer_lr_prototype_vectors
        self.warm_optimizer_weight_decay = warm_optimizer_weight_decay
        
        # Last layer optimizer parameters (Phase 2)
        self.last_layer_optimizer_lr = last_layer_optimizer_lr
        
        # Training options
        self.ignore_void_class = ignore_void_class                    # Ignore void class in loss
        self.iter_size = iter_size                                    # Gradient accumulation steps

        # Create necessary directories
        os.makedirs(self.prototypes_dir, exist_ok=True)               # For saving prototype images
        os.makedirs(self.checkpoints_dir, exist_ok=True)              # For saving model checkpoints

        # =============================================================================
        # METRICS AND TRAINING STATE INITIALIZATION
        # =============================================================================
        
        # Initialize metrics tracking for different splits
        self.metrics = {}
        for split_key in ['train', 'val', 'test', 'train_last_layer']:
            self.metrics[split_key] = reset_metrics()  # Reset all metrics to zero

        # Training state variables
        self.optimizer_defaults = None  # Will be set by configure_optimizers()
        self.start_step = None          # Track when training started

        # Manual optimization control (PyTorch Lightning feature)
        self.automatic_optimization = False  # We handle optimization manually
        self.best_acc = 0.0                 # Track best validation accuracy

        # =============================================================================
        # TRAINING PHASE CONFIGURATION
        # =============================================================================
        # Configure which parameters are trainable based on training phase
        if self.training_phase == 0:
            # =============================================================================
            # PHASE 0: WARMUP - Initialize prototypes and add-on layers
            # =============================================================================
            # Function: warm_only() in train_and_test.py
            # Freeze: Feature backbone (ResNet101, VGG, etc.)
            # Train: Add-on layers, ASPP layers, prototype vectors, last layer
            warm_only(model=self.ppnet, log=log)
            log(f'🔥 WARM-UP TRAINING START. ({self.max_steps} steps)')
            log('   - Frozen: Feature backbone')
            log('   - Trainable: Add-on layers, ASPP, prototypes, last layer')
            
        elif self.training_phase == 1:
            # =============================================================================
            # PHASE 1: JOINT - Fine-tune entire network
            # =============================================================================
            # Function: joint() in train_and_test.py
            # Train: ALL components with different learning rates
            # Backbone: Low LR, Add-on/Prototypes: High LR
            joint(model=self.ppnet, log=log)
            log(f'🔄 JOINT TRAINING START. ({self.max_steps} steps)')
            log('   - Trainable: ALL components')
            log('   - Learning rates: Backbone (low) vs Add-on/Prototypes (high)')
            
        else:
            # =============================================================================
            # PHASE 2: LAST LAYER - Fine-tune only classification head
            # =============================================================================
            # Function: last_only() in train_and_test.py
            # Freeze: Feature backbone, add-on layers, prototype vectors
            # Train: ONLY last layer (classification head)
            last_only(model=self.ppnet, log=log)
            log('🎯 LAST LAYER TRAINING START.')
            log('   - Frozen: Backbone, add-on layers, prototypes')
            log('   - Trainable: ONLY last layer')

        self.ppnet.prototype_class_identity = self.ppnet.prototype_class_identity.cuda()
        self.lr_scheduler = None
        self.iter_steps = 0
        self.batch_metrics = defaultdict(list)

    def forward(self, x):
        """
        =============================================================================
        FORWARD PASS
        =============================================================================
        Purpose: Forward pass through the prototype network
        Input: x - Input images [B, 3, 513, 513]
        Output: Model predictions (logits and prototype activations)
        =============================================================================
        """
        return self.ppnet(x)

    def _step(self, split_key: str, batch):
        """
        =============================================================================
        TRAINING/VALIDATION STEP
        =============================================================================
        Purpose: Single training or validation step
        Process:
            1. Forward pass through model
            2. Compute losses (cross-entropy, KLD, L1)
            3. Backward pass and optimization (training only)
            4. Update metrics
        =============================================================================
        """
        optimizer = self.optimizers()
        if split_key == 'train' and self.iter_steps == 0:
            optimizer.zero_grad()

        if self.start_step is None:
            self.start_step = self.trainer.global_step

        self.ppnet.features.base.freeze_bn()
        prototype_class_identity = self.ppnet.prototype_class_identity.to(self.device)

        metrics = self.metrics[split_key]

        image, mcs_target = batch

        image = image.to(self.device).to(torch.float32)
        mcs_target = mcs_target.cpu().detach().numpy().astype(np.float32)

        mcs_model_outputs = self.ppnet.forward(image, return_activations=False)
        if not isinstance(mcs_model_outputs, list):
            mcs_model_outputs = [mcs_model_outputs]

        mcs_loss, mcs_cross_entropy, mcs_kld_loss, mcs_cls_act_loss = 0.0, 0.0, 0.0, 0.0
        for output, patch_activations in mcs_model_outputs:
            target = []
            for sample_target in mcs_target:
                target.append(resize_label(sample_target, size=(output.shape[2], output.shape[1])).to(self.device))
            target = torch.stack(target, dim=0)

            # we flatten target/output - classification is done per patch
            output = output.reshape(-1, output.shape[-1])
            target_img = target.reshape(target.shape[0], -1) # (batch_size, img_size)
            target = target.flatten()

            patch_activations = patch_activations.permute(0, 2, 3, 1)
            patch_activations_img = patch_activations.reshape(patch_activations.shape[0], -1, patch_activations.shape[-1]) # (batch_size, img_size, num_proto)

            if self.ignore_void_class:
                # do not predict label for void class (0)
                target_not_void = (target != 0).nonzero().squeeze()
                target = target[target_not_void] - 1
                output = output[target_not_void]

            cross_entropy = torch.nn.functional.cross_entropy(
                output,
                target.long(),
            )

            # calculate KLD over class pixels between prototypes from same class
            kld_loss = []
            for img_i in range(len(target_img)):
                for cls_i in torch.unique(target_img[img_i]).cpu().detach().numpy():
                    if cls_i < 0 or cls_i >= self.ppnet.prototype_class_identity.shape[1]:
                        continue
                    cls_protos = torch.nonzero(self.ppnet.prototype_class_identity[:, cls_i]). \
                        flatten().cpu().detach().numpy()
                    if len(cls_protos) == 0:
                        continue

                    cls_mask = (target_img[img_i] == cls_i)

                    log_cls_activations = [torch.masked_select(patch_activations_img[img_i, :, i], cls_mask)
                                           for i in cls_protos]

                    log_cls_activations = [torch.nn.functional.log_softmax(act, dim=0) for act in log_cls_activations]

                    for i in range(len(cls_protos)):
                        if len(cls_protos) < 2 or len(log_cls_activations[0]) < 2:
                            # no distribution over given class
                            continue

                        log_p1_scores = log_cls_activations[i]
                        for j in range(i + 1, len(cls_protos)):
                            log_p2_scores = log_cls_activations[j]

                            # add kld1 and kld2 to make 'symmetrical kld'
                            kld1 = torch.nn.functional.kl_div(log_p1_scores, log_p2_scores,
                                                              log_target=True, reduction='sum')
                            kld2 = torch.nn.functional.kl_div(log_p2_scores, log_p1_scores,
                                                              log_target=True, reduction='sum')
                            kld = (kld1 + kld2) / 2.0
                            kld_loss.append(kld)

            if len(kld_loss) > 0:
                kld_loss = torch.stack(kld_loss)
                # to make 'loss' (lower == better) take exponent of the negative (maximum value is 1.0, for KLD == 0.0)
                kld_loss = torch.exp(-kld_loss)
                kld_loss = torch.mean(kld_loss)
            else:
                kld_loss = 0.0

            output_class = torch.argmax(output, dim=-1)
            is_correct = output_class == target

            if hasattr(self.ppnet, 'nearest_proto_only') and self.ppnet.nearest_proto_only:
                l1_mask = 1 - torch.eye(self.ppnet.num_classes, device=self.device)
            else:
                l1_mask = 1 - torch.t(prototype_class_identity)

            l1 = (self.ppnet.last_layer.weight * l1_mask).norm(p=1)

            loss = (self.loss_weight_crs_ent * cross_entropy +
                    self.loss_weight_kld * kld_loss +
                    self.loss_weight_l1 * l1)

            mcs_loss += loss / len(mcs_model_outputs)
            mcs_cross_entropy += cross_entropy / len(mcs_model_outputs)
            mcs_kld_loss += kld_loss / len(mcs_model_outputs)
            metrics['n_correct'] += torch.sum(is_correct)
            metrics['n_patches'] += output.shape[0]

        self.batch_metrics['loss'].append(mcs_loss.item())
        self.batch_metrics['cross_entropy'].append(mcs_cross_entropy.item())
        self.batch_metrics['kld_loss'].append(mcs_kld_loss.item())
        self.iter_steps += 1

        if split_key == 'train':
            self.manual_backward(mcs_loss / self.iter_size)

            if self.iter_steps == self.iter_size:
                self.iter_steps = 0
                optimizer.step()

                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

            lr = get_lr(optimizer)
            self.log('lr', lr, on_step=True)

        elif self.iter_steps == self.iter_size:
            self.iter_steps = 0

        if self.iter_steps == 0:
            for key, values in self.batch_metrics.items():
                mean_value = float(np.mean(self.batch_metrics[key]))
                metrics[key] += mean_value
                if key == 'loss':
                    self.log('train_loss_step', mean_value, on_step=True, prog_bar=True)
                # print(key, mean_value)
            # print()
            metrics['n_batches'] += 1

            self.batch_metrics = defaultdict(list)

    def training_step(self, batch, batch_idx):
        """
        =============================================================================
        TRAINING STEP
        =============================================================================
        Purpose: Single training step with gradient computation and optimization
        Features: Enables gradient computation, optimizer updates, and learning rate scheduling
        =============================================================================
        """
        return self._step('train', batch)

    def validation_step(self, batch, batch_idx):
        """
        =============================================================================
        VALIDATION STEP
        =============================================================================
        Purpose: Single validation step without gradient computation
        Features: Disables gradients, computes metrics for validation set
        =============================================================================
        """
        return self._step('val', batch)

    def test_step(self, batch, batch_idx):
        """
        =============================================================================
        TEST STEP
        =============================================================================
        Purpose: Single test step for final evaluation
        Features: Same as validation but for test set evaluation
        =============================================================================
        """
        return self._step('test', batch)

    def on_train_epoch_start(self):
        # reset metrics
        for split_key in self.metrics.keys():
            self.metrics[split_key] = reset_metrics()

        # Freeze the pre-trained batch norm
        self.ppnet.features.base.freeze_bn()

    def on_validation_epoch_end(self):
        val_acc = (self.metrics['val']['n_correct'] / self.metrics['val']['n_patches']).item()

        self.log('training_stage', float(self.training_phase))

        if self.training_phase == 0:
            stage_key = 'warmup'
        elif self.training_phase == 1:
            stage_key = 'nopush'
        else:
            stage_key = 'push'

        torch.save(obj=self.ppnet, f=os.path.join(self.checkpoints_dir, f'{stage_key}_last.pth'))

        if val_acc > self.best_acc:
            log(f'Saving best model, accuracy: ' + str(val_acc))
            self.best_acc = val_acc
            torch.save(obj=self.ppnet, f=os.path.join(self.checkpoints_dir, f'{stage_key}_best.pth'))

    def _epoch_end(self, split_key: str):
        metrics = self.metrics[split_key]
        if len(self.batch_metrics) > 0:
            for key, values in self.batch_metrics.items():
                mean_value = float(np.mean(self.batch_metrics[key]))
                metrics[key] += mean_value
            metrics['n_batches'] += 1

        n_batches = metrics['n_batches']

        self.batch_metrics = defaultdict(list)

        for key in ['loss', 'cross_entropy', 'kld_loss']:
            self.log(f'{split_key}/{key}', metrics[key] / n_batches)

        self.log(f'{split_key}/accuracy', metrics['n_correct'] / metrics['n_patches'])
        self.log('l1', self.ppnet.last_layer.weight.norm(p=1).item())
        if hasattr(self.ppnet, 'nearest_proto_only') and self.ppnet.nearest_proto_only:
            self.log('gumbel_tau', self.ppnet.gumbel_tau)

    def training_epoch_end(self, step_outputs):
        return self._epoch_end('train')

    def validation_epoch_end(self, step_outputs):
        p = self.ppnet.prototype_vectors.view(self.ppnet.prototype_vectors.shape[0], -1).cpu()
        with torch.no_grad():
            p_avg_pair_dist = torch.mean(list_of_distances(p, p))
        self.log('p dist pair', p_avg_pair_dist.item())

        return self._epoch_end('val')

    def test_epoch_end(self, step_outputs):
        return self._epoch_end('test')

    def configure_optimizers(self):
        """
        =============================================================================
        OPTIMIZER CONFIGURATION FOR EACH TRAINING PHASE
        =============================================================================
        Purpose: Set up different optimizers for each training phase
        Phases:
            - Phase 0: Warmup optimizer (handled by warm_only() function)
            - Phase 1: Joint optimizer (handled by joint() function)  
            - Phase 2: Last layer optimizer (handled by last_only() function)
        =============================================================================
        """
        # =============================================================================
        # OPTIMIZER CONFIGURATION FOR EACH TRAINING PHASE
        # =============================================================================
        if self.training_phase == 0:  # PHASE 0: WARMUP
            # =====================================================================
            # WARMUP PHASE OPTIMIZER SETUP
            # =====================================================================
            # Function: warm_only() in train_and_test.py
            # Purpose: Initialize prototypes and add-on layers
            # Trainable: Add-on layers, ASPP layers, prototype vectors, last layer
            # Frozen: Feature backbone (ResNet101, VGG, etc.)
            # =====================================================================
            
            # Get ASPP (Atrous Spatial Pyramid Pooling) parameters from DeepLab
            # These are the dilated convolution layers that capture multi-scale features
            aspp_params = [
                self.ppnet.features.base.aspp.c0.weight,  # ASPP conv 1x1
                self.ppnet.features.base.aspp.c0.bias,
                self.ppnet.features.base.aspp.c1.weight,  # ASPP conv 3x3, rate=6
                self.ppnet.features.base.aspp.c1.bias,
                self.ppnet.features.base.aspp.c2.weight,  # ASPP conv 3x3, rate=12
                self.ppnet.features.base.aspp.c2.bias,
                self.ppnet.features.base.aspp.c3.weight,  # ASPP conv 3x3, rate=18
                self.ppnet.features.base.aspp.c3.bias
            ]
            
            # Configure optimizer with two parameter groups:
            optimizer_specs = [
                {
                    # Group 1: Add-on layers + ASPP layers
                    # These process features from the frozen backbone
                    'params': list(self.ppnet.add_on_layers.parameters()) + aspp_params,
                    'lr': self.warm_optimizer_lr_add_on_layers,  # e.g., 2.5e-4
                    'weight_decay': self.warm_optimizer_weight_decay  # e.g., 5e-4
                },
                {
                    # Group 2: Prototype vectors
                    # These are the learnable prototypes that represent class features
                    'params': self.ppnet.prototype_vectors,
                    'lr': self.warm_optimizer_lr_prototype_vectors  # e.g., 2.5e-4
                    # No weight decay for prototypes (they are normalized)
                }
            ]
        elif self.training_phase == 1:  # PHASE 1: JOINT TRAINING
            # =====================================================================
            # JOINT PHASE OPTIMIZER SETUP
            # =====================================================================
            # Function: joint() in train_and_test.py
            # Purpose: Fine-tune entire network with different learning rates
            # Trainable: ALL components (backbone, add-on, prototypes, last layer)
            # Strategy: Lower LR for backbone, higher LR for new components
            # =====================================================================
            
            optimizer_specs = [
                {
                    # Group 1: Backbone layers (ResNet101 layers)
                    # Use lower learning rate to preserve pretrained features
                    "params": get_params(self.ppnet.features, key="1x"),  # ResNet layers
                    'lr': self.joint_optimizer_lr_features,  # e.g., 2.5e-5 (low)
                    'weight_decay': self.joint_optimizer_weight_decay  # e.g., 5e-4
                },
                {
                    # Group 2: ASPP conv weights (10x higher LR)
                    # These are important for multi-scale feature extraction
                    "params": get_params(self.ppnet.features, key="10x"),  # ASPP conv weights
                    'lr': 10 * self.joint_optimizer_lr_features,  # e.g., 2.5e-4 (10x higher)
                    'weight_decay': self.joint_optimizer_weight_decay
                },
                {
                    # Group 3: ASPP conv biases (10x higher LR)
                    "params": get_params(self.ppnet.features, key="20x"),  # ASPP conv biases
                    'lr': 10 * self.joint_optimizer_lr_features,  # e.g., 2.5e-4 (10x higher)
                    'weight_decay': self.joint_optimizer_weight_decay
                },
                {
                    # Group 4: Add-on layers
                    # These are new layers added on top of the backbone
                    'params': self.ppnet.add_on_layers.parameters(),
                    'lr': self.joint_optimizer_lr_add_on_layers,  # e.g., 2.5e-4 (high)
                    'weight_decay': self.joint_optimizer_weight_decay
                },
                {
                    # Group 5: Prototype vectors
                    # These are the core learnable prototypes
                    'params': self.ppnet.prototype_vectors,
                    'lr': self.joint_optimizer_lr_prototype_vectors  # e.g., 2.5e-4 (high)
                    # No weight decay for prototypes
                }
            ]
        else:  # PHASE 2: LAST LAYER FINE-TUNING
            # =====================================================================
            # LAST LAYER PHASE OPTIMIZER SETUP
            # =====================================================================
            # Function: last_only() in train_and_test.py
            # Purpose: Fine-tune only the classification head
            # Trainable: ONLY last layer (classification head)
            # Frozen: Feature backbone, add-on layers, prototype vectors
            # =====================================================================
            
            optimizer_specs = [
                {
                    # Only train the final classification layer
                    # This layer maps prototype activations to class predictions
                    'params': self.ppnet.last_layer.parameters(),
                    'lr': self.last_layer_optimizer_lr  # e.g., 1e-5 (very low)
                    # No weight decay for last layer
                }
            ]

        # Create Adam optimizer with the configured parameter groups
        optimizer = torch.optim.Adam(optimizer_specs)

        # =============================================================================
        # LEARNING RATE SCHEDULER SETUP
        # =============================================================================
        if self.training_phase == 1:  # Only use scheduler for joint training
            # Polynomial learning rate decay for joint training
            # Helps with convergence during long joint training phase
            self.lr_scheduler = PolynomialLR(
                optimizer=optimizer,
                step_size=1,
                iter_max=self.max_steps // self.iter_size,  # Total iterations
                power=self.poly_lr_power  # e.g., 0.9 (decay rate)
            )

        return optimizer
