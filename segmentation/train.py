"""
Training prototype segmentation model on Cityscapes or SUN dataset

Example run:

python -m segmentation.train cityscapes 2022_03_26_cityscapes
"""
import os
import shutil
from typing import Optional

import argh
import torch
import neptune.new as neptune
import torchvision
from pytorch_lightning import Trainer, seed_everything
import gin
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import NeptuneLogger, TensorBoardLogger, CSVLogger

from segmentation.data_module import PatchClassificationDataModule
from segmentation.dataset import PatchClassificationDataset
from segmentation.module import PatchClassificationModule
from segmentation.config import get_operative_config_json
from model import construct_PPNet
from segmentation.push import push_prototypes
from settings import log
from deeplab_features import torchvision_resnet_weight_key_to_deeplab2


Trainer = gin.external_configurable(Trainer)


@gin.configurable(denylist=['config_path', 'experiment_name', 'neptune_experiment', 'pruned'])
def train(
        # =============================================================================
        # MAIN TRAINING FUNCTION FOR PROTOSEGMENTATION
        # =============================================================================
        # This function implements the complete 3-phase training pipeline:
        # 
        # PHASE 0: WARMUP TRAINING
        #   - Initialize prototypes and add-on layers
        #   - Keep backbone frozen (ResNet101, VGG, etc.)
        #   - Train: Add-on layers, ASPP, prototypes, last layer
        #   - Purpose: Learn meaningful prototypes before joint training
        #
        # PHASE 1: JOINT TRAINING  
        #   - Fine-tune entire network with different learning rates
        #   - Train: ALL components (backbone + add-on + prototypes + last layer)
        #   - Strategy: Lower LR for backbone, higher LR for new components
        #   - Purpose: Joint optimization of all components
        #
        # PHASE 2: PROTOTYPE PUSHING
        #   - Find best representative patches for each prototype
        #   - Replace learned prototypes with actual image patches
        #   - Purpose: Make model interpretable and improve performance
        #
        # PHASE 3: LAST LAYER FINE-TUNING
        #   - Optimize only the final classification layer
        #   - Freeze: Backbone, add-on layers, prototype vectors
        #   - Train: ONLY last layer (classification head)
        #   - Purpose: Fine-tune classification weights for optimal performance
        #
        # The function also handles:
        # - Model initialization and checkpoint loading
        # - Pretrained weight loading (ImageNet or COCO)
        # - Logging setup (TensorBoard, CSV, Neptune)
        # - Pruned model training (skip to phase 3)
        # =============================================================================
        # =============================================================================
        # TRAIN FUNCTION ARGUMENTS EXPLANATION
        # =============================================================================
        
        # === EXPERIMENT CONFIGURATION ===
        config_path: str,                    # Path to gin config file (e.g., 'cityscapes_kld_imnet')
        experiment_name: str,                # Name for this training run (e.g., 'my_experiment_001')
        neptune_experiment: Optional[str] = None,  # Neptune experiment ID for resuming/logging
        pruned: bool = False,                # Whether to train a pruned model (skip warmup/joint phases)
        
        # === CHECKPOINT RESUMPTION ===
        start_checkpoint: str = '',          # Path to checkpoint to resume from (if any)
        
        # === TRAINING HYPERPARAMETERS ===
        random_seed: int = gin.REQUIRED,     # Random seed for reproducibility
        early_stopping_patience_last_layer: int = gin.REQUIRED,  # Patience for early stopping in phase 3
        
        # === TRAINING PHASE STEPS ===
        warmup_steps: int = gin.REQUIRED,    # Number of steps for warmup phase (phase 0)
        joint_steps: int = gin.REQUIRED,     # Number of steps for joint training (phase 1)
        finetune_steps: int = gin.REQUIRED,  # Number of steps for last layer fine-tuning (phase 2)
        
        # === BATCH SIZES ===
        warmup_batch_size: int = gin.REQUIRED,  # Batch size for warmup phase (usually smaller)
        joint_batch_size: int = gin.REQUIRED,   # Batch size for joint training (usually larger)
        
        # === PRETRAINING OPTIONS ===
        load_coco: bool = False              # Whether to load COCO pretrained weights instead of ImageNet
):
    # =============================================================================
    # STEP 1: INITIALIZATION AND SETUP
    # =============================================================================
    
    # Set random seed for reproducibility across all random operations
    seed_everything(random_seed)
    log(f'🌱 Set random seed to {random_seed} for reproducibility')

    # Create results directory for this experiment
    results_dir = os.path.join(os.environ['RESULTS_DIR'], experiment_name)
    os.makedirs(results_dir, exist_ok=True)
    log(f'📁 Created results directory: "{results_dir}"')
    log(f'📋 Using config: {config_path}')

    # Define checkpoint path for resuming training
    last_checkpoint = os.path.join(results_dir, 'checkpoints', 'nopush_best.pth')

    # =============================================================================
    # STEP 2: MODEL INITIALIZATION OR CHECKPOINT LOADING
    # =============================================================================
    
    if start_checkpoint:
        # Option 1: Load from specific checkpoint path
        log(f'🔄 Loading checkpoint from {start_checkpoint}')
        ppnet = torch.load(start_checkpoint)
        pre_loaded = True
        log('✅ Model loaded from specified checkpoint')
    elif neptune_experiment is not None and os.path.exists(last_checkpoint):
        # Option 2: Resume from previous experiment (Neptune integration)
        log(f'🔄 Resuming Neptune experiment from {last_checkpoint}')
        ppnet = torch.load(last_checkpoint)
        pre_loaded = True
        log('✅ Model loaded from previous experiment')
    else:
        # Option 3: Create new model from scratch
        pre_loaded = False
        ppnet = construct_PPNet()
        log('🆕 Created new PPNet model from scratch')

    # =============================================================================
    # STEP 3: PRETRAINED WEIGHTS LOADING (Only for new models)
    # =============================================================================
    
    if not pre_loaded:
        if load_coco:
            # Option 1: Load COCO pretrained weights (better for segmentation)
            log('🌐 Loading COCO pretrained weights for better segmentation performance')
            state_dict = torch.load('deeplab_pytorch/data/models/coco/deeplabv1_resnet101/'
                                    'caffemodel/deeplabv1_resnet101-coco.pth')
            load_result = ppnet.features.base.load_state_dict(state_dict, strict=False)
            log(f'✅ Loaded {len(state_dict)} weights from pretrained COCO model')
            log('   - COCO weights are better for segmentation tasks')
            log('   - Contains more diverse object classes than ImageNet')

            # Expected missing/unexpected keys for COCO loading
            assert len(load_result.missing_keys) == 8  # ASPP layer (has different shape)
            assert len(load_result.unexpected_keys) == 2  # final FC for COCO
            log('✅ COCO weight loading validation passed')
        else:
            # Option 2: Load ImageNet pretrained weights (standard approach)
            log('🖼️ Loading ImageNet pretrained weights from ResNet101')
            resnet_state_dict = torchvision.models.resnet101(pretrained=True).state_dict()
            new_state_dict = {}
            
            # Convert ResNet weight keys to DeepLab format
            for k, v in resnet_state_dict.items():
                new_key = torchvision_resnet_weight_key_to_deeplab2(k)
                if new_key is not None:
                    new_state_dict[new_key] = v

            load_result = ppnet.features.base.load_state_dict(new_state_dict, strict=False)
            log(f'✅ Loaded {len(new_state_dict)} weights from pretrained ResNet101')
            log('   - ImageNet weights provide good general feature representations')

            # Expected missing/unexpected keys for ImageNet loading
            assert len(load_result.missing_keys) == 8  # ASPP layer (has different shape)
            assert len(load_result.unexpected_keys) == 0
            log('✅ ImageNet weight loading validation passed')

        log(f'📊 Weight loading details: {load_result}')

    # =============================================================================
    # STEP 4: LOGGING SETUP
    # =============================================================================
    
    # Create logging directories
    logs_dir = os.path.join(results_dir, 'logs')
    os.makedirs(os.path.join(logs_dir, 'tb'), exist_ok=True)  # TensorBoard logs
    os.makedirs(os.path.join(logs_dir, 'csv'), exist_ok=True)  # CSV logs
    log('📊 Created logging directories')

    # Setup TensorBoard and CSV loggers
    tb_logger = TensorBoardLogger(logs_dir, name='tb')
    csv_logger = CSVLogger(logs_dir, name='csv')
    loggers = [tb_logger, csv_logger]
    log('📈 Initialized TensorBoard and CSV loggers')

    # Get current gin configuration for logging
    json_gin_config = get_operative_config_json()
    tb_logger.log_hyperparams(json_gin_config)
    csv_logger.log_hyperparams(json_gin_config)
    log('📋 Logged hyperparameters to TensorBoard and CSV')

    # =============================================================================
    # STEP 5: NEPTUNE LOGGING SETUP (Optional)
    # =============================================================================
    
    if not pruned:
        use_neptune = bool(int(os.environ['USE_NEPTUNE']))
        if use_neptune:
            log('🌊 Setting up Neptune logging')
            if neptune_experiment is not None:
                # Resume existing Neptune experiment
                neptune_run = neptune.init(
                    project=os.environ['NEPTUNE_PROJECT'],
                    run=neptune_experiment
                )
                neptune_logger = NeptuneLogger(run=neptune_run)
                log(f'🔄 Resumed Neptune experiment: {neptune_experiment}')
            else:
                # Create new Neptune experiment
                neptune_logger = NeptuneLogger(
                    project=os.environ['NEPTUNE_PROJECT'],
                    tags=[config_path, 'segmentation', 'protopnet'],
                    name=experiment_name
                )
                loggers.append(neptune_logger)
                log(f'🆕 Created new Neptune experiment: {experiment_name}')

            # Upload config file and hyperparameters to Neptune
            neptune_run = neptune_logger.run
            neptune_run['config_file'].upload(f'segmentation/configs/{config_path}.gin')
            neptune_run['config'] = json_gin_config
            log('📤 Uploaded config and hyperparameters to Neptune')
        else:
            log('⏭️ Neptune logging disabled (USE_NEPTUNE=0)')

        # Save config file to results directory for reference
        shutil.copy(f'segmentation/configs/{config_path}.gin', os.path.join(results_dir, 'config.gin'))
        log(f'💾 Saved config file to {results_dir}/config.gin')

        # =============================================================================
        # PHASE 1: WARMUP TRAINING
        # =============================================================================
        # Purpose: Initialize prototypes and add-on layers while keeping backbone frozen
        # Trainable: Add-on layers, ASPP layers, prototype vectors, last layer
        # Frozen: Feature extractor backbone (ResNet101, VGG, etc.)
        # Goal: Let prototypes learn meaningful representations before joint training
        # =============================================================================
        if warmup_steps > 0:
            log('🔥 STARTING PHASE 1: WARMUP TRAINING')
            log(f'   - Training steps: {warmup_steps}')
            log(f'   - Batch size: {warmup_batch_size}')
            log('   - Trainable: Add-on layers, ASPP, prototypes, last layer')
            log('   - Frozen: Feature backbone')
            
            # =============================================================================
            # PATCHCLASSIFICATIONDATAMODULE EXPLANATION
            # =============================================================================
            # PatchClassificationDataModule is a PyTorch Lightning DataModule that:
            # 
            # PURPOSE:
            #   - Manages data loading for prototype-based segmentation training
            #   - Handles different data splits (train/val/test) automatically
            #   - Provides data loaders with proper configuration
            #   - Manages multiprocessing for efficient data loading
            #
            # KEY FEATURES:
            #   - Uses PatchClassificationDataset for actual data handling
            #   - Supports different batch sizes for different training phases
            #   - Handles data augmentation and normalization
            #   - Manages train/validation/test splits
            #   - Provides special dataloader for prototype pushing
            #
            # DATA FLOW:
            #   1. Loads images and segmentation masks from disk
            #   2. Applies data augmentation (scaling, flipping, cropping)
            #   3. Normalizes images using ImageNet statistics
            #   4. Converts segmentation masks to class labels
            #   5. Returns batches of (image, label) pairs
            #
            # PATCH CLASSIFICATION APPROACH:
            #   - Treats segmentation as patch classification
            #   - Each pixel/patch gets classified into a semantic class
            #   - Model learns prototypes that represent each class
            #   - Prototypes are matched against image patches
            # =============================================================================
            
            # Create data module with warmup batch size (usually smaller for stability)
            # Returns batches of: (image: [B,3,513,513], label: [B,513,513]) where B=batch_size
            # Labels are pixel-wise class IDs: 0=void, 1=road, 2=sidewalk, ..., 18=car (19 classes total)
            #
            # =============================================================================
            # PATCH SIZE EXPLANATION
            # =============================================================================
            # 
            # INPUT IMAGE SIZE:
            # - Window Size: 513×513 pixels (from config: window_size = (513, 513))
            # - This is the size of input images fed to the model
            # - Images are resized/cropped to this size during preprocessing
            #
            # PROTOTYPE PATCH SIZE:
            # - Prototype Shape: (190, 64, 1, 1) from config
            #   - 190: Number of prototypes
            #   - 64: Feature dimensions per prototype
            #   - 1: Height of prototype patch (1 pixel)
            #   - 1: Width of prototype patch (1 pixel)
            # - Each prototype represents a 1×1 pixel patch
            # - Model treats each pixel as a separate patch for classification
            #
            # PATCH CLASSIFICATION APPROACH:
            # - Each pixel in the 513×513 image is classified independently
            # - Prototypes are 1×1 patches that match against 1×1 image regions
            # - This creates a dense pixel-wise classification (semantic segmentation)
            # - Total of 513×513 = 263,169 pixels to classify per image
            #
            # PROTOTYPE MATCHING:
            # - Each prototype (1×1 patch) is compared to every 1×1 region in the image
            # - Distance is computed between prototype vector and image patch features
            # - Closest matches determine the class prediction for each pixel
            # =============================================================================
            data_module = PatchClassificationDataModule(batch_size=warmup_batch_size)
            
            # Create module with training_phase=0 (warmup phase)
            module = PatchClassificationModule(
                model_dir=results_dir,
                ppnet=ppnet,
                training_phase=0,  # 0 = warmup phase
                max_steps=warmup_steps,
            )
            
            # Create trainer for warmup phase
            trainer = Trainer(logger=loggers, checkpoint_callback=None, enable_progress_bar=False,
                              min_steps=1, max_steps=warmup_steps)
            
            # Start warmup training
            trainer.fit(model=module, datamodule=data_module)
            current_epoch = trainer.current_epoch
        else:
            current_epoch = -1

        # Load the best warmup model for joint training
        last_checkpoint = os.path.join(results_dir, 'checkpoints/warmup_last.pth')
        if os.path.exists(last_checkpoint):
            log(f'Loading model after warmup from {last_checkpoint}')
            ppnet = torch.load(last_checkpoint)
            ppnet = ppnet.cuda()

        # =============================================================================
        # PHASE 2: JOINT TRAINING
        # =============================================================================
        # Purpose: Fine-tune entire network with different learning rates for components
        # Trainable: ALL components (backbone, add-on layers, prototypes, last layer)
        # Learning Rates: Different LR for backbone (lower) vs add-on/prototypes (higher)
        # Goal: Joint optimization of all components with prototype pushing
        # =============================================================================
        log('🔄 STARTING PHASE 2: JOINT TRAINING')
        log(f'   - Training steps: {joint_steps}')
        log(f'   - Batch size: {joint_batch_size}')
        log('   - Trainable: ALL components (backbone + add-on + prototypes + last layer)')
        log('   - Learning rates: Backbone (low) vs Add-on/Prototypes (high)')
        log('   - Scheduler: Polynomial LR decay')
        
        # Create data module with joint batch size (usually larger than warmup)
        data_module = PatchClassificationDataModule(batch_size=joint_batch_size)
        
        # Create module with training_phase=1 (joint phase)
        module = PatchClassificationModule(
            model_dir=results_dir,
            ppnet=ppnet,
            training_phase=1,  # 1 = joint phase
            max_steps=joint_steps
        )
        
        # Create trainer for joint phase
        trainer = Trainer(logger=loggers, checkpoint_callback=None, enable_progress_bar=False,
                          min_steps=1, max_steps=joint_steps)
        trainer.fit_loop.current_epoch = current_epoch + 1
        
        # Start joint training
        trainer.fit(model=module, datamodule=data_module)
        log('✅ Joint training completed successfully')

        # =============================================================================
        # STEP 6: PROTOTYPE PUSHING (After Joint Training)
        # =============================================================================
        # Purpose: Find the best representative patches for each prototype
        # Process: For each prototype, find the training patch that activates it most
        # Result: Prototypes become actual image patches from the training set
        # =============================================================================
        
        log('🎯 STARTING PROTOTYPE PUSHING')
        log('   - Finding best representative patches for each prototype')
        log('   - Prototypes will become actual image patches from training set')
        log('   - This makes the model more interpretable')
        
        # Move model to GPU and set to evaluation mode
        ppnet = ppnet.cuda()
        module.eval()
        torch.set_grad_enabled(False)  # Disable gradients for pushing
        log('🔧 Model set to evaluation mode for prototype pushing')

        # =============================================================================
        # CREATE PUSH DATASET FOR PROTOTYPE PUSHING
        # =============================================================================
        # Purpose: Create special dataset to find the best image patches for each prototype
        # 
        # INPUT:
        #   - split_key='train': Uses training data to search for prototype matches
        #   - is_eval=True: Disables data augmentation for consistent matching
        #   - push_prototypes=True: Enables special prototype pushing mode
        #
        # OUTPUT:
        #   - push_dataset: Dataset containing training images without augmentation
        #   - Each sample: (image: [3,513,513], label: [513,513])
        #   - Used to find actual image patches that best match learned prototypes
        # =============================================================================
        push_dataset = PatchClassificationDataset(
            split_key='train',        # Use training data
            is_eval=True,            # No data augmentation
            push_prototypes=True     # Special mode for prototype pushing
        )
        log(f'📊 Created push dataset with {len(push_dataset)} training samples')

        # =============================================================================
        # PROTOTYPE PUSHING FUNCTION CALL
        # =============================================================================
        # Purpose: Replace learned prototype vectors with actual image patches from training data
        #
        # INPUT PARAMETERS:
        #   - push_dataset: Training dataset to search through for best matches
        #   - prototype_network_parallel: Trained model with learned prototype vectors
        #   - prototype_layer_stride=1: Stride for prototype matching (1x1 patches)
        #   - root_dir_for_saving_prototypes: Directory to save prototype images
        #   - prototype_img_filename_prefix: Prefix for saved prototype images
        #   - prototype_self_act_filename_prefix: Prefix for activation maps
        #   - proto_bound_boxes_filename_prefix: Prefix for bounding box files
        #   - save_prototype_class_identity: Whether to save class information
        #   - pascal: Dataset type flag (Pascal vs Cityscapes)
        #
        # OUTPUT:
        #   - Updated prototype vectors: Replaced with actual image patches
        #   - Prototype images: Saved as .png files showing what each prototype represents
        #   - Activation maps: Visualizations of prototype activations
        #   - Bounding boxes: Location information for each prototype match
        #   - Class identities: Which class each prototype belongs to
        # =============================================================================
        push_prototypes(
            push_dataset,                                    # Dataset to search through
            prototype_network_parallel=ppnet,               # The trained model
            prototype_layer_stride=1,                       # Stride for prototype matching
            root_dir_for_saving_prototypes=module.prototypes_dir,  # Where to save prototype images
            prototype_img_filename_prefix='prototype-img',  # Filename prefix for prototype images
            prototype_self_act_filename_prefix='prototype-self-act',  # Activation maps
            proto_bound_boxes_filename_prefix='bb',         # Bounding box information
            save_prototype_class_identity=True,             # Save which class each prototype belongs to
            pascal=not push_dataset.only_19_from_cityscapes,  # Dataset type (Pascal vs Cityscapes)
            log=log
        )
        log('✅ Prototype pushing completed - prototypes now represent actual image patches')

        # Save the pushed prototypes model
        torch.save(obj=ppnet, f=os.path.join(results_dir, f'checkpoints/push_last.pth'))
        torch.save(obj=ppnet, f=os.path.join(results_dir, f'checkpoints/push_best.pth'))
        log('💾 Saved pushed prototypes model to checkpoints')

        # Reload the model for final fine-tuning
        ppnet = torch.load(os.path.join(results_dir, f'checkpoints/push_last.pth'))
        ppnet = ppnet.cuda()
        log('🔄 Reloaded pushed prototypes model for final fine-tuning')
    else:
        # =============================================================================
        # PRUNED MODEL PATH (Skip warmup and joint training)
        # =============================================================================
        # Purpose: Load a previously pruned model and go directly to fine-tuning
        # Use case: When training a model that has already been pruned
        # =============================================================================
        
        best_checkpoint = os.path.join(results_dir, 'pruned/pruned.pth')
        log(f'✂️ Loading pruned model from {best_checkpoint}')
        log('   - Skipping warmup and joint training phases')
        log('   - Going directly to last layer fine-tuning')
        ppnet = torch.load(best_checkpoint)
        ppnet = ppnet.cuda()
        trainer = None
        log('✅ Pruned model loaded successfully')

        # Setup Neptune logging for pruned model training
        use_neptune = bool(int(os.environ['USE_NEPTUNE']))
        if use_neptune:
            log('🌊 Setting up Neptune logging for pruned model')
            neptune_logger = NeptuneLogger(
                project=os.environ['NEPTUNE_PROJECT'],
                tags=[config_path, 'patch_classification', 'protopnet', 'pruned'],
                name=f'{experiment_name}_pruned' if pruned else experiment_name
            )
            loggers.append(neptune_logger)

            neptune_run = neptune_logger.run
            neptune_run['config_file'].upload(f'segmentation/configs/{config_path}.gin')
            neptune_run['config'] = json_gin_config
            log('📤 Uploaded pruned model config to Neptune')

    # =============================================================================
    # PHASE 3: LAST LAYER FINE-TUNING
    # =============================================================================
    # Purpose: Optimize only the final classification layer after prototype pushing
    # Trainable: ONLY last layer (classification head)
    # Frozen: Feature backbone, add-on layers, prototype vectors
    # Goal: Fine-tune classification weights for optimal performance
    # =============================================================================
    log('🎯 STARTING PHASE 3: LAST LAYER FINE-TUNING')
    log(f'   - Training steps: {finetune_steps}')
    log(f'   - Batch size: {warmup_batch_size}')
    log('   - Trainable: ONLY last layer (classification head)')
    log('   - Frozen: Backbone, add-on layers, prototype vectors')
    log('   - Early stopping: Monitor val/accuracy')
    
    torch.set_grad_enabled(True)
    
    # Setup early stopping callback for last layer training
    callbacks = [
        EarlyStopping(monitor='val/accuracy', patience=early_stopping_patience_last_layer, mode='max')
    ]
    
    # Create data module (use warmup batch size for fine-tuning)
    data_module = PatchClassificationDataModule(batch_size=warmup_batch_size)
    
    # Create module with training_phase=2 (last layer phase)
    module = PatchClassificationModule(
        model_dir=os.path.join(results_dir, 'pruned') if pruned else results_dir,
        ppnet=ppnet,
        training_phase=2,  # 2 = last layer phase
        max_steps=finetune_steps,
    )
    
    current_epoch = trainer.current_epoch if trainer is not None else 0
    trainer = Trainer(logger=loggers, callbacks=callbacks, checkpoint_callback=None,
                      enable_progress_bar=False, max_steps=finetune_steps)
    trainer.fit_loop.current_epoch = current_epoch + 1
    
    # Start last layer fine-tuning
    trainer.fit(model=module, datamodule=data_module)
    log('🎉 TRAINING COMPLETED SUCCESSFULLY!')
    log('   - All three phases completed')
    log('   - Model is ready for evaluation')
    log(f'   - Results saved in: {results_dir}')


# =============================================================================
# CONFIGURATION LOADER AND TRAINING WRAPPER
# =============================================================================
def load_config_and_train(
        config_path: str,                    # Path to gin config file (e.g., 'cityscapes_kld_imnet')
        experiment_name: str,                # Name for this training run
        neptune_experiment: Optional[str] = None,  # Neptune experiment ID (optional)
        pruned: bool = False,                # Whether to train pruned model
        start_checkpoint: str = ''           # Checkpoint to resume from (optional)
):
    """
    =============================================================================
    CONFIGURATION LOADER AND TRAINING WRAPPER
    =============================================================================
    Purpose: Load gin configuration and start training
    Process: 
        1. Parse gin config file to set all hyperparameters
        2. Call the main train() function with loaded configuration
    Usage: This is the main entry point for training
    =============================================================================
    """
    log(f'📋 Loading configuration from: segmentation/configs/{config_path}.gin')
    
    # Parse gin configuration file to set all hyperparameters
    gin.parse_config_file(f'segmentation/configs/{config_path}.gin')
    log('✅ Configuration loaded successfully')
    
    # Start training with the loaded configuration
    train(
        config_path=config_path,
        experiment_name=experiment_name,
        pruned=pruned,
        neptune_experiment=neptune_experiment,
        start_checkpoint=start_checkpoint
    )


if __name__ == '__main__':
    argh.dispatch_command(load_config_and_train)
