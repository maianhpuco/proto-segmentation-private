# 🧬 Simplified Prototype Segmentation

A simplified but accurate implementation of prototype-based semantic segmentation using PyTorch. This version replaces the complex GIN configuration system with easy-to-understand YAML configuration and removes PyTorch Lightning dependencies for better clarity.

## 🎯 What is Prototype Segmentation?

Prototype segmentation is an interpretable approach to semantic segmentation that:
- **Learns prototype vectors** representing characteristic features of each class
- **Treats each pixel** as a separate classification problem
- **Matches image patches** to learned prototypes for classification
- **Provides interpretability** by showing what each prototype represents

## 🏗️ Architecture Overview

```
Input Image (513×513) 
    ↓
Feature Backbone (DeepLab + ResNet101)
    ↓
Add-on Layers (Feature Processing)
    ↓
Prototype Matching (190 prototypes × 64 features)
    ↓
Classification Layer (19 classes)
    ↓
Segmentation Mask (513×513)
```

## 📁 Project Structure

```
src_clue_ver1/
├── README.md              # This file
├── config.yaml            # Configuration file (replaces .gin)
├── train.py               # Main training script
├── model.py               # PPNet model definition
├── module.py              # Training module (simplified)
├── dataset.py             # Dataset loading and preprocessing
├── deeplab_features.py    # DeepLab feature extractor
├── download_voc.py        # Detailed VOC dataset downloader
├── download_datasets.py   # Quick dataset downloader
└── example_usage.py       # Usage examples and testing
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch torchvision numpy pyyaml tqdm pillow
```

### 2. Prepare Dataset

#### Option A: Download PASCAL VOC 2012 (Recommended for testing)

```bash
# Quick download
python download_datasets.py voc

# Or use the detailed downloader
python download_voc.py
```

#### Option B: Download Cityscapes (For full training)

```bash
# Create data directory structure
mkdir -p data/cityscapes/leftImg8bit/train
mkdir -p data/cityscapes/leftImg8bit/val
mkdir -p data/cityscapes/gtFine/train
mkdir -p data/cityscapes/gtFine/val

# Download and extract Cityscapes dataset
# Place images in data/cityscapes/leftImg8bit/
# Place masks in data/cityscapes/gtFine/
```

### 3. Configure Training

Edit `config.yaml` to match your setup:

```yaml
# Dataset paths
paths:
  data_dir: "./data"           # Your dataset directory
  results_dir: "./results"     # Results directory
  model_dir: "./models"        # Model checkpoints directory

# Model configuration
model:
  prototype_shape: [190, 64, 1, 1]  # [num_prototypes, features, height, width]
  num_classes: 19                   # Number of classes
```

### 4. Run Training

```bash
# Train complete pipeline (all 3 phases)
python train.py

# Train specific phase
python train.py --phase 0  # Warmup only
python train.py --phase 1  # Joint training only
python train.py --phase 2  # Fine-tuning only

# Resume from checkpoint
python train.py --resume models/warmup_checkpoint.pth

# Skip prototype pushing
python train.py --skip_push
```

## 📋 Training Phases

### Phase 0: Warmup (15,000 steps)
- **Purpose**: Initialize prototypes and add-on layers
- **Trainable**: Prototypes, add-on layers, last layer
- **Frozen**: Feature backbone (ResNet101)
- **Goal**: Learn meaningful prototype representations

### Phase 1: Joint Training (150,000 steps)
- **Purpose**: Fine-tune entire network
- **Trainable**: All components with different learning rates
- **Strategy**: Lower LR for backbone, higher LR for new components
- **Goal**: Joint optimization of all components

### Phase 2: Fine-tuning (10,000 steps)
- **Purpose**: Optimize final classification
- **Trainable**: Only last layer
- **Frozen**: Everything else
- **Goal**: Fine-tune classification weights

## ⚙️ Configuration

The `config.yaml` file contains all configuration parameters:

### Model Configuration
```yaml
model:
  base_architecture: "deeplabv2_resnet101"  # Backbone network
  prototype_shape: [190, 64, 1, 1]         # Prototype tensor shape
  num_classes: 19                           # Number of output classes
  prototype_activation_function: "log"      # Activation function
```

### Training Configuration
```yaml
training:
  warmup_steps: 15000      # Warmup phase steps
  joint_steps: 150000      # Joint training steps
  finetune_steps: 10000    # Fine-tuning steps
  
  # Learning rates
  warmup_lr_add_on_layers: 0.00025
  joint_lr_features: 0.000025
  last_layer_lr: 0.00001
  
  # Loss weights
  loss_weight_cross_entropy: 1.0
  loss_weight_l1: 0.0001
  loss_weight_kld: 0.25
```

### Dataset Configuration
```yaml
dataset:
  name: "cityscapes"                    # Dataset name
  window_size: [513, 513]              # Input image size
  mean: [0.485, 0.456, 0.406]         # Normalization mean
  std: [0.229, 0.224, 0.225]          # Normalization std
```

## 📊 Key Features

### 🎯 Interpretability
- **Visual Prototypes**: See exactly what each prototype represents
- **Patch Matching**: Understand which image patches activate prototypes
- **Class Attribution**: Know which prototypes belong to which classes

### 🔧 Flexibility
- **Multiple Architectures**: Support for ResNet, VGG, DenseNet backbones
- **Configurable Parameters**: Easy modification via YAML
- **Different Datasets**: Support for Cityscapes, PASCAL VOC

### ⚡ Efficiency
- **1×1 Patch Size**: Dense pixel-wise classification
- **Prototype Pruning**: Remove redundant prototypes
- **Gradient Accumulation**: Handle large batch sizes

## 🧪 Usage Examples

### Basic Training
```python
import yaml
from train import main

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Run training
main()
```

### Custom Model Creation
```python
import yaml
from model import construct_PPNet

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Create model
model = construct_PPNet(config)

# Test with dummy input
import torch
dummy_input = torch.randn(1, 3, 513, 513)
logits, activations = model.forward_segmentation(dummy_input)
print(f"Output shape: {logits.shape}")
```

### Custom Dataset
```python
import yaml
from dataset import PatchClassificationDataset

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Create dataset
dataset = PatchClassificationDataset('train', config)

# Get sample
image, mask = dataset[0]
print(f"Image shape: {image.shape}, Mask shape: {mask.shape}")
```

## 🔍 Understanding the Code

### File-by-File Guide

1. **`config.yaml`**: All configuration parameters in YAML format
2. **`train.py`**: Main training script with 3-phase pipeline
3. **`model.py`**: PPNet model definition and construction
4. **`module.py`**: Training module with loss computation and optimization
5. **`dataset.py`**: Dataset loading and preprocessing
6. **`deeplab_features.py`**: DeepLab feature extractor

### Key Concepts

- **Prototypes**: Learnable vectors representing class features
- **Patch Classification**: Treating segmentation as pixel-wise classification
- **3-Phase Training**: Gradual unfreezing of model components
- **Prototype Pushing**: Replacing learned vectors with actual image patches

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size in `config.yaml`
   - Use gradient accumulation (`iter_size` parameter)

2. **Dataset Not Found**
   - Check data directory structure
   - Verify dataset paths in `config.yaml`

3. **Training Diverges**
   - Reduce learning rates
   - Check loss weights
   - Verify data preprocessing

### Performance Tips

1. **Use GPU**: Ensure CUDA is available
2. **Adjust Batch Size**: Balance memory usage and training speed
3. **Monitor Metrics**: Watch training/validation accuracy
4. **Save Checkpoints**: Regular checkpointing for recovery

## 📈 Expected Results

### Cityscapes Dataset
- **Training Time**: ~24-48 hours on single GPU
- **Memory Usage**: ~8-12 GB GPU memory
- **Final Accuracy**: ~70-75% mIoU (depending on configuration)

### Model Size
- **Parameters**: ~50M parameters
- **Checkpoint Size**: ~200MB per checkpoint
- **Prototypes**: 190 prototypes × 64 features

## 🤝 Contributing

This is a simplified educational version. For production use, consider:
- Adding more robust error handling
- Implementing advanced data augmentation
- Adding more backbone architectures
- Optimizing memory usage

## 📚 References

- Original ProtoPNet paper
- DeepLab v2 architecture
- Cityscapes dataset
- PyTorch documentation

## 📄 License

This simplified version is for educational purposes. Please refer to the original repository for licensing information.

---

**Happy Learning! 🎉**

For questions or issues, please refer to the code comments and configuration examples provided in each file.
