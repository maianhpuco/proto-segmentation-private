# 🧬 Prototype Segmentation Architecture Overview

## 🎯 What is Prototype Segmentation?

Prototype segmentation is an **interpretable approach** to semantic segmentation that:

- **Learns prototype vectors** representing characteristic features of each class
- **Treats each pixel** as a separate classification problem  
- **Matches image patches** to learned prototypes for classification
- **Provides interpretability** by showing what each prototype represents

## 🏗️ Overall Architecture

```
Input Image (513×513) 
    ↓
Feature Backbone (DeepLab + ResNet101) - Extracts rich features
    ↓
Add-on Layers (Feature Processing) - Processes backbone features
    ↓
Prototype Matching (190 prototypes × 64 features) - Compares features to prototypes
    ↓
Classification Layer (19 classes) - Maps prototypes to class predictions
    ↓
Segmentation Mask (513×513) - Final pixel-wise predictions
```

## 📁 Repository Structure

```
src_clue_ver1/
├── 📋 config.yaml              # Configuration file (replaces complex .gin)
├── 🚀 train.py                 # Main training script (3-phase pipeline)
├── 🧬 model.py                 # PPNet model definition
├── 🔧 module.py                # Training module (simplified)
├── 📊 dataset.py               # Dataset loading and preprocessing
├── 🏗️ deeplab_features.py      # DeepLab feature extractor
├── 📥 download_voc.py          # Detailed VOC dataset downloader
├── 📥 download_datasets.py     # Quick dataset downloader
├── 🧪 example_usage.py         # Usage examples and testing
├── 📚 README.md                # Detailed documentation
└── 📦 requirements.txt         # Dependencies
```

## 🔧 Key Components Explained

### 1. 🧬 Model (`model.py`)
**Core PPNet implementation:**
- **Prototype Vectors**: 190 learnable feature representations (64 features each)
- **Feature Backbone**: DeepLab v2 with ResNet101 (extracts rich features)
- **Add-on Layers**: Process backbone features for prototype matching
- **Classification Layer**: Maps prototype activations to class predictions
- **L2 Distance**: Measures similarity between image patches and prototypes

### 2. 🚀 Training (`train.py`)
**3-phase training pipeline:**
- **Phase 0 (Warmup)**: Initialize prototypes and add-on layers (15,000 steps)
- **Phase 1 (Joint)**: Train all components with different learning rates (150,000 steps)
- **Phase 2 (Fine-tuning)**: Train only the last layer (10,000 steps)
- **Prototype Pushing**: Replace learned vectors with actual image patches

### 3. 🔧 Training Module (`module.py`)
**Handles training logic:**
- **Loss Computation**: Cross-entropy + L1 regularization + KLD loss
- **Optimization**: Different learning rates for different components
- **Metrics**: Training/validation accuracy tracking
- **Checkpointing**: Save/load model states

### 4. 📊 Dataset (`dataset.py`)
**Data loading and preprocessing:**
- **Supported Datasets**: Cityscapes (19 classes) and PASCAL VOC (21 classes)
- **Data Augmentation**: Random crop, flip, scaling
- **Normalization**: ImageNet statistics
- **Class Mapping**: Converts dataset-specific class IDs to model classes

### 5. 🏗️ Feature Extractor (`deeplab_features.py`)
**DeepLab v2 with ResNet101:**
- **ASPP Module**: Atrous Spatial Pyramid Pooling for multi-scale features
- **ResNet101 Backbone**: Pretrained on ImageNet
- **Output**: 64-channel features ready for prototype matching

## 🎯 Training Phases Detailed

### 📚 Phase 0: Warmup (15,000 steps)
```python
# What's trainable:
- Prototypes: ✅ (LR: 0.00025)
- Add-on layers: ✅ (LR: 0.00025)  
- Last layer: ✅ (LR: 0.00025)
- Backbone: ❌ (Frozen)

# Goal: Learn meaningful prototype representations
```

### 🔄 Phase 1: Joint Training (150,000 steps)
```python
# What's trainable:
- Prototypes: ✅ (LR: 0.00025)
- Add-on layers: ✅ (LR: 0.00025)
- Last layer: ✅ (LR: 0.00025)
- Backbone: ✅ (LR: 0.000025) - Lower LR for pretrained features

# Goal: Joint optimization of all components
# Includes: Prototype pushing
```

### 🎯 Phase 2: Fine-tuning (10,000 steps)
```python
# What's trainable:
- Prototypes: ❌ (Frozen)
- Add-on layers: ❌ (Frozen)
- Last layer: ✅ (LR: 0.00001)
- Backbone: ❌ (Frozen)

# Goal: Fine-tune classification weights
```

## 🔧 Loss Function Components

```python
Total Loss = λ₁ × CrossEntropy + λ₂ × L1 + λ₃ × KLD

Where:
- λ₁ = 1.0 (Cross-entropy weight)
- λ₂ = 0.0001 (L1 regularization weight)  
- λ₃ = 0.25 (KLD diversity weight)
```

### Loss Components:
1. **Cross-Entropy Loss**: Main segmentation loss
2. **L1 Regularization**: Encourages sparse prototype activations
3. **KLD Loss**: Encourages prototype diversity

## 📊 Supported Datasets

### 🌆 Cityscapes Dataset
- **Classes**: 19 evaluation classes
- **Images**: 2975 train, 500 val
- **Resolution**: 1024×2048
- **Classes**: road, sidewalk, building, wall, fence, pole, traffic_light, traffic_sign, vegetation, terrain, sky, person, rider, car, truck, bus, train, motorcycle, bicycle

### 🏷️ PASCAL VOC 2012
- **Classes**: 21 classes (20 objects + background)
- **Images**: 1464 train, 1449 val
- **Resolution**: ~500×500
- **Classes**: person, bird, cat, cow, dog, horse, sheep, airplane, bicycle, boat, bus, car, motorcycle, train, bottle, chair, dining table, potted plant, sofa, tv/monitor, background

## 🚀 How to Use

### 1. Setup Environment
```bash
pip install torch torchvision numpy pyyaml tqdm pillow
```

### 2. Download Dataset
```bash
# For VOC
python download_voc.py

# For Cityscapes (manual download required)
# Place in data/cityscapes/ directory
```

### 3. Configure Training
Edit `config.yaml`:
```yaml
dataset:
  name: "cityscapes"  # or "pascal"
  window_size: [513, 513]

model:
  prototype_shape: [190, 64, 1, 1]
  num_classes: 19
```

### 4. Run Training
```bash
# Run all phases
python train.py

# Run specific phase
python train.py --phase 1

# Resume from checkpoint
python train.py --resume models/warmup_checkpoint.pth
```

## 🧪 Testing the Code

```bash
# Run example to test everything works
python example_usage.py
```

This will:
- Create a dummy model
- Test all training phases
- Verify checkpoint saving/loading
- Ensure the pipeline works correctly

## 🔍 Key Concepts

### Prototypes
- **What**: Learnable vectors representing characteristic features
- **Shape**: (190, 64, 1, 1) - 190 prototypes × 64 features
- **Purpose**: Capture typical patterns for each class
- **Interpretability**: Can visualize what each prototype represents

### L2 Distance
- **What**: Measures similarity between image patches and prototypes
- **Formula**: ||patch - prototype||₂
- **Purpose**: Find which prototype best matches each image patch
- **Activation**: Convert distances to activation scores

### Pixel-wise Classification
- **What**: Treat each pixel as independent classification problem
- **Process**: Extract features → Match to prototypes → Classify
- **Output**: Class prediction for each pixel
- **Result**: Dense segmentation mask

## 🎯 Expected Results

### Performance
- **Training Time**: ~24-48 hours on single GPU
- **Memory Usage**: ~8-12 GB GPU memory
- **Final Accuracy**: ~70-75% mIoU (Cityscapes)

### Model Size
- **Parameters**: ~50M parameters
- **Checkpoint Size**: ~200MB per checkpoint
- **Prototypes**: 190 prototypes × 64 features

## 🔧 Customization

### Adding New Datasets
1. Modify `dataset.py` to add new dataset class
2. Update class mappings in `_setup_class_mappings()`
3. Add data loading logic in `_get_data_paths()`

### Changing Architecture
1. Modify `deeplab_features.py` for different backbones
2. Update `model.py` for different prototype configurations
3. Adjust `config.yaml` for new parameters

### Training Modifications
1. Edit `config.yaml` for different learning rates
2. Modify `module.py` for different loss functions
3. Update `train.py` for different training schedules

## 🐛 Troubleshooting

### Common Issues
1. **CUDA Out of Memory**: Reduce batch size in config
2. **Dataset Not Found**: Check data directory structure
3. **Training Diverges**: Reduce learning rates
4. **Import Errors**: Install missing dependencies

### Performance Tips
1. **Use GPU**: Ensure CUDA is available
2. **Adjust Batch Size**: Balance memory and speed
3. **Monitor Metrics**: Watch training/validation accuracy
4. **Save Checkpoints**: Regular checkpointing for recovery

---

**Happy Learning! 🎉**

This simplified version makes prototype segmentation accessible while maintaining the core interpretability benefits of the original approach.
