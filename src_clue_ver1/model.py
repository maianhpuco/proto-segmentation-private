"""
=============================================================================
SIMPLIFIED PROTOTYPE SEGMENTATION MODEL
=============================================================================

USAGE INSTRUCTIONS:
1. This file contains the PPNet (Prototype-based Network) model
2. The model learns prototype vectors that represent characteristic features of each class
3. For semantic segmentation, it treats each pixel as a separate classification problem
4. Main components:
   - Feature backbone (ResNet, VGG, etc.)
   - Add-on layers (process features)
   - Prototype vectors (learnable feature representations)
   - Classification layer (maps prototypes to classes)

HOW TO USE:
- Import: from model import PPNet, construct_PPNet
- Create model: model = construct_PPNet(config)
- Forward pass: predictions = model(images)

CONFIGURATION:
- All parameters are configured via config.yaml
- No need to modify this file for different experiments
=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


class PPNet(nn.Module):
    """
    Simplified ProtoPNet for semantic segmentation
    """
    def __init__(self, features, img_size, prototype_shape, num_classes, 
                 prototype_activation_function='log', add_on_layers_type='deeplab_simple'):
        super(PPNet, self).__init__()
        
        # Store configuration
        self.img_size = img_size
        self.num_classes = num_classes
        self.prototype_activation_function = prototype_activation_function
        
        # Prototype vectors - learnable feature representations
        # Shape: (num_prototypes, feature_dim, height, width)
        self.prototype_vectors = nn.Parameter(torch.rand(prototype_shape), requires_grad=True)
        
        # Prototype class identity - maps each prototype to a class
        # Equal distribution of prototypes per class
        self.num_prototypes = prototype_shape[0]
        num_prototypes_per_class = self.num_prototypes // self.num_classes
        
        self.prototype_class_identity = torch.zeros(self.num_prototypes, num_classes)
        for i in range(self.num_classes):
            start_idx = i * num_prototypes_per_class
            end_idx = (i + 1) * num_prototypes_per_class
            self.prototype_class_identity[start_idx:end_idx, i] = 1
        
        self.num_prototypes_per_class = num_prototypes_per_class
        
        # Feature backbone (ResNet, VGG, etc.)
        self.features = features
        
        # Add-on layers - process backbone features
        self._create_add_on_layers(prototype_shape[1], add_on_layers_type)
        
        # Last layer - maps prototype activations to class predictions
        self.last_layer = nn.Linear(self.num_prototypes, self.num_classes, bias=False)
        
        # Ones tensor for distance computation
        self.ones = nn.Parameter(torch.ones(prototype_shape), requires_grad=False)
        
        # Initialize weights
        self._initialize_weights()
    
    def _create_add_on_layers(self, prototype_feature_dim, add_on_layers_type):
        """Create add-on layers based on backbone architecture"""
        # Get output channels from backbone
        if hasattr(self.features, 'out_channels'):
            first_add_on_layer_in_channels = self.features.out_channels
        else:
            # Default for DeepLab
            first_add_on_layer_in_channels = 64
        
        if add_on_layers_type == 'deeplab_simple':
            # Simple add-on layers for DeepLab
            self.add_on_layers = nn.Sequential(nn.Sigmoid())
        else:
            # Default add-on layers
            self.add_on_layers = nn.Sequential(
                nn.Conv2d(first_add_on_layer_in_channels, prototype_feature_dim, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(prototype_feature_dim, prototype_feature_dim, kernel_size=1),
                nn.Sigmoid()
            )
    
    def _initialize_weights(self):
        """Initialize model weights"""
        # Initialize last layer weights
        nn.init.xavier_uniform_(self.last_layer.weight)
        
        # Initialize prototype vectors
        nn.init.normal_(self.prototype_vectors, mean=0.0, std=0.1)
    
    def conv_features(self, x):
        """Extract features for prototype matching"""
        # Extract features using backbone
        x = self.features(x)
        
        # Process through add-on layers
        x = self.add_on_layers(x)
        return x
    
    def _l2_convolution(self, input, filter):
        """Compute L2 distance between input and filter"""
        # Reshape for convolution
        input_squared = input ** 2
        input_patch_sums = F.conv2d(input_squared, self.ones, stride=1, padding=0)
        
        filter_squared = filter ** 2
        filter_sums = torch.sum(filter_squared, dim=(1, 2, 3), keepdim=True)
        
        # Compute cross-correlation
        cross_correlation = F.conv2d(input, filter, stride=1, padding=0)
        
        # Compute L2 distance
        distances = input_patch_sums - 2 * cross_correlation + filter_sums
        return distances
    
    def _prototype_activations(self, distances):
        """Convert distances to prototype activations"""
        if self.prototype_activation_function == 'log':
            return torch.log((distances + 1) / (distances + self.epsilon))
        elif self.prototype_activation_function == 'linear':
            return -distances
        else:
            return self.prototype_activation_function(distances)
    
    def forward(self, x):
        """Forward pass through the model"""
        # Extract features
        conv_features = self.conv_features(x)
        
        # Compute distances between features and prototypes
        distances = self._l2_convolution(conv_features, self.prototype_vectors)
        
        # Convert distances to activations
        prototype_activations = self._prototype_activations(distances)
        
        # Global max pooling to get prototype activations per image
        global_max_pooled_activations = F.max_pool2d(prototype_activations, 
                                                   kernel_size=prototype_activations.size()[2:])
        global_max_pooled_activations = global_max_pooled_activations.view(-1, self.num_prototypes)
        
        # Classify using last layer
        logits = self.last_layer(global_max_pooled_activations)
        
        return logits, prototype_activations
    
    def forward_segmentation(self, x):
        """Forward pass for segmentation (pixel-wise classification)"""
        # Extract features
        conv_features = self.conv_features(x)
        
        # Compute distances between features and prototypes
        distances = self._l2_convolution(conv_features, self.prototype_vectors)
        
        # Convert distances to activations
        prototype_activations = self._prototype_activations(distances)
        
        # Reshape for classification
        batch_size, num_prototypes, h, w = prototype_activations.shape
        prototype_activations_flat = prototype_activations.view(batch_size, num_prototypes, -1)
        prototype_activations_flat = prototype_activations_flat.permute(0, 2, 1)  # [B, H*W, P]
        
        # Classify each pixel
        logits_flat = self.last_layer(prototype_activations_flat)  # [B, H*W, C]
        logits = logits_flat.permute(0, 2, 1).view(batch_size, self.num_classes, h, w)
        
        return logits, prototype_activations


def construct_PPNet(config):
    """
    Construct PPNet model from configuration
    
    Args:
        config: Configuration dictionary from YAML file
    
    Returns:
        PPNet model instance
    """
    model_config = config['model']
    training_config = config['training']
    dataset_config = config['dataset']
    
    # Create feature backbone (simplified)
    if model_config['base_architecture'] == 'deeplabv2_resnet101':
        from deeplab_features import deeplabv2_resnet101_features
        features = deeplabv2_resnet101_features(pretrained=model_config['pretrained'])
    else:
        raise NotImplementedError(f"Architecture {model_config['base_architecture']} not implemented")
    
    # Create PPNet model
    model = PPNet(
        features=features,
        img_size=dataset_config['window_size'][0],  # Use height as img_size
        prototype_shape=model_config['prototype_shape'],
        num_classes=model_config['num_classes'],
        prototype_activation_function=model_config['prototype_activation_function'],
        add_on_layers_type=model_config['add_on_layers_type']
    )
    
    return model


# Example usage
if __name__ == "__main__":
    import yaml
    
    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create model
    model = construct_PPNet(config)
    print(f"Model created with {model.num_prototypes} prototypes")
    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
