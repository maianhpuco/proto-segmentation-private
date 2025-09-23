"""
=============================================================================
SIMPLIFIED PROTOTYPE SEGMENTATION MODEL
=============================================================================

🎯 WHAT IS PROTOTYPE SEGMENTATION?
Prototype segmentation is an interpretable approach to semantic segmentation that:
- Learns "prototype vectors" representing characteristic features of each class
- Treats each pixel as a separate classification problem
- Matches image patches to learned prototypes for classification
- Provides interpretability by showing what each prototype represents

🏗️ ARCHITECTURE OVERVIEW:
Input Image (513×513) 
    ↓
Feature Backbone (DeepLab + ResNet101) - Extracts features
    ↓
Add-on Layers (Feature Processing) - Processes backbone features
    ↓
Prototype Matching (190 prototypes × 64 features) - Compares features to prototypes
    ↓
Classification Layer (19 classes) - Maps prototypes to class predictions
    ↓
Segmentation Mask (513×513) - Final pixel-wise predictions

🔧 MAIN COMPONENTS:
1. Feature Backbone: DeepLab v2 with ResNet101 (extracts rich features)
2. Add-on Layers: Process backbone features for prototype matching
3. Prototype Vectors: Learnable feature representations (190 prototypes × 64 features)
4. Classification Layer: Maps prototype activations to class predictions

📚 USAGE INSTRUCTIONS:
- Import: from model import PPNet, construct_PPNet
- Create model: model = construct_PPNet(config)
- Forward pass: predictions = model(images)
- Segmentation: logits, activations = model.forward_segmentation(images)

⚙️ CONFIGURATION:
- All parameters are configured via config.yaml
- No need to modify this file for different experiments
- Key config parameters:
  * prototype_shape: [190, 64, 1, 1] (num_prototypes, features, height, width)
  * num_classes: 19 (number of output classes)
  * base_architecture: "deeplabv2_resnet101" (backbone network)

🧠 KEY CONCEPTS:
- Prototypes: Learnable vectors that represent characteristic features
- L2 Distance: Measures similarity between image patches and prototypes
- Prototype Activation: Converts distances to activation scores
- Pixel-wise Classification: Each pixel is classified independently
=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


class PPNet(nn.Module):
    """
    🧬 Simplified ProtoPNet for semantic segmentation
    
    This is the main model class that implements prototype-based segmentation.
    It learns prototype vectors that represent characteristic features of each class
    and uses them to classify each pixel in the image.
    
    Key Features:
    - Learnable prototype vectors (190 prototypes × 64 features)
    - L2 distance-based prototype matching
    - Pixel-wise classification for segmentation
    - Interpretable predictions through prototype activations
    """
    def __init__(self, features, img_size, prototype_shape, num_classes, 
                 prototype_activation_function='log', add_on_layers_type='deeplab_simple'):
        super(PPNet, self).__init__()
        
        # 📋 Store configuration parameters
        self.img_size = img_size                    # Input image size (e.g., 513)
        self.num_classes = num_classes              # Number of output classes (e.g., 19)
        self.prototype_activation_function = prototype_activation_function  # Activation function ('log' or 'linear')
        self.epsilon = 1e-4                         # Small constant for numerical stability in log activation
        
        # 🎯 Prototype vectors - the core of the model
        # These are learnable feature representations that capture characteristic patterns
        # Shape: (num_prototypes, feature_dim, height, width) = (190, 64, 1, 1)
        self.prototype_vectors = nn.Parameter(torch.rand(prototype_shape), requires_grad=True)
        
        # 🏷️ Prototype class identity - maps each prototype to a class
        # This ensures each prototype belongs to a specific class
        # Equal distribution: 190 prototypes ÷ 19 classes = 10 prototypes per class
        self.num_prototypes = prototype_shape[0]
        num_prototypes_per_class = self.num_prototypes // self.num_classes
        
        # Create identity matrix: prototype_class_identity[i, j] = 1 if prototype i belongs to class j
        self.prototype_class_identity = torch.zeros(self.num_prototypes, num_classes)
        for i in range(self.num_classes):
            start_idx = i * num_prototypes_per_class
            end_idx = (i + 1) * num_prototypes_per_class
            self.prototype_class_identity[start_idx:end_idx, i] = 1
        
        self.num_prototypes_per_class = num_prototypes_per_class
        
        # 🏗️ Feature backbone (DeepLab + ResNet101) - extracts rich features from images
        self.features = features
        
        # 🔧 Add-on layers - process backbone features for prototype matching
        self._create_add_on_layers(prototype_shape[1], add_on_layers_type)
        
        # 🎯 Last layer - maps prototype activations to class predictions
        # Input: prototype activations (190 values)
        # Output: class logits (19 values)
        self.last_layer = nn.Linear(self.num_prototypes, self.num_classes, bias=False)
        
        # Note: We no longer need the ones tensor since we compute input patch sums directly
        
        # 🎲 Initialize weights properly
        self._initialize_weights()
    
    def _create_add_on_layers(self, prototype_feature_dim, add_on_layers_type):
        """
        🔧 Create add-on layers based on backbone architecture
        
        PURPOSE:
        The add-on layers are crucial for preparing backbone features for prototype matching.
        They transform the backbone features into a format suitable for prototype comparison.
        2
        WHY WE NEED ADD-ON LAYERS:
        1. Feature Dimension Matching: Backbone outputs may not match prototype feature dimensions
        2. Feature Processing: Transform features to be more suitable for prototype matching
        3. Activation Normalization: Apply sigmoid to ensure features are in [0,1] range
        4. Dimensionality Reduction: Reduce feature channels if needed
        
        ARCHITECTURE OPTIONS:
        - 'deeplab_simple': Just applies sigmoid (minimal processing)
        - 'bottleneck': Full processing with conv layers (more complex)
        
        Args:
            prototype_feature_dim (int): Target feature dimension for prototypes (e.g., 64)
            add_on_layers_type (str): Type of add-on layers ('deeplab_simple' or 'bottleneck')
        """
        # 🔍 Get output channels from backbone
        # This determines how many input channels the add-on layers need to handle
        if hasattr(self.features, 'out_channels'):
            # If backbone has out_channels attribute, use it
            first_add_on_layer_in_channels = self.features.out_channels
        else:
            # Default for DeepLab (should be 64 channels after ASPP)
            first_add_on_layer_in_channels = 64
        
        if add_on_layers_type == 'deeplab_simple':
            # 🎯 SIMPLE ADD-ON LAYERS (Recommended for DeepLab)
            # This is the minimal processing approach:
            # - Just applies sigmoid activation to normalize features to [0,1]
            # - Assumes backbone features are already suitable for prototype matching
            # - Faster and simpler, works well with DeepLab's ASPP features
            self.add_on_layers = nn.Sequential(nn.Sigmoid())
            
        else:
            # 🏗️ DEFAULT ADD-ON LAYERS (More complex processing)
            # This is a more sophisticated approach with two conv layers:
            # 1. First conv: Projects backbone features to prototype feature dimension
            # 2. ReLU: Adds non-linearity
            # 3. Second conv: Further refines features (same input/output dims)
            # 4. Sigmoid: Normalizes to [0,1] range for prototype matching
            self.add_on_layers = nn.Sequential(
                nn.Conv2d(first_add_on_layer_in_channels, prototype_feature_dim, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(prototype_feature_dim, prototype_feature_dim, kernel_size=1),
                nn.Sigmoid()
            )
        
        # 📊 Layer Summary:
        # - Input: Backbone features (e.g., 64 channels from DeepLab ASPP)
        # - Output: Processed features (64 channels) ready for prototype matching
        # - Activation: Sigmoid ensures features are in [0,1] range
        # - Purpose: Bridge between backbone features and prototype vectors
    
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
        """
        📐 Compute L2 distance between input features and prototype filters
        
        PURPOSE:
        This is the core of prototype matching! It efficiently computes the L2 distance
        between every patch in the input features and every prototype vector.
        
        MATHEMATICAL FORMULA:
        L2_distance = ||input_patch - prototype||²
        Expanded: ||a - b||² = ||a||² + ||b||² - 2⟨a,b⟩
        
        EFFICIENT IMPLEMENTATION:
        Instead of computing distances for each patch individually, we use convolution
        operations to compute all distances simultaneously across the entire feature map.
        
        Args:
            input: Feature map from add-on layers [B, C, H, W]
            filter: Prototype vectors [P, C, 1, 1] where P = num_prototypes
        
        Returns:
            distances: L2 distances [B, P, H, W] for each prototype at each location
        """
        # 🔢 Step 1: Compute ||input_patch||² for all patches
        # Square each element in the input feature map
        input_squared = input ** 2
        
        # Sum across channels to get ||input_patch||² for each spatial location
        # input: [B, C, H, W] -> input_patch_sums: [B, 1, H, W]
        input_patch_sums = torch.sum(input_squared, dim=1, keepdim=True)
        
        # 🔢 Step 2: Compute ||prototype||² for all prototypes
        # Square each element in the prototype vectors
        filter_squared = filter ** 2
        
        # Sum across all dimensions (channels, height, width) for each prototype
        # This gives us ||prototype||² for each prototype
        # filter: [P, C, 1, 1] -> filter_sums: [P, 1, 1, 1]
        filter_sums = torch.sum(filter_squared, dim=(1, 2, 3), keepdim=True)
        
        # 🔢 Step 3: Compute cross-correlation ⟨input_patch, prototype⟩
        # This is the dot product between input patches and prototypes
        # F.conv2d efficiently computes this for all patch locations simultaneously
        # input: [B, C, H, W], filter: [P, C, 1, 1] -> cross_correlation: [B, P, H, W]
        cross_correlation = F.conv2d(input, filter, stride=1, padding=0)
        
        # 🔢 Step 4: Combine using the expanded L2 distance formula
        # ||a - b||² = ||a||² + ||b||² - 2⟨a,b⟩
        # input_patch_sums: [B, 1, H, W], filter_sums: [P, 1, 1, 1], cross_correlation: [B, P, H, W]
        # We need to expand input_patch_sums to [B, P, H, W] for broadcasting
        input_patch_sums_expanded = input_patch_sums.expand(-1, filter.shape[0], -1, -1)
        filter_sums_expanded = filter_sums.view(1, -1, 1, 1).expand(input.shape[0], -1, input.shape[2], input.shape[3])
        
        distances = input_patch_sums_expanded - 2 * cross_correlation + filter_sums_expanded
        
        # 📊 Result: distances[b, p, h, w] = L2 distance between
        # input patch at location (h,w) in batch b and prototype p
        return distances
    
    def _prototype_activations(self, distances):
        """
        🎯 Convert L2 distances to prototype activation scores
        
        PURPOSE:
        This function converts the L2 distances (which measure dissimilarity) into
        activation scores (which measure similarity). Lower distances should result
        in higher activations, indicating better prototype matches.
        
        ACTIVATION FUNCTIONS:
        
        1. LOG ACTIVATION (Recommended):
           activation = log((distance + 1) / (distance + epsilon))
           - Converts distances to log-scale activations
           - Higher activations for lower distances
           - Smooth, differentiable function
           - Helps with training stability
        
        2. LINEAR ACTIVATION:
           activation = -distance
           - Simple negative of distance
           - Linear relationship between distance and activation
           - Direct but less stable for training
        
        Args:
            distances: L2 distances [B, P, H, W] from _l2_convolution
        
        Returns:
            activations: Prototype activation scores [B, P, H, W]
                        Higher values = better prototype matches
        """
        if self.prototype_activation_function == 'log':
            # 🎯 LOG ACTIVATION (Recommended)
            # Formula: log((distance + 1) / (distance + epsilon))
            # 
            # Why this works:
            # - When distance = 0: activation = log(1/epsilon) = high positive value
            # - When distance = ∞: activation = log(1) = 0
            # - Smooth, differentiable, and stable for training
            # - epsilon prevents division by zero and log(0)
            return torch.log((distances + 1) / (distances + self.epsilon))
            
        elif self.prototype_activation_function == 'linear':
            # 🎯 LINEAR ACTIVATION
            # Formula: -distance
            # 
            # Why this works:
            # - Direct negative relationship: lower distance → higher activation
            # - Simple and intuitive
            # - Can be less stable during training
            return -distances
            
        else:
            # 🎯 CUSTOM ACTIVATION FUNCTION
            # Allow for custom activation functions if provided
            return self.prototype_activation_function(distances)
    
    def forward(self, x):
        """
        🚀 Forward pass through the model (Image Classification)
        
        PURPOSE:
        This method performs image-level classification using prototype matching.
        It finds the best prototype match for each image and classifies accordingly.
        
        PROCESS:
        1. Extract features from input image
        2. Compute prototype distances for all locations
        3. Convert distances to activation scores
        4. Use global max pooling to find best prototype match per image
        5. Classify using the last layer
        
        Args:
            x: Input images [B, 3, H, W]
        
        Returns:
            logits: Class predictions [B, num_classes]
            prototype_activations: Prototype activations [B, P, H, W]
        """
        # 🏗️ Step 1: Extract features using backbone + add-on layers
        conv_features = self.conv_features(x)
        
        # 📐 Step 2: Compute L2 distances between features and prototypes
        # Result: distances[b, p, h, w] = distance between patch at (h,w) and prototype p
        distances = self._l2_convolution(conv_features, self.prototype_vectors)
        
        # 🎯 Step 3: Convert distances to activation scores
        # Result: activations[b, p, h, w] = activation score for prototype p at location (h,w)
        prototype_activations = self._prototype_activations(distances)
        
        # 🌍 Step 4: Global max pooling to find best prototype match per image
        # This finds the highest activation for each prototype across all spatial locations
        # Result: [B, P, 1, 1] - best activation for each prototype in each image
        global_max_pooled_activations = F.max_pool2d(prototype_activations, 
                                                   kernel_size=prototype_activations.size()[2:])
        # Reshape to [B, P] for classification
        global_max_pooled_activations = global_max_pooled_activations.view(-1, self.num_prototypes)
        
        # 🎯 Step 5: Classify using the last layer
        # Maps prototype activations to class predictions
        # Result: logits[b, c] = class prediction for image b
        logits = self.last_layer(global_max_pooled_activations)
        
        return logits, prototype_activations
    
    def forward_segmentation(self, x):
        """
        🎯 Forward pass for segmentation (Pixel-wise Classification)
        
        PURPOSE:
        This is the main method for semantic segmentation! It performs pixel-wise
        classification by treating each pixel as a separate classification problem.
        Each pixel is classified based on its prototype activations.
        
        KEY DIFFERENCE FROM forward():
        - forward(): Uses global max pooling (image-level classification)
        - forward_segmentation(): Preserves spatial information (pixel-level classification)
        
        PROCESS:
        1. Extract features from input image
        2. Compute prototype distances for all locations
        3. Convert distances to activation scores
        4. Reshape for pixel-wise classification
        5. Classify each pixel independently
        6. Reshape back to spatial format
        
        Args:
            x: Input images [B, 3, H, W]
        
        Returns:
            logits: Pixel-wise class predictions [B, num_classes, H, W]
            prototype_activations: Prototype activations [B, P, H, W]
        """
        # 🏗️ Step 1: Extract features using backbone + add-on layers
        conv_features = self.conv_features(x)
        
        # 📐 Step 2: Compute L2 distances between features and prototypes
        # Result: distances[b, p, h, w] = distance between patch at (h,w) and prototype p
        distances = self._l2_convolution(conv_features, self.prototype_vectors)
        
        # 🎯 Step 3: Convert distances to activation scores
        # Result: activations[b, p, h, w] = activation score for prototype p at location (h,w)
        prototype_activations = self._prototype_activations(distances)
        
        # 🔄 Step 4: Reshape for pixel-wise classification
        # We need to classify each pixel independently, so we reshape to treat
        # each spatial location as a separate sample
        batch_size, num_prototypes, h, w = prototype_activations.shape
        
        # Flatten spatial dimensions: [B, P, H, W] → [B, P, H*W]
        prototype_activations_flat = prototype_activations.view(batch_size, num_prototypes, -1)
        
        # Permute for classification: [B, P, H*W] → [B, H*W, P]
        # Now each pixel (H*W total) has its own prototype activation vector (P)
        prototype_activations_flat = prototype_activations_flat.permute(0, 2, 1)
        
        # 🎯 Step 5: Classify each pixel independently
        # Apply last layer to each pixel: [B, H*W, P] → [B, H*W, C]
        # Each pixel gets its own class prediction
        logits_flat = self.last_layer(prototype_activations_flat)
        
        # 🔄 Step 6: Reshape back to spatial format
        # Convert back to spatial format: [B, H*W, C] → [B, C, H, W]
        logits = logits_flat.permute(0, 2, 1).view(batch_size, self.num_classes, h, w)
        
        # 🔄 Step 7: Upsample logits to match input image size
        # The feature map is downsampled compared to the input image
        # We need to upsample the logits to match the target segmentation mask size
        target_size = x.shape[2:]  # Get input image height and width
        logits = F.interpolate(logits, size=target_size, mode='bilinear', align_corners=False)
        
        # 📊 Result: logits[b, c, h, w] = class prediction for pixel (h,w) in image b
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
