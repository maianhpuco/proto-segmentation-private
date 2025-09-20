"""
=============================================================================
SIMPLIFIED DEEPLAB FEATURE EXTRACTOR
=============================================================================

USAGE INSTRUCTIONS:
1. This file provides a simplified DeepLab feature extractor
2. Based on ResNet101 backbone with ASPP (Atrous Spatial Pyramid Pooling)
3. Used as the backbone for prototype-based segmentation
4. Outputs features ready for prototype matching

HOW TO USE:
- Import: from deeplab_features import deeplabv2_resnet101_features
- Create: features = deeplabv2_resnet101_features(pretrained=True)
- Use: output = features(input_images)

CONFIGURATION:
- pretrained: Whether to use ImageNet pretrained weights
- out_channels: Number of output feature channels (default: 64)
=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ASPP(nn.Module):
    """Simplified Atrous Spatial Pyramid Pooling module"""
    
    def __init__(self, in_channels, out_channels=64):
        super(ASPP, self).__init__()
        
        # 1x1 convolution
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1x1 = nn.BatchNorm2d(out_channels)
        
        # 3x3 convolutions with different dilation rates
        self.conv3x3_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                                   padding=6, dilation=6, bias=False)
        self.bn3x3_1 = nn.BatchNorm2d(out_channels)
        
        self.conv3x3_2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                                   padding=12, dilation=12, bias=False)
        self.bn3x3_2 = nn.BatchNorm2d(out_channels)
        
        self.conv3x3_3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                                   padding=18, dilation=18, bias=False)
        self.bn3x3_3 = nn.BatchNorm2d(out_channels)
        
        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_global = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn_global = nn.BatchNorm2d(out_channels)
        
        # Final projection
        self.conv_proj = nn.Conv2d(out_channels * 5, out_channels, kernel_size=1, bias=False)
        self.bn_proj = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        size = x.shape[2:]
        
        # 1x1 convolution
        x1 = self.relu(self.bn1x1(self.conv1x1(x)))
        
        # 3x3 convolutions with different dilations
        x2 = self.relu(self.bn3x3_1(self.conv3x3_1(x)))
        x3 = self.relu(self.bn3x3_2(self.conv3x3_2(x)))
        x4 = self.relu(self.bn3x3_3(self.conv3x3_3(x)))
        
        # Global average pooling
        x5 = self.global_avg_pool(x)
        x5 = self.relu(self.bn_global(self.conv_global(x5)))
        x5 = F.interpolate(x5, size=size, mode='bilinear', align_corners=False)
        
        # Concatenate all features
        x = torch.cat([x1, x2, x3, x4, x5], dim=1)
        
        # Final projection
        x = self.relu(self.bn_proj(self.conv_proj(x)))
        
        return x


class DeepLabV2ResNet101(nn.Module):
    """Simplified DeepLab v2 with ResNet101 backbone"""
    
    def __init__(self, pretrained=True, out_channels=64):
        super(DeepLabV2ResNet101, self).__init__()
        
        # Load pretrained ResNet101
        resnet = models.resnet101(pretrained=pretrained)
        
        # Remove the last two layers (avgpool and fc)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        
        # ASPP module
        self.aspp = ASPP(2048, out_channels)  # ResNet101 outputs 2048 channels
        
        self.out_channels = out_channels
    
    def forward(self, x):
        # Extract features using ResNet101 backbone
        x = self.backbone(x)
        
        # Apply ASPP
        x = self.aspp(x)
        
        return x


def deeplabv2_resnet101_features(pretrained=True, out_channels=64):
    """
    Create DeepLab v2 ResNet101 feature extractor
    
    Args:
        pretrained: Whether to use ImageNet pretrained weights
        out_channels: Number of output feature channels
    
    Returns:
        DeepLabV2ResNet101 model
    """
    model = DeepLabV2ResNet101(pretrained=pretrained, out_channels=out_channels)
    return model


# Example usage
if __name__ == "__main__":
    # Create model
    model = deeplabv2_resnet101_features(pretrained=True, out_channels=64)
    
    # Test with dummy input
    dummy_input = torch.randn(1, 3, 513, 513)
    output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
