"""
Model Utilities
===============
Shared model architecture and utilities for training and inference.
"""

import torch
import torch.nn as nn
import timm


class EfficientNetClassifier(nn.Module):
    """EfficientNet-B4 with custom classifier head for binary classification."""
    
    def __init__(self, num_classes: int = 2, dropout: float = 0.5, pretrained: bool = True):
        """
        Initialize model.
        
        Args:
            num_classes: Number of output classes (default: 2 for binary)
            dropout: Dropout probability (default: 0.5)
            pretrained: Use pretrained weights (default: True)
        """
        super(EfficientNetClassifier, self).__init__()
        
        # Load EfficientNet-B4 backbone
        self.backbone = timm.create_model(
            'efficientnet_b4',
            pretrained=pretrained,
            num_classes=0,  # Remove default classifier
            global_pool='avg'
        )
        
        # Get feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            features = self.backbone(dummy_input)
            feature_dim = features.shape[1]
        
        # Custom classifier head
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        """Forward pass."""
        features = self.backbone(x)
        output = self.classifier(features)
        return output

