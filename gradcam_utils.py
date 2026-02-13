"""
Grad-CAM Visualization Utilities
==================================
Gradient-weighted Class Activation Mapping (Grad-CAM) for medical AI explainability.

Grad-CAM highlights the regions of the input image that are most important
for the model's prediction. In medical imaging, this helps clinicians:
- Understand which anatomical features the model focuses on
- Verify the model is looking at clinically relevant regions
- Identify potential biases or errors in the model's attention
- Build trust in AI-assisted diagnosis

Medical Relevance:
- Red regions indicate high importance for the predicted class
- Helps identify if model focuses on lesion boundaries, texture, or color
- Can reveal if model is using spurious correlations (e.g., background artifacts)
- Essential for regulatory compliance and clinical validation

Author: Senior ML Engineer
Date: 2024
"""

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, Optional
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image
    GRAD_CAM_AVAILABLE = True
except ImportError:
    GRAD_CAM_AVAILABLE = False
    print("Warning: pytorch-grad-cam not installed. Install with: pip install grad-cam")


def get_efficientnet_target_layer(model):
    """
    Get the target layer for Grad-CAM in EfficientNet-B4.
    
    For EfficientNet-B4, we target the final convolutional block before
    global average pooling. This captures high-level features that are
    most relevant for classification decisions.
    
    Args:
        model: EfficientNetClassifier model
        
    Returns:
        Target layer for Grad-CAM
    """
    # EfficientNet-B4 structure: backbone -> blocks -> final conv
    # The final convolutional features are in the last block
    backbone = model.backbone
    
    # For timm EfficientNet models, the final conv layer is typically
    # in the last block before global pooling
    # We need to find the last convolutional layer
    
    # Method 1: Try to access blocks directly
    if hasattr(backbone, 'blocks'):
        # Get the last block
        last_block = backbone.blocks[-1]
        # Find the last conv layer in the block
        for name, module in reversed(list(last_block.named_modules())):
            if isinstance(module, torch.nn.Conv2d):
                return module
    
    # Method 2: Search through all modules
    for name, module in reversed(list(backbone.named_modules())):
        if isinstance(module, torch.nn.Conv2d):
            # Prefer layers closer to the end (higher level features)
            return module
    
    # Fallback: return the backbone itself (will use final features)
    return backbone


def preprocess_image_for_gradcam(
    image_path: str,
    image_size: int = 224
) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Preprocess image for Grad-CAM visualization.
    
    Args:
        image_path: Path to input image
        image_size: Target image size (default: 224)
        
    Returns:
        Tuple of (preprocessed_tensor, original_image_array)
    """
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    
    # Resize for model input
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet mean
            std=[0.229, 0.224, 0.225]   # ImageNet std
        )
    ])
    
    # Preprocessed tensor for model
    input_tensor = transform(image).unsqueeze(0)
    
    # Original image array for visualization (0-1 range)
    img_array = np.array(image.resize((image_size, image_size))) / 255.0
    
    return input_tensor, img_array


def generate_gradcam(
    model: torch.nn.Module,
    image_path: str,
    target_class: Optional[int] = None,
    output_path: Optional[str] = None,
    image_size: int = 224,
    device: Optional[torch.device] = None
) -> Tuple[np.ndarray, str]:
    """
    Generate Grad-CAM visualization for a given image.
    
    This function:
    1. Loads and preprocesses the image
    2. Runs forward pass to get prediction
    3. Computes gradients w.r.t. the target layer
    4. Generates heatmap showing important regions
    5. Overlays heatmap on original image
    6. Saves the visualization
    
    Medical Context:
    - The heatmap shows which image regions influenced the diagnosis
    - Red regions = high importance for predicted class
    - Helps verify model focuses on clinically relevant features
    - Essential for explainable AI in medical applications
    
    Args:
        model: Trained EfficientNetClassifier model
        image_path: Path to input image
        target_class: Target class index (None = use predicted class)
        output_path: Path to save output (auto-generated if None)
        image_size: Input image size (default: 224)
        device: Device to use (auto-detect if None)
        
    Returns:
        Tuple of (overlay_image_array, output_file_path)
    """
    if not GRAD_CAM_AVAILABLE:
        raise ImportError(
            "pytorch-grad-cam is required. Install with: pip install grad-cam"
        )
    
    # Device setup
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.eval()
    model = model.to(device)
    
    # Preprocess image
    input_tensor, img_array = preprocess_image_for_gradcam(image_path, image_size)
    input_tensor = input_tensor.to(device)
    
    # Get prediction
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = F.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][predicted_class].item()
    
    # Use predicted class if target_class not specified
    if target_class is None:
        target_class = predicted_class
    
    # Get target layer for Grad-CAM
    target_layer = get_efficientnet_target_layer(model)
    
    # Create Grad-CAM object
    # Note: use_cuda is deprecated in newer versions, use device instead
    cam = GradCAM(
        model=model,
        target_layers=[target_layer]
    )
    
    # Define target (which class to visualize)
    targets = [ClassifierOutputTarget(target_class)]
    
    # Generate CAM
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]  # Remove batch dimension
    
    # Create overlay
    # show_cam_on_image expects image in 0-1 range
    # Note: use_rgb parameter might not be available in all versions
    try:
        overlay = show_cam_on_image(
            img_array,
            grayscale_cam,
            use_rgb=True
        )
    except TypeError:
        # Fallback for versions without use_rgb parameter
        overlay = show_cam_on_image(img_array, grayscale_cam)
    
    # Generate output path
    if output_path is None:
        image_name = Path(image_path).stem
        output_dir = Path('outputs')
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f'gradcam_{image_name}.png'
    
    # Save visualization
    plt.figure(figsize=(12, 5))
    
    # Original image
    plt.subplot(1, 2, 1)
    plt.imshow(img_array)
    plt.title(f'Original Image\nPredicted: Class {predicted_class} ({confidence:.1%})')
    plt.axis('off')
    
    # Grad-CAM overlay
    plt.subplot(1, 2, 2)
    plt.imshow(overlay / 255.0)  # Normalize to 0-1 for display
    plt.title(f'Grad-CAM Visualization\nTarget Class: {target_class}')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return overlay, str(output_path)


def generate_gradcam_simple(
    model: torch.nn.Module,
    image_path: str,
    output_path: Optional[str] = None,
    device: Optional[torch.device] = None
) -> str:
    """
    Simplified Grad-CAM function for easy use.
    
    Args:
        model: Trained model
        image_path: Path to input image
        output_path: Output path (auto-generated if None)
        device: Device to use
        
    Returns:
        Path to saved Grad-CAM visualization
    """
    _, output_file = generate_gradcam(
        model=model,
        image_path=image_path,
        output_path=output_path,
        device=device
    )
    return output_file

