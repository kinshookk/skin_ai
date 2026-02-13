"""
Medical AI Inference Pipeline with Explainability
==================================================
Safe inference system for skin disease classification with:
- Uncertainty quantification
- Grad-CAM visualization
- Clinical output formatting

Medical Safety Features:
- Confidence threshold (70%) for uncertain predictions
- Explicit recommendation for dermatologist review when uncertain
- Visual explainability via Grad-CAM for clinical validation
- Softmax probabilities for transparency

Author: Senior ML Engineer
Date: 2024
"""

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import argparse

# Import model architecture
try:
    from model_utils import EfficientNetClassifier
except ImportError:
    # Fallback: import from train.py
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from train import EfficientNetClassifier

# Import Grad-CAM utilities
try:
    from gradcam_utils import generate_gradcam
    GRAD_CAM_AVAILABLE = True
except ImportError:
    GRAD_CAM_AVAILABLE = False
    print("Warning: Grad-CAM utilities not available")

# Import LLM module
try:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from utils.llm_module import generate_patient_explanation, LLMExplanationError
    LLM_AVAILABLE = True
except ImportError as e:
    LLM_AVAILABLE = False
    print(f"Warning: LLM module not available: {e}")


class SafeInferencePipeline:
    """
    Safe inference pipeline with uncertainty quantification.
    
    Medical AI systems must be transparent and indicate when
    predictions are uncertain. This pipeline:
    - Provides confidence scores
    - Flags uncertain predictions (< 70% confidence)
    - Recommends expert review for uncertain cases
    - Generates explainability visualizations
    """
    
    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.70,
        image_size: int = 224,
        device: Optional[torch.device] = None,
        class_names: Optional[list] = None
    ):
        """
        Initialize inference pipeline.
        
        Args:
            model_path: Path to trained model checkpoint
            confidence_threshold: Minimum confidence for "confident" prediction
            image_size: Input image size
            device: Device to use (auto-detect if None)
            class_names: List of class names (default: ['eczema', 'psoriasis'])
        """
        self.confidence_threshold = confidence_threshold
        self.image_size = image_size
        self.class_names = class_names or ['eczema', 'psoriasis']
        
        # Device setup
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        print(f"Using device: {self.device}")
        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        
        # Load model
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # Preprocessing transform
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet mean
                std=[0.229, 0.224, 0.225]   # ImageNet std
            )
        ])
    
    def _load_model(self, model_path: str) -> torch.nn.Module:
        """Load trained model from checkpoint."""
        print(f"\nLoading model from: {model_path}")
        
        # Initialize model architecture
        model = EfficientNetClassifier(num_classes=2, dropout=0.5, pretrained=False)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        else:
            # Assume checkpoint is just state dict
            model.load_state_dict(checkpoint)
        
        model = model.to(self.device)
        print("Model loaded successfully!")
        
        return model
    
    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """
        Preprocess image for inference.
        
        Args:
            image_path: Path to input image
            
        Returns:
            Preprocessed tensor
        """
        image = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(image).unsqueeze(0)
        return input_tensor.to(self.device)
    
    def _describe_gradcam_focus(self, gradcam_path: Optional[str]) -> Optional[str]:
        """
        Generate a simple description of Grad-CAM focus areas.
        
        In a production system, this could analyze the Grad-CAM heatmap
        to describe which regions were most important. For now, we provide
        a generic description.
        
        Args:
            gradcam_path: Path to Grad-CAM visualization (if available)
            
        Returns:
            Description string or None
        """
        if gradcam_path:
            # In a full implementation, this could analyze the heatmap
            # For now, provide a generic description
            return "visual patterns in the skin lesion area, including texture, color, and boundary features"
        return None
    
    def predict(
        self,
        image_path: str,
        generate_gradcam: bool = True,
        use_llm: bool = True
    ) -> Dict:
        """
        Run inference on a single image.
        
        Args:
            image_path: Path to input image
            generate_gradcam: Whether to generate Grad-CAM visualization
            use_llm: Whether to use LLM for explanation (default: True)
            
        Returns:
            Dictionary with prediction results
        """
        # Preprocess
        input_tensor = self.preprocess_image(image_path)
        
        # Forward pass
        with torch.no_grad():
            logits = self.model(input_tensor)
            probs = F.softmax(logits, dim=1)
        
        # Get predictions
        probs_np = probs[0].cpu().numpy()
        predicted_class_idx = int(torch.argmax(probs, dim=1).item())
        confidence = float(probs_np[predicted_class_idx])
        predicted_class = self.class_names[predicted_class_idx]
        
        # Determine status
        is_confident = confidence >= self.confidence_threshold
        status = "Confident" if is_confident else "Uncertain — Dermatologist review recommended"
        
        # Generate Grad-CAM if requested
        gradcam_path = None
        if generate_gradcam and GRAD_CAM_AVAILABLE:
            try:
                output_dir = Path('outputs')
                output_dir.mkdir(exist_ok=True)
                image_name = Path(image_path).stem
                gradcam_path = output_dir / f'gradcam_{image_name}.png'
                
                generate_gradcam(
                    model=self.model,
                    image_path=image_path,
                    target_class=predicted_class_idx,
                    output_path=str(gradcam_path),
                    device=self.device
                )
                gradcam_path = str(gradcam_path)
            except Exception as e:
                print(f"Warning: Grad-CAM generation failed: {e}")
                gradcam_path = None
        
        # Generate patient-friendly explanation using LLM
        explanation = None
        if LLM_AVAILABLE:
            try:
                gradcam_description = self._describe_gradcam_focus(gradcam_path)
                explanation = generate_patient_explanation(
                    prediction=predicted_class,
                    confidence=confidence,
                    gradcam_focus_description=gradcam_description,
                    use_llm=use_llm
                )
            except Exception as e:
                print(f"Warning: Failed to generate LLM explanation: {e}")
                # Will use fallback explanation if LLM fails
        
        # Build results dictionary
        results = {
            'predicted_class': predicted_class,
            'predicted_class_idx': predicted_class_idx,
            'confidence': confidence,
            'confidence_percent': confidence * 100,
            'status': status,
            'is_confident': is_confident,
            'probabilities': {
                self.class_names[i]: float(probs_np[i])
                for i in range(len(self.class_names))
            },
            'gradcam_path': gradcam_path,
            'explanation': explanation
        }
        
        return results
    
    def print_clinical_output(self, results: Dict):
        """
        Print formatted clinical output.
        
        Format:
        Prediction: <class>
        Confidence: <percentage>%
        Status: <Confident/Uncertain>
        Explanation: <patient-friendly explanation>
        Grad-CAM saved at: <path> (if available)
        """
        print("\n" + "=" * 80)
        print("CLINICAL PREDICTION RESULTS")
        print("=" * 80)
        print(f"\nPrediction: {results['predicted_class'].capitalize()}")
        print(f"Confidence: {results['confidence_percent']:.1f}%")
        print(f"Status: {results['status']}")
        
        # Print explanation if available
        if results.get('explanation'):
            print(f"\nExplanation:")
            print("-" * 80)
            print(results['explanation'])
            print("-" * 80)
        
        # Print probability distribution
        print(f"\nProbability Distribution:")
        for class_name, prob in results['probabilities'].items():
            print(f"  {class_name.capitalize()}: {prob*100:.1f}%")
        
        # Grad-CAM info
        if results['gradcam_path']:
            print(f"\nGrad-CAM saved at: {results['gradcam_path']}")
        elif GRAD_CAM_AVAILABLE:
            print("\nGrad-CAM: Not generated (use --gradcam flag)")
        
        print("=" * 80)
        
        # Medical recommendation
        if not results['is_confident']:
            print("\n⚠ CLINICAL RECOMMENDATION:")
            print("   Low confidence prediction detected.")
            print("   Recommend dermatologist review for final diagnosis.")
            print("=" * 80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Medical AI Inference Pipeline with Explainability'
    )
    parser.add_argument(
        '--image',
        type=str,
        required=True,
        help='Path to input image'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='checkpoints/best_model.pth',
        help='Path to model checkpoint (default: checkpoints/best_model.pth)'
    )
    parser.add_argument(
        '--confidence_threshold',
        type=float,
        default=0.70,
        help='Confidence threshold for uncertain predictions (default: 0.70)'
    )
    parser.add_argument(
        '--no_gradcam',
        action='store_true',
        help='Skip Grad-CAM visualization'
    )
    parser.add_argument(
        '--force_cuda',
        action='store_true',
        help='Force CUDA usage'
    )
    parser.add_argument(
        '--force_cpu',
        action='store_true',
        help='Force CPU usage'
    )
    parser.add_argument(
        '--no_llm',
        action='store_true',
        help='Skip LLM explanation generation (use fallback)'
    )
    
    args = parser.parse_args()
    
    # Device selection
    if args.force_cpu:
        device = torch.device('cpu')
    elif args.force_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available but --force_cuda specified")
        device = torch.device('cuda')
    else:
        device = None  # Auto-detect
    
    # Check image exists
    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {args.image}")
    
    # Check model exists
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {args.model}")
    
    # Initialize pipeline
    pipeline = SafeInferencePipeline(
        model_path=str(model_path),
        confidence_threshold=args.confidence_threshold,
        device=device
    )
    
    # Run inference
    print(f"\nProcessing image: {args.image}")
    results = pipeline.predict(
        image_path=str(image_path),
        generate_gradcam=not args.no_gradcam,
        use_llm=not args.no_llm
    )
    
    # Print clinical output
    pipeline.print_clinical_output(results)
    
    return results


if __name__ == '__main__':
    main()

