"""
FastAPI Backend for Medical AI Skin Disease Classification
==========================================================
Production-ready API for Psoriasis vs Eczema classification.

Endpoints:
- POST /predict: Upload image and get prediction with explainability

Author: Senior ML Engineer
Date: 2024
"""

import os
import io
import base64
from pathlib import Path
from typing import Optional
import tempfile

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Import existing modules
try:
    from inference import SafeInferencePipeline
    from gradcam_utils import generate_gradcam
    from utils.llm_module import generate_patient_explanation, LLMExplanationError
except ImportError as e:
    print(f"Warning: Import error - {e}")
    print("Some features may not be available")


# Response model
class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    status: str
    explanation: str
    gradcam_image: Optional[str] = None
    probabilities: dict


# Initialize FastAPI app
app = FastAPI(
    title="AI Skin Inflammation Screening API",
    description="Medical AI API for Psoriasis vs Eczema classification",
    version="1.0.0"
)

# CORS middleware (allow all origins for demo - restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model pipeline (loaded on startup)
pipeline: Optional[SafeInferencePipeline] = None


@app.on_event("startup")
async def load_model():
    """
    Load model on application startup.
    This ensures the model is loaded once and reused for all requests.
    """
    global pipeline
    
    print("=" * 80)
    print("Loading Medical AI Model...")
    print("=" * 80)
    
    try:
        model_path = 'checkpoints/best_model.pth'
        
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Model checkpoint not found at {model_path}. "
                "Please ensure best_model.pth exists in checkpoints/ directory."
            )
        
        # Detect device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        if device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        
        # Initialize pipeline
        pipeline = SafeInferencePipeline(
            model_path=model_path,
            confidence_threshold=0.70,
            device=device
        )
        
        # Set model to eval mode
        pipeline.model.eval()
        
        print("✅ Model loaded successfully!")
        print("=" * 80)
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "AI Skin Inflammation Screening API",
        "version": "1.0.0",
        "model_loaded": pipeline is not None
    }


@app.get("/health")
async def health_check():
    """Detailed health check."""
    health_status = {
        "status": "healthy",
        "model_loaded": pipeline is not None,
        "device": str(pipeline.device) if pipeline else None,
        "gpu_available": torch.cuda.is_available() if torch.cuda.is_available() else False
    }
    
    if pipeline:
        health_status["model_ready"] = True
    else:
        health_status["model_ready"] = False
        health_status["error"] = "Model not loaded"
    
    return health_status


def image_to_base64(image_path: str) -> str:
    """
    Convert image file to base64 string.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Base64 encoded string (with data URI prefix)
    """
    try:
        with open(image_path, 'rb') as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            # Determine image format
            img = Image.open(image_path)
            format_map = {
                'PNG': 'png',
                'JPEG': 'jpeg',
                'JPG': 'jpeg'
            }
            img_format = format_map.get(img.format, 'jpeg')
            return f"data:image/{img_format};base64,{encoded_string}"
    except Exception as e:
        print(f"Error converting image to base64: {e}")
        return None


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(..., description="Skin image file (JPG, JPEG, PNG)")
):
    """
    Predict skin disease from uploaded image.
    
    This endpoint:
    1. Accepts image upload
    2. Runs inference using EfficientNet-B4
    3. Generates Grad-CAM visualization
    4. Calls LLM API for patient-friendly explanation
    5. Returns prediction with explainability
    
    Medical Safety:
    - Uses uncertainty threshold (70%)
    - Flags uncertain predictions
    - Provides non-diagnostic language
    - Includes medical disclaimers
    
    Args:
        file: Uploaded image file
        
    Returns:
        JSON response with prediction, confidence, explanation, and Grad-CAM
    """
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check server logs."
        )
    
    # Validate file type
    allowed_types = ['image/jpeg', 'image/jpg', 'image/png']
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed types: {allowed_types}"
        )
    
    # Save uploaded file temporarily
    temp_file = None
    try:
        # Create temporary file
        suffix = Path(file.filename).suffix or '.jpg'
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            # Read uploaded file content
            content = await file.read()
            tmp_file.write(content)
            temp_path = tmp_file.name
        
        # Validate image can be opened
        try:
            img = Image.open(temp_path)
            img.verify()  # Verify it's a valid image
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid image file: {str(e)}"
            )
        
        # Run inference
        try:
            results = pipeline.predict(
                image_path=temp_path,
                generate_gradcam=True,
                use_llm=True
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Inference failed: {str(e)}"
            )
        
        # Convert Grad-CAM image to base64
        gradcam_base64 = None
        if results.get('gradcam_path') and Path(results['gradcam_path']).exists():
            try:
                gradcam_base64 = image_to_base64(results['gradcam_path'])
            except Exception as e:
                print(f"Warning: Failed to convert Grad-CAM to base64: {e}")
                gradcam_base64 = None
        
        # Ensure explanation exists (fallback if LLM failed)
        explanation = results.get('explanation')
        if not explanation:
            # Fallback explanation
            if results['is_confident']:
                explanation = (
                    f"The AI analysis suggests features that may be consistent with "
                    f"{results['predicted_class']}. This is an AI-assisted screening "
                    f"result and not a medical diagnosis. Please consult a qualified "
                    f"dermatologist for professional evaluation."
                )
            else:
                explanation = (
                    "The AI system is not fully confident in this assessment. "
                    "A dermatologist review is recommended. This is an AI-assisted "
                    "screening tool and not a medical diagnosis."
                )
        
        # Build response
        response = PredictionResponse(
            prediction=results['predicted_class'].capitalize(),
            confidence=round(results['confidence'], 4),
            status=results['status'],
            explanation=explanation,
            gradcam_image=gradcam_base64,
            probabilities={
                k: round(v, 4) for k, v in results['probabilities'].items()
            }
        )
        
        return response
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Handle unexpected errors
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error: {str(e)}"
        )
    finally:
        # Clean up temporary file
        if temp_path and Path(temp_path).exists():
            try:
                os.unlink(temp_path)
            except:
                pass


@app.post("/predict/batch")
async def predict_batch(files: list[UploadFile] = File(...)):
    """
    Batch prediction endpoint (for multiple images).
    
    Note: This is a placeholder for future implementation.
    For now, use /predict endpoint multiple times.
    """
    return {
        "message": "Batch prediction not yet implemented",
        "suggestion": "Use /predict endpoint for individual images"
    }


import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=port)

