"""
LLM Module for Patient-Friendly Explanations
============================================
Generates patient-friendly explanations of AI predictions using MegaLLM API.

Medical AI systems must communicate results clearly and safely:
- Use non-diagnostic language
- Avoid definitive medical statements
- Encourage professional consultation
- Provide clear disclaimers

Author: Senior ML Engineer
Date: 2024
"""

import os
import requests
from typing import Optional, Dict
import json


class LLMExplanationError(Exception):
    """Custom exception for LLM explanation errors."""
    pass


def get_api_key() -> str:
    """
    Get API key from environment variable.
    
    Returns:
        API key string
        
    Raises:
        LLMExplanationError: If API key not found
    """
    api_key = os.environ.get("MEGALLM_API_KEY")
    if not api_key:
        raise LLMExplanationError(
            "MEGALLM_API_KEY environment variable not set. "
            "Please set it before running inference with LLM explanations."
        )
    return api_key


def generate_fallback_explanation(
    prediction: str,
    confidence: float,
    gradcam_focus_description: Optional[str] = None
) -> str:
    """
    Generate fallback explanation using rule-based template.
    
    Used when LLM API is unavailable or fails.
    
    Args:
        prediction: Predicted class name
        confidence: Confidence score (0-1)
        gradcam_focus_description: Optional description of Grad-CAM focus areas
        
    Returns:
        Patient-friendly explanation string
    """
    prediction_lower = prediction.lower()
    
    # Base explanation
    explanation = f"The AI system analyzed the skin image and identified features "
    explanation += f"that may be consistent with {prediction_lower}. "
    explanation += f"The confidence level is {confidence*100:.1f}%.\n\n"
    
    # Add Grad-CAM context if available
    if gradcam_focus_description:
        explanation += f"The analysis focused on areas showing: {gradcam_focus_description}\n\n"
    
    # Add class-specific information
    if prediction_lower == "psoriasis":
        explanation += "The visual patterns detected include features often seen in psoriasis, "
        explanation += "such as scaly patches or plaques. "
    elif prediction_lower == "eczema":
        explanation += "The visual patterns detected include features often seen in eczema, "
        explanation += "such as redness, dryness, or inflammation. "
    
    # Add disclaimer and recommendation
    explanation += "\n\n"
    explanation += "⚠️ IMPORTANT: This is an AI-assisted screening result and not a medical diagnosis. "
    explanation += "Please consult with a qualified dermatologist for a proper evaluation and diagnosis."
    
    return explanation


def generate_patient_explanation(
    prediction: str,
    confidence: float,
    gradcam_focus_description: Optional[str] = None,
    use_llm: bool = True
) -> str:
    """
    Generate patient-friendly explanation using LLM API or fallback.
    
    Medical Safety Guidelines:
    - Uses non-diagnostic language
    - Avoids definitive statements like "you have"
    - Uses phrases like "may be consistent with" or "features often seen in"
    - Includes clear disclaimers
    - Encourages professional consultation
    
    Args:
        prediction: Predicted class name (e.g., "Psoriasis", "Eczema")
        confidence: Confidence score (0-1)
        gradcam_focus_description: Optional description of Grad-CAM focus areas
        use_llm: Whether to use LLM API (default: True)
        
    Returns:
        Patient-friendly explanation string
    """
    # If confidence is low, return fixed message
    if confidence < 0.70:
        return (
            "The AI system is not fully confident in this assessment. "
            "A dermatologist review is recommended. "
            "This is an AI-assisted screening tool and not a medical diagnosis."
        )
    
    # Try LLM API if enabled
    if use_llm:
        try:
            return _generate_llm_explanation(
                prediction=prediction,
                confidence=confidence,
                gradcam_focus_description=gradcam_focus_description
            )
        except Exception as e:
            # Fall back to rule-based explanation on error
            print(f"Warning: LLM explanation failed ({e}). Using fallback explanation.")
            return generate_fallback_explanation(
                prediction=prediction,
                confidence=confidence,
                gradcam_focus_description=gradcam_focus_description
            )
    else:
        # Use fallback directly
        return generate_fallback_explanation(
            prediction=prediction,
            confidence=confidence,
            gradcam_focus_description=gradcam_focus_description
        )


def _generate_llm_explanation(
    prediction: str,
    confidence: float,
    gradcam_focus_description: Optional[str] = None
) -> str:
    """
    Internal function to call LLM API.
    
    Args:
        prediction: Predicted class name
        confidence: Confidence score (0-1)
        gradcam_focus_description: Optional description of Grad-CAM focus areas
        
    Returns:
        LLM-generated explanation
        
    Raises:
        LLMExplanationError: If API call fails
    """
    # Get API key
    api_key = get_api_key()
    
    # Build prompt
    prompt = _build_prompt(
        prediction=prediction,
        confidence=confidence,
        gradcam_focus_description=gradcam_focus_description
    )
    
    # API endpoint
    url = "https://ai.megallm.io/v1/chat/completions"
    
    # Headers
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Request body
    payload = {
        "model": "openai-gpt-oss-20b",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a medical AI assistant explaining dermatology screening results "
                    "in simple, non-technical language. Use phrases like 'may be consistent with' "
                    "or 'features often seen in'. Avoid definitive statements. "
                    "Always include a disclaimer that this is not a medical diagnosis."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.3  # Lower temperature for more consistent, factual responses
    }
    
    try:
        # Make API request
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=30  # 30 second timeout
        )
        
        # Check response status
        response.raise_for_status()
        
        # Parse response
        result = response.json()
        
        # Extract explanation from response
        if "choices" in result and len(result["choices"]) > 0:
            explanation = result["choices"][0]["message"]["content"]
            return explanation.strip()
        else:
            raise LLMExplanationError("Invalid response format from LLM API")
            
    except requests.exceptions.RequestException as e:
        raise LLMExplanationError(f"API request failed: {str(e)}")
    except (KeyError, json.JSONDecodeError) as e:
        raise LLMExplanationError(f"Failed to parse API response: {str(e)}")


def _build_prompt(
    prediction: str,
    confidence: float,
    gradcam_focus_description: Optional[str] = None
) -> str:
    """
    Build prompt for LLM API.
    
    Args:
        prediction: Predicted class name
        confidence: Confidence score (0-1)
        gradcam_focus_description: Optional description of Grad-CAM focus areas
        
    Returns:
        Formatted prompt string
    """
    prompt = "The AI model analyzed a skin image and detected visual patterns.\n\n"
    
    # Add Grad-CAM description if available
    if gradcam_focus_description:
        prompt += f"Focus areas identified: {gradcam_focus_description}\n\n"
    
    prompt += f"Prediction: {prediction}\n"
    prompt += f"Confidence: {confidence*100:.1f}%\n\n"
    
    prompt += (
        "Explain this result in simple, patient-friendly language.\n"
        "Requirements:\n"
        "- Use non-diagnostic language\n"
        "- Avoid definitive statements like 'you have'\n"
        "- Use phrases like 'may be consistent with' or 'features often seen in'\n"
        "- Include a clear disclaimer that this is an AI-assisted screening result and not a medical diagnosis\n"
        "- Encourage consultation with a dermatologist\n"
        "- Keep the explanation concise (2-3 sentences)\n"
    )
    
    return prompt


def test_llm_connection() -> bool:
    """
    Test LLM API connection.
    
    Returns:
        True if connection successful, False otherwise
    """
    try:
        api_key = get_api_key()
        print(f"✓ API key found (length: {len(api_key)})")
        
        # Try a simple test call
        url = "https://ai.megallm.io/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "openai-gpt-oss-20b",
            "messages": [
                {
                    "role": "user",
                    "content": "Say 'test' if you can read this."
                }
            ],
            "temperature": 0.3
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        
        print("✓ LLM API connection successful")
        return True
        
    except Exception as e:
        print(f"✗ LLM API connection failed: {e}")
        return False


if __name__ == "__main__":
    # Test the module
    print("Testing LLM Module...")
    print("=" * 80)
    
    # Test API key retrieval
    try:
        api_key = get_api_key()
        print(f"✓ API key found")
    except LLMExplanationError as e:
        print(f"✗ {e}")
        print("\nTo set the API key:")
        print("  Windows: set MEGALLM_API_KEY=your_key_here")
        print("  Linux/Mac: export MEGALLM_API_KEY=your_key_here")
        exit(1)
    
    # Test connection
    if test_llm_connection():
        print("\nTesting explanation generation...")
        
        # Test with high confidence
        explanation = generate_patient_explanation(
            prediction="Psoriasis",
            confidence=0.85,
            gradcam_focus_description="scaly patches and reddened areas"
        )
        print("\nGenerated Explanation:")
        print("-" * 80)
        print(explanation)
        print("-" * 80)
        
        # Test with low confidence
        explanation_low = generate_patient_explanation(
            prediction="Eczema",
            confidence=0.65
        )
        print("\nLow Confidence Explanation:")
        print("-" * 80)
        print(explanation_low)
        print("-" * 80)

