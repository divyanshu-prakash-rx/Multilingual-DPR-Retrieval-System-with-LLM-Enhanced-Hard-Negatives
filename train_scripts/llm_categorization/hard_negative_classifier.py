"""
Hard Negative Classifier using LLM (Gemini 2.0 Flash)

This module provides LLM-based classification of hard negatives for retrieval training.
It determines whether a negative passage is "hard" (semantically similar but incorrect) 
or "regular" (clearly different from the query).
"""

import os
import time
from pathlib import Path
import google.generativeai as genai
from typing import Dict, Any, Optional

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Try to load from current directory first, then from script directory
    env_paths = [
        Path('.env'),
        Path(__file__).parent / '.env'
    ]
    
    for env_path in env_paths:
        if env_path.exists():
            load_dotenv(env_path)
            print(f"🔧 Loaded API key from {env_path}")
            break
    else:
        print("⚠️ No .env file found, expecting GEMINI_API_KEY in environment")
except ImportError:
    print("⚠️ python-dotenv not installed, expecting GEMINI_API_KEY in environment")

class HardNegativeClassifier:
    """LLM-based hard negative classifier using Gemini"""
    
    def __init__(self, model_name: str = "gemini-2.0-flash-exp"):
        """Initialize the classifier"""
        self.model_name = model_name
        self.model = None
        self._setup_model()
        
    def _setup_model(self):
        """Setup the Gemini model"""
        try:
            # Check for API key
            api_key = os.getenv('GEMINI_API_KEY')
            if not api_key:
                raise ValueError("GEMINI_API_KEY not found in environment variables")
            
            # Configure Gemini API
            genai.configure(api_key=api_key)
            
            # Initialize model
            self.model = genai.GenerativeModel(self.model_name)
            
            print(f"Initialized Gemini classifier (model={self.model_name}, api=v1)")
            
        except Exception as e:
            print(f"Failed to initialize Gemini classifier: {e}")
            raise
    
    def health_check(self) -> Dict[str, Any]:
        """Check if the classifier is working"""
        try:
            if self.model is None:
                return {'ok': False, 'error': 'Model not initialized'}
            
            # Test with a simple query
            test_response = self.model.generate_content("Test: respond with 'OK'")
            
            if test_response and test_response.text:
                return {
                    'ok': True,
                    'model': self.model_name,
                    'api_version': 'v1'
                }
            else:
                return {'ok': False, 'error': 'No response from model'}
                
        except Exception as e:
            return {'ok': False, 'error': str(e)}
    
    def classify_single(self, query: str, negative_passage: str) -> Dict[str, Any]:
        """Classify a single query-negative pair"""
        
        if self.model is None:
            return {'is_hard_negative': False, 'confidence': 0.0, 'error': 'Model not initialized'}
        
        try:
            # Create classification prompt
            prompt = f"""
Analyze this query-passage pair for retrieval training:

QUERY: "{query}"
NEGATIVE PASSAGE: "{negative_passage}"

Determine if this negative passage is a "hard negative" or "regular negative":

- HARD NEGATIVE: Semantically similar to the query topic but factually incorrect, misleading, or doesn't answer the query properly. These are challenging cases that help improve model discrimination.

- REGULAR NEGATIVE: Clearly different topic, obviously irrelevant, or easy to distinguish from the query.

Respond with ONLY:
CLASSIFICATION: [HARD or REGULAR]
CONFIDENCE: [0.0-1.0]
REASON: [Brief explanation]
"""

            # Get response from model
            response = self.model.generate_content(prompt)
            
            if not response or not response.text:
                return {'is_hard_negative': False, 'confidence': 0.5, 'error': 'Empty response'}
            
            # Parse response
            response_text = response.text.upper()
            
            is_hard = 'HARD' in response_text and 'CLASSIFICATION:' in response_text
            
            # Extract confidence if available
            confidence = 0.7  # Default confidence
            if 'CONFIDENCE:' in response_text:
                try:
                    conf_line = [line for line in response_text.split('\n') if 'CONFIDENCE:' in line][0]
                    confidence = float(conf_line.split(':')[1].strip())
                except:
                    confidence = 0.7
            
            return {
                'is_hard_negative': is_hard,
                'confidence': confidence,
                'raw_response': response.text,
                'query': query,
                'negative': negative_passage
            }
            
        except Exception as e:
            return {
                'is_hard_negative': False,
                'confidence': 0.0,
                'error': str(e),
                'query': query,
                'negative': negative_passage
            }
    
    def classify_batch(self, query_negative_pairs: list) -> list:
        """Classify multiple query-negative pairs"""
        results = []
        
        for i, (query, negative) in enumerate(query_negative_pairs):
            result = self.classify_single(query, negative)
            results.append(result)
            
            # Rate limiting
            if i < len(query_negative_pairs) - 1:
                time.sleep(0.5)
        
        return results