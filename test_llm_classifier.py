"""
Test script for LLM Hard Negative Classifier
"""

import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment and setup path
env_path = Path('train_scripts_modified/llm_categorization/.env')
if env_path.exists():
    load_dotenv(env_path)

sys.path.append('train_scripts_modified/llm_categorization')

def test_llm_classifier():
    """Test LLM classifier with one sample and return result"""
    try:
        from hard_negative_classifier import HardNegativeClassifier
        
        # Initialize classifier
        classifier = HardNegativeClassifier()
        
        # Health check
        health = classifier.health_check()
        if not health.get('ok', False):
            return {'success': False, 'error': 'Health check failed', 'details': health}
        
        # Test with one sample
        query = "What is the capital of France?"
        negative = "Paris is a beautiful city with many museums"
        
        print(f"Query: {query}")
        print(f"Negative: {negative}")
        
        # Classify
        result = classifier.classify_single(query, negative)
        
        # Return structured result
        return {
            'success': True,
            'query': query,
            'negative': negative,
            'classification': result.get('is_hard_negative', False),
            'confidence': result.get('confidence', 0.0),
            'raw_response': result.get('raw_response', ''),
            'full_result': result
        }
        
    except Exception as e:
        return {'success': False, 'error': str(e)}

def main():
    """Main entry point"""
    result = test_llm_classifier()
    
    if result['success']:
        classification_type = "HARD NEGATIVE" if result['classification'] else "REGULAR NEGATIVE"
        print(f"\nResult: {classification_type}")
        print(f"Confidence: {result['confidence']}")
        return result
    else:
        print(f"Test failed: {result['error']}")
        return result

if __name__ == "__main__":
    main()