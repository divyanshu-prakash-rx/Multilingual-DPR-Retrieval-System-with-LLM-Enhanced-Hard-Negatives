"""
LLM Categorization Module for Hard Negative Classification

This module provides both low-level LLM classification and high-level training integration.

Low-level API:
    - HardNegativeClassifier: Direct LLM classification

High-level API:
    - LLMTrainingEnhancer: Complete training data enhancement
    - enhance_training_data(): Quick enhancement function
    - create_llm_enhancer(): Factory function
"""

from .hard_negative_classifier import HardNegativeClassifier
from .llm_integration import (
    LLMTrainingEnhancer,
    create_llm_enhancer,
    enhance_training_data,
    get_llm_model_config
)

__all__ = [
    'HardNegativeClassifier',
    'LLMTrainingEnhancer', 
    'create_llm_enhancer',
    'enhance_training_data',
    'get_llm_model_config'
]