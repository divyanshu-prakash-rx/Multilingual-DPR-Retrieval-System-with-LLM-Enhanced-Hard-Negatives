"""
Common LLM Integration Module for Negative Sampling Training

This module provides a unified interface for integrating LLM-based hard negative
classification into any negative sampling training method.

Workflow:
1. Load training data from negative sampling methods
2. Apply LLM classification to categorize negatives as hard/regular
3. Return enhanced training data with LLM classifications
4. Support for all negative sampling methods (ANCE, DPR, ICT, TAS)

Usage:
    from llm_integration import LLMTrainingEnhancer
    
    enhancer = LLMTrainingEnhancer()
    enhanced_data = enhancer.enhance_training_data(
        train_data, 
        method='ANCE',
        sample_limit=1000
    )
"""

import os
import sys
import pandas as pd
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import logging

# Import the hard negative classifier
try:
    from .hard_negative_classifier import HardNegativeClassifier
except ImportError:
    # Fallback for direct script execution
    from hard_negative_classifier import HardNegativeClassifier

class LLMTrainingEnhancer:
    """Main class for LLM-enhanced training integration"""
    
    def __init__(self, 
                 enable_llm: bool = True,
                 sample_limit: int = 1000,
                 batch_size: int = 50,
                 verbose: bool = True):
        """
        Initialize the LLM Training Enhancer
        
        Args:
            enable_llm: Whether to enable LLM enhancement
            sample_limit: Maximum number of samples to process with LLM
            batch_size: Batch size for progress reporting
            verbose: Whether to print detailed progress
        """
        self.enable_llm = enable_llm
        self.sample_limit = sample_limit
        self.batch_size = batch_size
        self.verbose = verbose
        
        # Initialize classifier
        self.classifier = None
        self.llm_available = False
        
        if self.enable_llm:
            self._initialize_classifier()
    
    def _initialize_classifier(self):
        """Initialize the LLM classifier"""
        try:
            self.classifier = HardNegativeClassifier()
            self.llm_available = True
            if self.verbose:
                print("✅ LLM classifier initialized successfully")
        except Exception as e:
            if self.verbose:
                print(f"⚠️ LLM classifier initialization failed: {e}")
                print("🔄 LLM enhancement will be disabled")
            self.llm_available = False
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on LLM classifier"""
        if not self.llm_available or not self.classifier:
            return {
                'ok': False, 
                'error': 'LLM classifier not available'
            }
        
        return self.classifier.health_check()
    
    def enhance_training_data(self, 
                            train_data: pd.DataFrame,
                            method: str = 'ANCE',
                            negative_column: str = None) -> pd.DataFrame:
        """
        Enhance training data with LLM classifications
        
        Args:
            train_data: Training data DataFrame
            method: Negative sampling method name (for logging)
            negative_column: Column name containing negative passages
            
        Returns:
            Enhanced DataFrame with LLM classifications
        """
        if not self.enable_llm or not self.llm_available:
            if self.verbose:
                print(f"🔄 Running baseline {method} (no LLM enhancement)")
            return train_data
        
        if self.verbose:
            print(f"🤖 Applying LLM enhancement to {method} training...")
        
        # Detect negative column automatically
        if negative_column is None:
            negative_column = self._detect_negative_column(train_data)
        
        if negative_column is None:
            if self.verbose:
                print("❌ Could not detect negative passage column")
            return train_data
        
        # Perform health check
        health = self.health_check()
        if not health['ok']:
            if self.verbose:
                print(f"❌ LLM health check failed: {health}")
            return train_data
        
        if self.verbose:
            print(f"✅ LLM classifier ready: {health['model']}")
        
        # Process samples with LLM
        return self._process_with_llm(train_data, negative_column, method)
    
    def _detect_negative_column(self, train_data: pd.DataFrame) -> Optional[str]:
        """Automatically detect the column containing negative passages"""
        possible_columns = [
            'hard_negative', 'negative', 'neg_text', 
            'negative_text', 'hard_neg', 'neg_passage'
        ]
        
        for col in possible_columns:
            if col in train_data.columns:
                return col
        
        # If none found, look for columns with 'negative' in the name
        for col in train_data.columns:
            if 'negative' in col.lower() or 'neg' in col.lower():
                return col
        
        return None
    
    def _process_with_llm(self, 
                         train_data: pd.DataFrame,
                         negative_column: str,
                         method: str) -> pd.DataFrame:
        """Process training data with LLM classifier"""
        
        # Process limited samples for efficiency
        sample_data = train_data.head(self.sample_limit).copy()
        
        if self.verbose:
            print(f"📊 Processing {len(sample_data)} samples with LLM for {method}...")
        
        enhanced_data = []
        llm_stats = {'hard': 0, 'regular': 0, 'error': 0}
        
        for idx, row in sample_data.iterrows():
            try:
                # Get query and negative text
                query = row.get('query_text', row.get('query', ''))
                negative = row.get(negative_column, '')
                
                if not query or not negative:
                    # Skip if missing data
                    enhanced_row = row.copy()
                    enhanced_row['llm_classification'] = 'unknown'
                    enhanced_data.append(enhanced_row)
                    llm_stats['error'] += 1
                    continue
                
                # Classify with LLM
                result = self.classifier.classify_single(query, negative)
                
                # Add LLM classification to row
                enhanced_row = row.copy()
                if 'error' not in result:
                    classification = 'hard' if result.get('is_hard_negative', False) else 'regular'
                    enhanced_row['llm_classification'] = classification
                    enhanced_row['llm_confidence'] = result.get('confidence', 0.0)
                    llm_stats[classification] += 1
                else:
                    enhanced_row['llm_classification'] = 'unknown'
                    enhanced_row['llm_confidence'] = 0.0
                    llm_stats['error'] += 1
                
                enhanced_data.append(enhanced_row)
                
                # Progress reporting
                if self.verbose and (idx + 1) % self.batch_size == 0:
                    print(f"   Processed {idx + 1}/{len(sample_data)} samples")
                
            except Exception as e:
                if self.verbose:
                    print(f"   ⚠️ LLM classification failed for sample {idx}: {e}")
                enhanced_row = row.copy()
                enhanced_row['llm_classification'] = 'unknown'
                enhanced_row['llm_confidence'] = 0.0
                enhanced_data.append(enhanced_row)
                llm_stats['error'] += 1
        
        # Create enhanced DataFrame
        enhanced_df = pd.DataFrame(enhanced_data)
        
        # Add remaining unprocessed data with 'unknown' classification
        if len(train_data) > self.sample_limit:
            remaining_data = train_data.iloc[self.sample_limit:].copy()
            remaining_data['llm_classification'] = 'unknown'
            remaining_data['llm_confidence'] = 0.0
            enhanced_df = pd.concat([enhanced_df, remaining_data], ignore_index=True)
        
        # Print statistics
        if self.verbose and llm_stats:
            total_processed = sum(llm_stats.values())
            print(f"✅ LLM Enhancement Complete for {method}:")
            print(f"   Hard negatives: {llm_stats['hard']} ({llm_stats['hard']/total_processed*100:.1f}%)")
            print(f"   Regular negatives: {llm_stats['regular']} ({llm_stats['regular']/total_processed*100:.1f}%)")
            print(f"   Errors: {llm_stats['error']}")
            print(f"   Total enhanced: {total_processed}/{len(train_data)}")
        
        return enhanced_df
    
    def get_enhanced_model_path(self, 
                               base_path: str, 
                               method: str,
                               add_llm_suffix: bool = None) -> str:
        """
        Generate enhanced model output path with LLM suffix
        
        Args:
            base_path: Base model output path
            method: Training method name
            add_llm_suffix: Whether to add LLM suffix (auto-detect if None)
            
        Returns:
            Enhanced model path
        """
        if add_llm_suffix is None:
            add_llm_suffix = self.enable_llm and self.llm_available
        
        if add_llm_suffix:
            return f"{base_path}-LLM"
        else:
            return base_path
    
    def get_wandb_config(self, 
                        base_name: str,
                        add_llm_suffix: bool = None) -> Dict[str, str]:
        """
        Generate Weights & Biases configuration with LLM suffix
        
        Args:
            base_name: Base experiment name
            add_llm_suffix: Whether to add LLM suffix (auto-detect if None)
            
        Returns:
            W&B configuration dictionary
        """
        if add_llm_suffix is None:
            add_llm_suffix = self.enable_llm and self.llm_available
        
        suffix = "-LLM" if add_llm_suffix else ""
        return {"name": f"{base_name}{suffix}"}


def create_llm_enhancer(enable_llm: bool = True, 
                       sample_limit: int = 1000,
                       **kwargs) -> LLMTrainingEnhancer:
    """
    Factory function to create LLM enhancer with common settings
    
    Args:
        enable_llm: Whether to enable LLM enhancement
        sample_limit: Maximum samples to process
        **kwargs: Additional arguments for LLMTrainingEnhancer
        
    Returns:
        Configured LLMTrainingEnhancer instance
    """
    return LLMTrainingEnhancer(
        enable_llm=enable_llm,
        sample_limit=sample_limit,
        **kwargs
    )


# Convenience functions for quick integration

def enhance_training_data(train_data: pd.DataFrame,
                         method: str = 'Generic',
                         enable_llm: bool = True,
                         sample_limit: int = 1000,
                         **kwargs) -> pd.DataFrame:
    """
    Quick function to enhance training data with LLM
    
    Args:
        train_data: Training DataFrame
        method: Method name for logging
        enable_llm: Whether to enable LLM
        sample_limit: Sample limit for processing
        **kwargs: Additional enhancer arguments
        
    Returns:
        Enhanced training DataFrame
    """
    enhancer = create_llm_enhancer(
        enable_llm=enable_llm,
        sample_limit=sample_limit,
        **kwargs
    )
    
    return enhancer.enhance_training_data(train_data, method=method)


def get_llm_model_config(base_output_dir: str,
                        base_wandb_name: str,
                        enable_llm: bool = True,
                        **kwargs) -> Dict[str, Any]:
    """
    Get model configuration with LLM enhancements
    
    Args:
        base_output_dir: Base output directory
        base_wandb_name: Base W&B experiment name
        enable_llm: Whether LLM is enabled
        **kwargs: Additional enhancer arguments
        
    Returns:
        Configuration dictionary with enhanced paths
    """
    enhancer = create_llm_enhancer(enable_llm=enable_llm, **kwargs)
    
    return {
        'output_dir': enhancer.get_enhanced_model_path(base_output_dir, 'model'),
        'wandb_kwargs': enhancer.get_wandb_config(base_wandb_name),
        'llm_enabled': enhancer.llm_available and enable_llm
    }


if __name__ == "__main__":
    # Example usage
    print("🧪 Testing LLM Integration Module...")
    
    # Create test data
    test_data = pd.DataFrame({
        'query_text': ['What is machine learning?', 'How does AI work?'],
        'hard_negative': ['Machine learning is fun', 'AI is complex'],
        'positive_passage': ['ML is a subset of AI', 'AI uses algorithms']
    })
    
    # Test enhancement
    enhancer = create_llm_enhancer(sample_limit=2)
    enhanced = enhancer.enhance_training_data(test_data, method='Test')
    
    print(f"✅ Test completed. Enhanced data shape: {enhanced.shape}")
    if 'llm_classification' in enhanced.columns:
        print(f"   LLM classifications: {enhanced['llm_classification'].value_counts().to_dict()}")