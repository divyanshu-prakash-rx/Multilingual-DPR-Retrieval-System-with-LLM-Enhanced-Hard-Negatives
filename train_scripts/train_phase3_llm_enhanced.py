"""
Phase 3: Train DPR with LLM-Enhanced Hard Negatives
Based on Implementation.ipynb cells 24-25
"""

import os
import torch
import pandas as pd
import time
from simpletransformers.retrieval import RetrievalModel, RetrievalArgs
import logging

logging.basicConfig(level=logging.INFO)

# Configuration
DATA_DIR = "data"
MODEL_DIR = "./models"
BASE_MODEL = "bert-base-multilingual-cased"
USE_MIXED_PRECISION = True
MAX_SEQ_LENGTH = 256
TRAIN_BATCH_SIZE = 8

def clear_gpu_memory():
    """Clear GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        print("✓ GPU memory cleared")

def check_prerequisites():
    """Check if Phase 1 and preprocessing are complete"""
    print("\n" + "="*60)
    print("CHECKING PREREQUISITES")
    print("="*60)
    
    # Check LLM-enhanced data
    FINAL_CSV = os.path.join(DATA_DIR, "llm_classified_data", "llm_final_training", "hard_negatives_final.csv")
    if not os.path.exists(FINAL_CSV):
        print(f"\n❌ LLM-enhanced data not found: {FINAL_CSV}")
        print("\nRun preprocessing first:")
        print("  python data_processing/llm_classification.py")
        return False
    
    print(f"✓ LLM-enhanced data found: {FINAL_CSV}")
    
    # Check Phase 1 model (optional but recommended)
    phase1_dir = os.path.join(MODEL_DIR, "dpr_bm25_baseline_epoch5")
    if os.path.exists(phase1_dir):
        print(f"✓ Phase 1 model found: {phase1_dir}")
    else:
        print(f"⚠ Phase 1 model not found - will train from scratch")
    
    return True

def main():
    print("="*60)
    print("PHASE 3: TRAIN DPR WITH LLM-ENHANCED DATA")
    print("="*60)
    
    # Check prerequisites
    if not check_prerequisites():
        return False
    
    # Load LLM-enhanced training data
    FINAL_CSV = os.path.join(DATA_DIR, "llm_classified_data", "llm_final_training", "hard_negatives_final.csv")
    
    print(f"\nLoading training data from: {FINAL_CSV}")
    final_train_df = pd.read_csv(FINAL_CSV)
    print(f"✓ Loaded {len(final_train_df):,} samples")
    
    # Configure model args
    phase3_args = RetrievalArgs()
    phase3_args.data_format = "beir"
    phase3_args.hard_negatives = True
    phase3_args.num_train_epochs = 3
    phase3_args.train_batch_size = TRAIN_BATCH_SIZE
    phase3_args.learning_rate = 5e-7
    phase3_args.max_seq_length = MAX_SEQ_LENGTH
    phase3_args.output_dir = os.path.join(MODEL_DIR, "dpr_llm_enhanced")
    phase3_args.fp16 = USE_MIXED_PRECISION
    phase3_args.evaluate_during_training = False
    phase3_args.save_model_every_epoch = False
    phase3_args.overwrite_output_dir = True
    phase3_args.include_title = False
    
    print(f"\nTraining configuration:")
    print(f"  Samples: {len(final_train_df):,}")
    print(f"  Epochs: {phase3_args.num_train_epochs}")
    print(f"  Output: {phase3_args.output_dir}")
    
    clear_gpu_memory()
    
    # Load Phase 1 model as starting point
    phase1_dir = os.path.join(MODEL_DIR, "dpr_bm25_baseline_epoch5")
    
    try:
        if os.path.exists(phase1_dir):
            print(f"\nLoading Phase 1 checkpoint: {phase1_dir}")
            dpr_model = RetrievalModel(
                model_type="custom",
                model_name=phase1_dir,
                args=phase3_args,
                use_cuda=torch.cuda.is_available()
            )
            print("✓ Loaded Phase 1 checkpoint")
        else:
            print(f"\nInitializing from scratch with: {BASE_MODEL}")
            dpr_model = RetrievalModel(
                model_type="custom",
                model_name=None,
                context_encoder_name=BASE_MODEL,
                query_encoder_name=BASE_MODEL,
                args=phase3_args,
                use_cuda=torch.cuda.is_available()
            )
            print(f"✓ Initialized with {BASE_MODEL}")
        
        # Train
        print(f"\nStarting training...")
        start_time = time.time()
        
        dpr_model.train_model(final_train_df)
        
        training_time = time.time() - start_time
        
        print("\n" + "="*60)
        print("✅ PHASE 3 COMPLETE")
        print("="*60)
        print(f"Model saved to: {phase3_args.output_dir}")
        print(f"Training time: {training_time/60:.1f} minutes")
        
        clear_gpu_memory()
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
