"""
Phase 1: Train DPR with BM25-mined Hard Negatives
Based on Implementation.ipynb cells 1-16
"""

import os
import sys
import torch
import pandas as pd
from simpletransformers.retrieval import RetrievalModel, RetrievalArgs
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

# Configuration
DATA_DIR = "data"
MODEL_DIR = "./models"
BASE_MODEL = "bert-base-multilingual-cased"
USE_MIXED_PRECISION = True
MAX_SEQ_LENGTH = 256
TRAIN_BATCH_SIZE = 8
DEV_MODE = True

def clear_gpu_memory():
    """Clear GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        print("✓ GPU memory cleared")

def check_prerequisites():
    """Check if preprocessing is complete"""
    print("\n" + "="*60)
    print("CHECKING PREREQUISITES")
    print("="*60)
    
    TRAINING_DATA_DIR = os.path.join(DATA_DIR, "training_data")
    msmarco_csv = os.path.join(TRAINING_DATA_DIR, "msmarco", "train.csv")
    
    if not os.path.exists(msmarco_csv):
        print(f"\n❌ Training data not found: {msmarco_csv}")
        print("\nRun preprocessing first:")
        print("  python data_processing/prepare_bm25_negatives.py")
        return False
    
    print(f"✓ Training data found: {msmarco_csv}")
    return True

def main():
    print("="*60)
    print("PHASE 1: TRAINING DPR WITH BM25 NEGATIVES")
    print("="*60)
    
    # Check prerequisites
    if not check_prerequisites():
        return False
    
    # Load training data
    TRAINING_DATA_DIR = os.path.join(DATA_DIR, "training_data")
    msmarco_csv = os.path.join(TRAINING_DATA_DIR, "msmarco", "train.csv")
    
    print(f"\nLoading MS MARCO training data from: {msmarco_csv}")
    msmarco_train_df = pd.read_csv(msmarco_csv)
    print(f"✓ Loaded {len(msmarco_train_df):,} samples")
    
    # Configure model
    model_args = RetrievalArgs()
    model_args.data_format = "beir"
    model_args.reprocess_input_data = True
    model_args.overwrite_output_dir = True
    model_args.use_cached_eval_features = False
    model_args.use_hf_datasets = True
    model_args.include_title = False
    model_args.max_seq_length = MAX_SEQ_LENGTH
    model_args.num_train_epochs = 5
    model_args.train_batch_size = TRAIN_BATCH_SIZE
    model_args.learning_rate = 1e-6
    model_args.warmup_steps = 5000
    model_args.save_steps = 300000
    model_args.hard_negatives = True
    model_args.evaluate_during_training = False
    model_args.save_model_every_epoch = False
    model_args.n_gpu = 1
    model_args.fp16 = USE_MIXED_PRECISION
    model_args.dataloader_num_workers = 4
    model_args.ance_training = False
    model_args.output_dir = f"{MODEL_DIR}/dpr_bm25_baseline_epoch5"
    
    print(f"\n✓ Model configuration complete")
    print(f"  Output dir: {model_args.output_dir}")
    print(f"  Epochs: {model_args.num_train_epochs}")
    print(f"  Batch size: {model_args.train_batch_size}")
    
    # Clear GPU memory
    clear_gpu_memory()
    
    # Initialize DPR model
    print(f"\nInitializing DPR model with: {BASE_MODEL}")
    dpr_model = RetrievalModel(
        model_type="custom",
        model_name=None,
        context_encoder_name=BASE_MODEL,
        query_encoder_name=BASE_MODEL,
        args=model_args,
        use_cuda=torch.cuda.is_available()
    )
    print(f"✓ DPR model initialized")
    print("="*60)
    
    # Train
    import time
    start_time = time.time()
    
    try:
        dpr_model.train_model(msmarco_train_df, eval_set="dev")
        
        training_time = time.time() - start_time
        
        print("\n" + "="*60)
        print("✅ PHASE 1 TRAINING COMPLETE!")
        print("="*60)
        print(f"Total training time: {training_time/60:.1f} minutes")
        print(f"Model saved to: {model_args.output_dir}")
        
        clear_gpu_memory()
        return True
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("\n❌ OUT OF MEMORY ERROR")
            print(f"Try reducing train_batch_size (currently: {model_args.train_batch_size})")
        return False
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
