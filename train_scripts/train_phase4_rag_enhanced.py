"""
Phase 4: Train DPR with RAG-Selected Hard Negatives
Based on Implementation.ipynb cells 29
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
USE_MIXED_PRECISION = True
MAX_SEQ_LENGTH = 256
TRAIN_BATCH_SIZE = 8

def clear_gpu_memory():
    """Clear GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        import gc
        gc.collect()

def check_prerequisites():
    """Check if Phase 3 and RAG preprocessing are complete"""
    print("\n" + "="*60)
    print("CHECKING PREREQUISITES")
    print("="*60)
    
    # Check RAG-selected data
    RAG_CSV = os.path.join(DATA_DIR, "llm_classified_data", "rag_selected_negatives", "selected_high_quality.csv")
    if not os.path.exists(RAG_CSV):
        print(f"\n❌ RAG-selected data not found: {RAG_CSV}")
        print("\nRun preprocessing first:")
        print("  python data_processing/rag_selection.py")
        return False
    
    print(f"✓ RAG-selected data found: {RAG_CSV}")
    
    # Check Phase 3 model (required)
    phase3_model_path = f"{MODEL_DIR}/dpr_llm_enhanced"
    if not os.path.exists(phase3_model_path):
        print(f"\n❌ Phase 3 model required: {phase3_model_path}")
        print("\nRun Phase 3 training first:")
        print("  python train_scripts/train_phase3_llm_enhanced.py")
        return False
    
    print(f"✓ Phase 3 model found: {phase3_model_path}")
    return True

def main():
    print("="*60)
    print("PHASE 4: TRAIN DPR WITH RAG-SELECTED NEGATIVES")
    print("="*60)
    
    # Check prerequisites
    if not check_prerequisites():
        return False
    
    # Load RAG-selected training data
    RAG_CSV = os.path.join(DATA_DIR, "llm_classified_data", "rag_selected_negatives", "selected_high_quality.csv")
    
    print(f"\nLoading RAG-selected data from: {RAG_CSV}")
    rag_final = pd.read_csv(RAG_CSV)
    
    # Prepare training data
    train_df = rag_final[['query_text', 'gold_passage', 'hard_negative']].copy()
    print(f"✓ Loaded {len(train_df):,} samples")
    
    # Configure model args
    phase4_args = RetrievalArgs()
    phase4_args.data_format = "beir"
    phase4_args.max_seq_length = MAX_SEQ_LENGTH
    phase4_args.include_title = False
    phase4_args.hard_negatives = True
    phase4_args.num_train_epochs = 3
    phase4_args.train_batch_size = TRAIN_BATCH_SIZE
    phase4_args.learning_rate = 5e-7
    phase4_args.output_dir = f"{MODEL_DIR}/dpr_rag_phase4"
    phase4_args.fp16 = USE_MIXED_PRECISION
    phase4_args.evaluate_during_training = False
    phase4_args.save_model_every_epoch = False
    phase4_args.overwrite_output_dir = True
    phase4_args.use_cached_eval_features = False
    
    print(f"\nTraining configuration:")
    print(f"  Samples: {len(train_df):,}")
    print(f"  Epochs: {phase4_args.num_train_epochs}")
    print(f"  Output: {phase4_args.output_dir}")
    
    # Load Phase 3 model (required)
    phase3_model_path = f"{MODEL_DIR}/dpr_llm_enhanced"
    
    if not os.path.exists(phase3_model_path):
        raise FileNotFoundError(f"Phase 3 model required at: {phase3_model_path}")
    
    try:
        print(f"\nLoading Phase 3 model: {phase3_model_path}")
        dpr_rag_model = RetrievalModel(
            model_type="custom",
            model_name=phase3_model_path,
            args=phase4_args,
            use_cuda=torch.cuda.is_available()
        )
        print("✓ Phase 3 model loaded")
        
        clear_gpu_memory()
        
        # Train
        print(f"\nStarting Phase 4 training...")
        start_time = time.time()
        
        dpr_rag_model.train_model(train_df)
        
        training_time = (time.time() - start_time) / 60
        
        print("\n" + "="*60)
        print("✅ PHASE 4 TRAINING COMPLETE")
        print("="*60)
        print(f"Training time: {training_time:.2f} minutes")
        print(f"Model saved to: {phase4_args.output_dir}")
        
        clear_gpu_memory()
        return True
        
    except Exception as e:
        print(f"❌ PHASE 4 TRAINING FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
