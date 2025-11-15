"""
Phase 5: Multilingual Fine-tuning on TyDi QA
Based on Implementation.ipynb cell 38
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

def clear_gpu_memory():
    """Clear GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        import gc
        gc.collect()

def check_prerequisites():
    """Check if previous phases are complete"""
    print("\n" + "="*60)
    print("CHECKING PREREQUISITES")
    print("="*60)
    
    # Check TyDi data
    TRAINING_DATA_DIR = os.path.join(DATA_DIR, "training_data")
    tydi_csv = os.path.join(TRAINING_DATA_DIR, "tydi", "train.csv")
    if not os.path.exists(tydi_csv):
        print(f"\n❌ TyDi training data not found: {tydi_csv}")
        print("\nTyDi preprocessing required - see notebook cells 8-9")
        return False
    
    print(f"✓ TyDi training data found: {tydi_csv}")
    
    # Check trained models
    models = {
        "BM25 Baseline": f"{MODEL_DIR}/dpr_bm25_baseline_epoch5",
        "LLM Enhanced": f"{MODEL_DIR}/dpr_llm_enhanced",
        "RAG Enhanced": f"{MODEL_DIR}/dpr_rag_phase4"
    }
    
    found_models = []
    for name, path in models.items():
        if os.path.exists(path):
            print(f"✓ {name} model found: {path}")
            found_models.append(name)
        else:
            print(f"⚠ {name} model not found: {path}")
    
    if not found_models:
        print("\n❌ No trained models found!")
        print("Train at least one model before fine-tuning")
        return False
    
    print(f"\n✓ Will fine-tune {len(found_models)} model(s)")
    return True

def main():
    print("="*60)
    print("PHASE 5: MULTILINGUAL FINE-TUNING ON TyDi")
    print("="*60)
    
    # Check prerequisites
    if not check_prerequisites():
        return False
    
    # Load TyDi training data
    TRAINING_DATA_DIR = os.path.join(DATA_DIR, "training_data")
    tydi_csv = os.path.join(TRAINING_DATA_DIR, "tydi", "train.csv")
    
    print(f"\nLoading TyDi training data from: {tydi_csv}")
    tydi_train_df = pd.read_csv(tydi_csv)
    print(f"✓ Loaded {len(tydi_train_df):,} samples")
    
    # Models to fine-tune
    models_to_finetune = {
        "BM25 Baseline": f"{MODEL_DIR}/dpr_bm25_baseline_epoch5",
        "LLM Enhanced": f"{MODEL_DIR}/dpr_llm_enhanced",
        "RAG Enhanced": f"{MODEL_DIR}/dpr_rag_phase4"
    }
    
    # Fine-tuning args
    finetune_args = RetrievalArgs()
    finetune_args.num_train_epochs = 3
    finetune_args.train_batch_size = 4
    finetune_args.gradient_accumulation_steps = 2
    finetune_args.learning_rate = 1e-5
    finetune_args.warmup_ratio = 0.1
    finetune_args.max_seq_length = 256
    finetune_args.fp16 = USE_MIXED_PRECISION
    finetune_args.dataloader_num_workers = 0
    finetune_args.logging_steps = 50
    finetune_args.save_model_every_epoch = False
    finetune_args.evaluate_during_training = False
    finetune_args.overwrite_output_dir = True
    finetune_args.include_title = False
    finetune_args.data_format = "beir"
    finetune_args.hard_negatives = True
    
    finetuned_models = {}
    
    for model_name, model_path in models_to_finetune.items():
        print(f"\n{'='*60}")
        print(f"Fine-tuning: {model_name}")
        print(f"{'='*60}")
        
        if not os.path.exists(model_path):
            print(f"❌ Model not found: {model_path}")
            print(f"Skipping {model_name}...")
            continue
        
        try:
            print(f"\nLoading {model_name}...")
            model = RetrievalModel(
                model_type="custom",
                model_name=model_path,
                args=finetune_args,
                use_cuda=torch.cuda.is_available()
            )
            print("✓ Model loaded")
            
            # Set output directory
            output_dir = f"{MODEL_DIR}/dpr_{model_name.lower().replace(' ', '_')}_tydi_final"
            finetune_args.output_dir = output_dir
            model.args = finetune_args
            
            print(f"\nFine-tuning configuration:")
            print(f"  Samples: {len(tydi_train_df):,}")
            print(f"  Epochs: {finetune_args.num_train_epochs}")
            print(f"  Output: {output_dir}")
            
            # Fine-tune
            start_time = time.time()
            model.train_model(tydi_train_df)
            finetune_time = (time.time() - start_time) / 60
            
            print(f"\n✅ {model_name} fine-tuned successfully!")
            print(f"   Time: {finetune_time:.2f} minutes")
            print(f"✓ Saved to {output_dir}")
            
            finetuned_models[model_name] = output_dir
            
        except Exception as e:
            print(f"❌ Fine-tuning failed: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        clear_gpu_memory()
    
    # Summary
    print("\n" + "="*60)
    print("✅ MULTILINGUAL FINE-TUNING COMPLETE")
    print("="*60)
    
    if finetuned_models:
        print(f"\n✓ Fine-tuned models ({len(finetuned_models)}):")
        for name, path in finetuned_models.items():
            print(f"  {name}: {path}")
        return True
    else:
        print("\n❌ No models were successfully fine-tuned!")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
