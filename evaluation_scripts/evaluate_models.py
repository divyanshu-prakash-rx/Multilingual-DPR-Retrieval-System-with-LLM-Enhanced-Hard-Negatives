"""
Comprehensive Model Evaluation Script
Evaluates all trained models on MS MARCO dev set
Based on Implementation.ipynb cells 17, 26, 31
"""

import os
import torch
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
from simpletransformers.retrieval import RetrievalModel, RetrievalArgs

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

def evaluate_dpr_model(model, eval_df, model_name, device, top_k=10, max_samples=None):
    """
    Evaluate DPR model on retrieval task
    
    Args:
        model: RetrievalModel instance
        eval_df: DataFrame with columns ['query_text', 'gold_passage']
        model_name: Name for logging
        device: torch device
        top_k: Recall cutoff (default 10)
        max_samples: Limit samples for speed (None = use all)
    
    Returns:
        Dictionary with metrics
    """
    
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*60}\n")
    
    # Limit samples if specified
    if max_samples:
        eval_subset = eval_df.head(max_samples).copy()
        print(f"Using {len(eval_subset)} samples (limited from {len(eval_df)})")
    else:
        eval_subset = eval_df.copy()
        print(f"Using all {len(eval_subset)} samples")
    
    mrr_scores = []
    ndcg_scores = []
    recall_1 = []
    recall_5 = []
    recall_10 = []
    
    # Get all passages as corpus
    all_passages = eval_subset['gold_passage'].tolist()
    print(f"Corpus size: {len(all_passages)} passages\n")
    
    # Evaluate each query
    for idx, row in tqdm(eval_subset.iterrows(), total=len(eval_subset), desc="Evaluating"):
        query = row['query_text']
        gold_passage = row['gold_passage']
        
        try:
            with torch.no_grad():
                # Encode query
                query_features = model.query_tokenizer(
                    query, 
                    padding='max_length',
                    truncation=True, 
                    max_length=256, 
                    return_tensors='pt'
                )
                
                query_input_ids = query_features['input_ids'].to(device)
                query_attention_mask = query_features['attention_mask'].to(device)
                
                query_emb = model.query_encoder(
                    input_ids=query_input_ids,
                    attention_mask=query_attention_mask
                )[1].cpu().numpy()
                
                # Score all passages
                passage_scores = []
                for passage in all_passages:
                    passage_features = model.context_tokenizer(
                        passage,
                        padding='max_length',
                        truncation=True,
                        max_length=256,
                        return_tensors='pt'
                    )
                    
                    passage_input_ids = passage_features['input_ids'].to(device)
                    passage_attention_mask = passage_features['attention_mask'].to(device)
                    
                    passage_emb = model.context_encoder(
                        input_ids=passage_input_ids,
                        attention_mask=passage_attention_mask
                    )[1].cpu().numpy()
                    
                    # Cosine similarity
                    score = np.dot(query_emb[0], passage_emb[0]) / (
                        np.linalg.norm(query_emb[0]) * np.linalg.norm(passage_emb[0]) + 1e-8
                    )
                    passage_scores.append(score)
                
                # Rank passages by score
                ranked_idx = np.argsort(passage_scores)[::-1]
                
                # Find rank of gold passage
                gold_rank = len(all_passages) + 1
                for rank, pidx in enumerate(ranked_idx):
                    if all_passages[pidx] == gold_passage:
                        gold_rank = rank + 1
                        break
                
                # Compute metrics
                if gold_rank <= top_k:
                    mrr_scores.append(1.0 / gold_rank)
                else:
                    mrr_scores.append(0.0)
                
                if gold_rank <= top_k:
                    ndcg_scores.append(1.0 / np.log2(gold_rank + 1))
                else:
                    ndcg_scores.append(0.0)
                
                recall_1.append(1.0 if gold_rank <= 1 else 0.0)
                recall_5.append(1.0 if gold_rank <= 5 else 0.0)
                recall_10.append(1.0 if gold_rank <= 10 else 0.0)
        
        except Exception as e:
            continue
    
    # Aggregate metrics
    metrics = {
        "MRR@10": np.mean(mrr_scores) if mrr_scores else 0.0,
        "nDCG@10": np.mean(ndcg_scores) if ndcg_scores else 0.0,
        "Recall@1": np.mean(recall_1) if recall_1 else 0.0,
        "Recall@5": np.mean(recall_5) if recall_5 else 0.0,
        "Recall@10": np.mean(recall_10) if recall_10 else 0.0,
    }
    
    return metrics

def main():
    print("="*60)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("="*60)
    
    # Load evaluation dataset
    eval_data_path = f"{DATA_DIR}/msmarco_dev_beir.tsv"
    
    if not os.path.exists(eval_data_path):
        raise FileNotFoundError(f"Evaluation data not found: {eval_data_path}")
    
    print(f"\nLoading evaluation data from: {eval_data_path}")
    msmarco_dev = pd.read_csv(eval_data_path, sep="\t")
    print(f"✓ Loaded {len(msmarco_dev):,} query-passage pairs")
    
    # Models to evaluate
    models_to_eval = {
        "Phase 2 (BM25 Baseline)": f"{MODEL_DIR}/dpr_bm25_baseline_epoch5",
        "Phase 3 (LLM Enhanced)": f"{MODEL_DIR}/dpr_llm_enhanced",
        "Phase 4 (RAG Enhanced)": f"{MODEL_DIR}/dpr_rag_phase4"
    }
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}\n")
    
    # Evaluation args
    eval_args = RetrievalArgs()
    eval_args.data_format = "beir"
    eval_args.max_seq_length = 256
    eval_args.include_title = False
    eval_args.hard_negatives = False
    eval_args.fp16 = USE_MIXED_PRECISION
    
    # Store results
    all_results = {}
    
    # Evaluate each model
    for model_name, model_path in models_to_eval.items():
        print(f"\n{'='*60}")
        print(f"Loading: {model_name}")
        print(f"{'='*60}")
        
        if not os.path.exists(model_path):
            print(f"❌ Model not found: {model_path}")
            print("Skipping...\n")
            continue
        
        try:
            # Load model
            model = RetrievalModel(
                model_type="custom",
                model_name=model_path,
                args=eval_args,
                use_cuda=torch.cuda.is_available()
            )
            model.query_encoder = model.query_encoder.to(device)
            model.context_encoder = model.context_encoder.to(device)
            print(f"✓ Model loaded\n")
            
            # Evaluate
            metrics = evaluate_dpr_model(
                model, 
                msmarco_dev, 
                model_name, 
                device,
                max_samples=300  # Evaluate on 300 samples
            )
            
            all_results[model_name] = metrics
            
            # Display results
            print(f"\nResults for {model_name}:")
            print(f"{'Metric':<20} {'Score':<15}")
            print("-" * 35)
            for metric, score in metrics.items():
                print(f"{metric:<20} {score:<15.4f}")
            
            clear_gpu_memory()
            
        except Exception as e:
            print(f"❌ Evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Compare all models
    if len(all_results) > 1:
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)
        
        # Create comparison table
        metrics_list = list(next(iter(all_results.values())).keys())
        
        comparison_df = pd.DataFrame({
            "Metric": metrics_list,
            **{
                model_name: [f"{all_results[model_name][m]:.4f}" for m in metrics_list]
                for model_name in all_results.keys()
            }
        })
        
        print("\n" + comparison_df.to_string(index=False))
        
        # Calculate improvements
        if "Phase 2 (BM25 Baseline)" in all_results:
            baseline = all_results["Phase 2 (BM25 Baseline)"]
            
            print("\n" + "="*60)
            print("IMPROVEMENT OVER BASELINE")
            print("="*60)
            
            for model_name, metrics in all_results.items():
                if model_name == "Phase 2 (BM25 Baseline)":
                    continue
                
                improvements = {
                    m: ((metrics[m] - baseline[m]) / max(baseline[m], 0.0001) * 100)
                    for m in metrics_list
                }
                avg_improvement = np.mean(list(improvements.values()))
                
                print(f"\n{model_name}:")
                print(f"  Average improvement: {avg_improvement:+.2f}%")
                for metric, imp in improvements.items():
                    print(f"    {metric}: {imp:+.2f}%")
    
    # Save results
    results_path = f"{MODEL_DIR}/phase5_complete_results.json"
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ Results saved to: {results_path}")
    
    print("\n" + "="*60)
    print("✅ EVALUATION COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
