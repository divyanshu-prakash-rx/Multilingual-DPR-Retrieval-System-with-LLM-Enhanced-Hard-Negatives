"""
RAG-Based Negative Selection
Builds FAISS index and selects high-quality hard negatives using RAG
Based on Implementation.ipynb cells 26-29
"""

import os
import pickle
import json
import time
import faiss
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import requests

# Configuration
DATA_DIR = "data"
MODEL_DIR = "./models"
LLM_DATA_DIR = os.path.join(DATA_DIR, "llm_classified_data")
RAG_OUTPUT_DIR = os.path.join(LLM_DATA_DIR, "rag_selected_negatives")
RAG_INDEX_DIR = os.path.join(MODEL_DIR, "rag_index")

BASE_MODEL = "bert-base-multilingual-cased"
USE_OLLAMA = True
OLLAMA_MODEL = "llama3.1:8b"
OLLAMA_URL = "http://localhost:11434"

QUALITY_PERCENTILE = 0.5  # Keep top 50%

def check_prerequisites():
    """Check if LLM processing is complete"""
    print("\n" + "="*60)
    print("CHECKING PREREQUISITES")
    print("="*60)
    
    final_csv = os.path.join(LLM_DATA_DIR, "llm_final_training", "hard_negatives_final.csv")
    
    if not os.path.exists(final_csv):
        print(f"\n❌ Missing: {final_csv}")
        print("\nRun LLM processing first:")
        print("  python data_processing/llm_classification.py")
        return False
    
    print(f"✓ LLM data found: {final_csv}")
    
    # Test Ollama
    if USE_OLLAMA:
        try:
            response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
            if response.status_code == 200:
                print(f"✓ Ollama is running")
            else:
                print("⚠ Ollama not responding - RAG scoring may fail")
        except:
            print("⚠ Cannot connect to Ollama - RAG scoring may fail")
    
    return True

def build_rag_index(hard_negatives_df):
    """Build FAISS index from hard negatives"""
    print("\n" + "="*60)
    print("BUILDING RAG INDEX")
    print("="*60)
    
    # Extract corpus
    corpus_passages = hard_negatives_df['hard_negative'].dropna().unique().tolist()
    print(f"✓ Corpus size: {len(corpus_passages):,} unique passages")
    
    # Initialize encoder
    print(f"\nLoading encoder: {BASE_MODEL}")
    encoder = SentenceTransformer(BASE_MODEL)
    encoder.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.to(device)
    print(f"✓ Encoder ready on {device}")
    
    # Encode corpus
    batch_size = 32
    all_embeddings = []
    
    print(f"\nEncoding {len(corpus_passages)} passages...")
    for i in tqdm(range(0, len(corpus_passages), batch_size), desc="Encoding"):
        batch = corpus_passages[i:i + batch_size]
        
        with torch.no_grad():
            embeddings = encoder.encode(
                batch,
                convert_to_numpy=True,
                show_progress_bar=False,
                batch_size=batch_size
            )
        all_embeddings.append(embeddings)
    
    corpus_embeddings = np.vstack(all_embeddings)
    print(f"✓ Encoded: {corpus_embeddings.shape}")
    
    # Build FAISS index
    print("\nBuilding FAISS index...")
    embedding_dim = corpus_embeddings.shape[1]
    index = faiss.IndexFlatIP(embedding_dim)
    
    faiss.normalize_L2(corpus_embeddings)
    index.add(corpus_embeddings.astype('float32'))
    
    print(f"✓ Index built: {index.ntotal:,} passages")
    
    # Save
    os.makedirs(RAG_INDEX_DIR, exist_ok=True)
    
    index_path = os.path.join(RAG_INDEX_DIR, "rag_corpus_index.faiss")
    faiss.write_index(index, index_path)
    print(f"✓ Saved: {index_path}")
    
    corpus_path = os.path.join(RAG_INDEX_DIR, "rag_corpus_passages.pkl")
    with open(corpus_path, "wb") as f:
        pickle.dump(corpus_passages, f)
    print(f"✓ Saved: {corpus_path}")
    
    metadata = {
        "corpus_size": len(corpus_passages),
        "embedding_model": BASE_MODEL,
        "embedding_dim": embedding_dim,
        "timestamp": pd.Timestamp.now().isoformat()
    }
    
    metadata_path = os.path.join(RAG_INDEX_DIR, "rag_index_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return encoder, index, corpus_passages

class RAGNegativeRanker:
    """Score negatives using LLM with RAG context"""
    
    def __init__(self, encoder, index, corpus, ollama_url, ollama_model):
        self.encoder = encoder
        self.index = index
        self.corpus = corpus
        self.ollama_url = ollama_url
        self.ollama_model = ollama_model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def retrieve_context(self, query, top_k=3):
        """Retrieve relevant context passages"""
        with torch.no_grad():
            query_emb = self.encoder.encode(query, convert_to_numpy=True, show_progress_bar=False)
        
        query_emb_normalized = query_emb / (np.linalg.norm(query_emb) + 1e-8)
        query_emb_normalized = query_emb_normalized.reshape(1, -1).astype('float32')
        
        scores, indices = self.index.search(query_emb_normalized, top_k)
        
        contexts = []
        for idx in indices[0]:
            if 0 <= idx < len(self.corpus):
                contexts.append(self.corpus[idx])
        
        return contexts
    
    def call_ollama(self, prompt):
        """Call Ollama API"""
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.2, "num_predict": 100}
                },
                timeout=60
            )
            
            if response.status_code == 200:
                return response.json().get("response", "").strip()
        except:
            pass
        
        return "50"
    
    def extract_score(self, response):
        """Extract score from LLM response"""
        try:
            score = float(response)
            if score > 1.0:
                score = score / 100.0
            return min(max(score, 0.0), 1.0)
        except:
            return 0.5
    
    def score_negative(self, query, gold_passage, negative_passage, top_k_context=3):
        """Score negative with RAG context"""
        try:
            # Retrieve context
            contexts = self.retrieve_context(query, top_k=top_k_context)
            context_str = "\n".join([f"• {c[:120]}" for c in contexts[:top_k_context]])
            
            # Enhanced query with context
            enhanced_query = f"Query: {query}\n\nContext:\n{context_str}"
            
            # LLM scoring
            prompt = f"""Rate hardness (0-100):

{enhanced_query}

Gold: {gold_passage[:200]}
Negative: {negative_passage[:200]}

HARD (70-100): Related but wrong answer
EASY (0-30): Unrelated topic

Number only:"""
            
            response = self.call_ollama(prompt)
            score = self.extract_score(response)
            
            return score
        except:
            return 0.5

def score_negatives_with_rag(hard_negatives_df, ranker):
    """Score all negatives using RAG"""
    print("\n" + "="*60)
    print("RAG SCORING")
    print("="*60)
    
    results = []
    
    print(f"\nScoring {len(hard_negatives_df)} negatives...")
    
    for idx, row in tqdm(hard_negatives_df.iterrows(), total=len(hard_negatives_df), desc="Scoring"):
        try:
            score = ranker.score_negative(
                query=row.get('query_text', ''),
                gold_passage=row.get('gold_passage', ''),
                negative_passage=row.get('hard_negative', ''),
                top_k_context=3
            )
            
            results.append({
                'query_text': row['query_text'],
                'gold_passage': row['gold_passage'],
                'hard_negative': row['hard_negative'],
                'rag_score': score
            })
        except:
            continue
    
    rag_df = pd.DataFrame(results)
    
    print(f"\n✓ Scoring complete:")
    print(f"  Scored: {len(rag_df):,}")
    print(f"  Mean score: {rag_df['rag_score'].mean():.4f}")
    print(f"  Median: {rag_df['rag_score'].median():.4f}")
    
    return rag_df

def select_high_quality_negatives(rag_df):
    """Select top negatives based on RAG scores"""
    print("\n" + "="*60)
    print("SELECTING HIGH-QUALITY NEGATIVES")
    print("="*60)
    
    threshold = rag_df['rag_score'].quantile(1 - QUALITY_PERCENTILE)
    
    print(f"\nQuality threshold: {threshold:.4f} (top {QUALITY_PERCENTILE*100:.0f}%)")
    
    selected_df = rag_df[rag_df['rag_score'] >= threshold].copy()
    selected_df = selected_df.sort_values('rag_score', ascending=False).reset_index(drop=True)
    
    print(f"✓ Selected: {len(selected_df):,} / {len(rag_df):,} ({len(selected_df)/len(rag_df)*100:.1f}%)")
    
    # Save
    os.makedirs(RAG_OUTPUT_DIR, exist_ok=True)
    
    # All scored
    all_csv = os.path.join(RAG_OUTPUT_DIR, "all_ranked_with_scores.csv")
    rag_df.to_csv(all_csv, index=False)
    print(f"✓ Saved all: {all_csv}")
    
    # Selected high-quality
    selected_csv = os.path.join(RAG_OUTPUT_DIR, "selected_high_quality.csv")
    selected_df.to_csv(selected_csv, index=False)
    print(f"✓ Saved selected: {selected_csv}")
    
    # Metadata
    metadata = {
        "total_scored": len(rag_df),
        "selected": len(selected_df),
        "selection_percentage": float(len(selected_df)/len(rag_df)*100),
        "quality_percentile": QUALITY_PERCENTILE,
        "threshold": float(threshold),
        "score_stats": {
            "mean": float(rag_df['rag_score'].mean()),
            "median": float(rag_df['rag_score'].median()),
            "min": float(rag_df['rag_score'].min()),
            "max": float(rag_df['rag_score'].max())
        },
        "timestamp": pd.Timestamp.now().isoformat()
    }
    
    metadata_path = os.path.join(RAG_OUTPUT_DIR, "rag_selection_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return selected_df

def main():
    print("="*60)
    print("RAG-BASED NEGATIVE SELECTION")
    print("="*60)
    
    if not check_prerequisites():
        return False
    
    # Load LLM-processed data
    final_csv = os.path.join(LLM_DATA_DIR, "llm_final_training", "hard_negatives_final.csv")
    print(f"\nLoading: {final_csv}")
    hard_negatives_df = pd.read_csv(final_csv)
    print(f"✓ Loaded {len(hard_negatives_df):,} samples")
    
    # Build RAG index
    encoder, index, corpus = build_rag_index(hard_negatives_df)
    
    # Initialize ranker
    print("\n" + "="*60)
    print("INITIALIZING RAG RANKER")
    print("="*60)
    
    ranker = RAGNegativeRanker(encoder, index, corpus, OLLAMA_URL, OLLAMA_MODEL)
    print("✓ RAG Ranker ready")
    
    # Score negatives
    rag_df = score_negatives_with_rag(hard_negatives_df, ranker)
    
    # Select high-quality
    selected_df = select_high_quality_negatives(rag_df)
    
    print("\n" + "="*60)
    print("✅ RAG SELECTION COMPLETE")
    print("="*60)
    print(f"\nData saved in: {RAG_OUTPUT_DIR}/")
    print(f"RAG index saved in: {RAG_INDEX_DIR}/")
    print(f"\nNext: python train_scripts/train_phase4_rag_enhanced.py")
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
