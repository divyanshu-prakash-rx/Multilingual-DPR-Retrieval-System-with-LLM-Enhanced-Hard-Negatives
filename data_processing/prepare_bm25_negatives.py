"""
BM25 Hard Negative Mining
Mines hard negatives using BM25 for MS MARCO and TyDi data
Based on Implementation.ipynb cells 6-10
"""

import os
import pickle
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from rank_bm25 import BM25Okapi
from typing import List

# Configuration
DATA_DIR = "data"
TRAINING_DATA_DIR = os.path.join(DATA_DIR, "training_data")
MSMARCO_DIR = os.path.join(DATA_DIR, "msmarco")
TYDI_DIR = os.path.join(DATA_DIR, "tydi")
DEV_MODE = True
DEV_SAMPLE_SIZE = 1000 if DEV_MODE else None

class BM25NegativeSampler:
    """Mine hard negatives using BM25"""
    
    def __init__(self, corpus: List[str]):
        print("Building BM25 index...")
        tokenized_corpus = [doc.lower().split() for doc in corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.corpus = corpus
        print(f"✓ BM25 index built with {len(corpus):,} documents")
    
    def get_hard_negatives(self, query: str, positive_passage: str, top_k: int = 100, n_negatives: int = 1) -> List[str]:
        """Get hard negatives for a query"""
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        negatives = []
        for idx in top_indices:
            candidate = self.corpus[idx]
            if candidate != positive_passage and candidate not in negatives:
                negatives.append(candidate)
            if len(negatives) >= n_negatives:
                break
        
        while len(negatives) < n_negatives:
            random_idx = np.random.randint(0, len(self.corpus))
            candidate = self.corpus[random_idx]
            if candidate != positive_passage and candidate not in negatives:
                negatives.append(candidate)
        
        return negatives[:n_negatives]

def clean_dataset(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """Remove nulls, duplicates, and invalid samples"""
    initial_size = len(df)
    
    df = df.dropna()
    df = df.drop_duplicates()
    df = df[
        (df['query_text'].str.strip() != '') & 
        (df['gold_passage'].str.strip() != '')
    ]
    df = df[
        (df['query_text'].str.len() >= 10) &
        (df['query_text'].str.len() <= 512) &
        (df['gold_passage'].str.len() >= 20) &
        (df['gold_passage'].str.len() <= 2048)
    ]
    
    if 'hard_negative' in df.columns:
        df = df[df['hard_negative'] != df['gold_passage']]
    
    print(f"  {dataset_name}: {initial_size:,} → {len(df):,} ({len(df)/initial_size*100:.1f}% retained)")
    return df.reset_index(drop=True)

def check_prerequisites():
    """Check if required data files exist"""
    print("\n" + "="*60)
    print("CHECKING PREREQUISITES")
    print("="*60)
    
    msmarco_train = os.path.join(MSMARCO_DIR, "msmarco-train.tsv")
    
    if not os.path.exists(msmarco_train):
        print(f"\n❌ Missing: {msmarco_train}")
        print("\nRun data download scripts first!")
        return False
    
    print("✓ All prerequisites satisfied")
    return True

def load_msmarco_data():
    """Load and clean MS MARCO data"""
    print("\n" + "="*60)
    print("LOADING MS MARCO DATA")
    print("="*60)
    
    msmarco_train = pd.read_csv(
        f"{MSMARCO_DIR}/msmarco-train.tsv",
        sep="\t",
        nrows=DEV_SAMPLE_SIZE
    )
    
    msmarco_train = msmarco_train.rename(columns={
        "query": "query_text",
        "positive_passage": "gold_passage",
        "negative_passage": "hard_negative"
    })
    
    print(f"✓ MS MARCO loaded: {len(msmarco_train):,} samples")
    msmarco_train = clean_dataset(msmarco_train, "MS MARCO")
    
    return msmarco_train

def load_tydi_data():
    """Load and prepare TyDi data"""
    print("\n" + "="*60)
    print("LOADING TyDi DATA")
    print("="*60)
    
    TYDI_LANGUAGES = ["swahili", "bengali", "telugu", "arabic", "finnish", 
                      "indonesian", "japanese", "korean", "russian"]
    
    tydi_data = []
    
    for lang in TYDI_LANGUAGES:
        base_dir = f"{TYDI_DIR}/{lang}"
        queries_file = os.path.join(base_dir, "queries.jsonl")
        corpus_file = os.path.join(base_dir, "corpus.jsonl")
        qrels_file = os.path.join(base_dir, "qrels/test.tsv")
        
        if not all(os.path.exists(f) for f in [queries_file, corpus_file, qrels_file]):
            print(f"  ⚠ Missing files for {lang}")
            continue
        
        try:
            queries_df = pd.read_json(queries_file, lines=True)
            corpus_df = pd.read_json(corpus_file, lines=True)
            qrels_df = pd.read_csv(qrels_file, sep='\t')
            
            queries_df = queries_df.rename(columns={"_id": "query-id", "text": "query"})
            corpus_df = corpus_df.rename(columns={"_id": "corpus-id", "text": "gold_passage"})
            
            merged = qrels_df.merge(
                queries_df[['query-id', 'query']],
                on="query-id"
            ).merge(
                corpus_df[['corpus-id', 'gold_passage']],
                on="corpus-id"
            )
            
            df = merged[['query', 'gold_passage']].rename(columns={'query': 'query_text'})
            df = df.dropna()
            
            if DEV_SAMPLE_SIZE:
                df = df.sample(min(len(df), 200), random_state=42)
            
            tydi_data.append(df)
            print(f"  ✓ {lang}: {len(df):,}")
        except Exception as e:
            print(f"  ⚠ Error loading {lang}: {e}")
            continue
    
    if tydi_data:
        tydi_combined = pd.concat(tydi_data, ignore_index=True)
        print(f"\n✓ Total TyDi: {len(tydi_combined):,} samples")
        tydi_combined = clean_dataset(tydi_combined, "TyDi")
        return tydi_combined
    
    return pd.DataFrame(columns=["query_text", "gold_passage"])

def mine_hard_negatives(train_df, dataset_name):
    """Mine hard negatives using BM25"""
    print("\n" + "="*60)
    print(f"MINING HARD NEGATIVES - {dataset_name}")
    print("="*60)
    
    # Build corpus
    all_passages = set()
    for _, row in train_df.iterrows():
        all_passages.add(row['gold_passage'])
        if 'hard_negative' in row and pd.notna(row['hard_negative']):
            all_passages.add(row['hard_negative'])
    
    corpus_list = list(all_passages)
    print(f"✓ Corpus size: {len(corpus_list):,} unique passages")
    
    # Initialize BM25
    bm25_sampler = BM25NegativeSampler(corpus_list)
    
    # Mine negatives
    print(f"Mining hard negatives...")
    for idx, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Mining"):
        if 'hard_negative' not in train_df.columns or pd.isna(row.get('hard_negative')):
            hard_negs = bm25_sampler.get_hard_negatives(
                row['query_text'],
                row['gold_passage'],
                top_k=100,
                n_negatives=1
            )
            train_df.at[idx, 'hard_negative'] = hard_negs[0]
        else:
            # Add additional BM25 negative
            bm25_negs = bm25_sampler.get_hard_negatives(
                row['query_text'],
                row['gold_passage'],
                top_k=100,
                n_negatives=1
            )
            if bm25_negs[0] not in str(row['hard_negative']):
                train_df.at[idx, 'hard_negative'] = bm25_negs[0]
    
    print("✓ Hard negative mining complete")
    return train_df

def save_training_data(train_df, dataset_name):
    """Save processed training data"""
    print("\n" + "="*60)
    print(f"SAVING {dataset_name.upper()} DATA")
    print("="*60)
    
    output_dir = os.path.join(TRAINING_DATA_DIR, dataset_name.lower())
    os.makedirs(output_dir, exist_ok=True)
    
    csv_path = os.path.join(output_dir, "train.csv")
    train_df.to_csv(csv_path, index=False)
    print(f"✓ CSV: {csv_path}")
    
    pkl_path = os.path.join(output_dir, "train.pkl")
    with open(pkl_path, 'wb') as f:
        pickle.dump(train_df, f)
    print(f"✓ PKL: {pkl_path}")
    
    print(f"✓ Samples: {len(train_df):,}")

def main():
    print("="*60)
    print("BM25 HARD NEGATIVE MINING")
    print("="*60)
    
    if not check_prerequisites():
        return False
    
    # Process MS MARCO
    msmarco_train = load_msmarco_data()
    msmarco_train = mine_hard_negatives(msmarco_train, "MS MARCO")
    save_training_data(msmarco_train, "msmarco")
    
    # Process TyDi
    tydi_train = load_tydi_data()
    if len(tydi_train) > 0:
        tydi_train = mine_hard_negatives(tydi_train, "TyDi")
        save_training_data(tydi_train, "tydi")
    
    print("\n" + "="*60)
    print("✅ BM25 PREPROCESSING COMPLETE")
    print("="*60)
    print(f"\nData saved in: {TRAINING_DATA_DIR}/")
    print(f"\nNext: python data_processing/llm_classification.py")
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
