"""
LLM-Based Hard Negative Classification and Generation
Classifies negatives as HARD/EASY and generates additional hard negatives using LLM
Based on Implementation.ipynb cells 19-24
"""

import os
import json
import time
import pickle
import re
import requests
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from typing import Dict, List

# Configuration
DATA_DIR = "data"
TRAINING_DATA_DIR = os.path.join(DATA_DIR, "training_data")
OUTPUT_DIR = os.path.join(DATA_DIR, "llm_classified_data")

USE_OLLAMA = True
OLLAMA_MODEL = "llama3.1:8b"
OLLAMA_URL = "http://localhost:11434"
LLM_TEMPERATURE = 0.2
LLM_MAX_TOKENS = 100

DEV_MODE = True
DEV_SAMPLE_SIZE = 500 if DEV_MODE else None

class LLMHardNegativeClassifier:
    """Classify and generate hard negatives using LLM"""
    
    def __init__(self, use_ollama=True, model=None, max_retries=3):
        self.use_ollama = use_ollama
        self.model = model or OLLAMA_MODEL
        self.url = OLLAMA_URL
        self.max_retries = max_retries
        self.retry_delay = 2
        
        if use_ollama:
            self._test_ollama_connection()
    
    def _test_ollama_connection(self):
        """Test Ollama connection"""
        try:
            response = requests.get(f"{self.url}/api/tags", timeout=5)
            models = response.json().get("models", [])
            model_names = [m.get("name") for m in models]
            
            if self.model in model_names:
                print(f"✓ Ollama connected - Model '{self.model}' ready")
            else:
                raise ValueError(f"Model '{self.model}' not found. Available: {model_names[:3]}")
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.url}\n"
                "Start with: ollama serve"
            )
    
    def call_ollama(self, prompt: str) -> str:
        """Call Ollama API with retry logic"""
        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.post(
                    f"{self.url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": LLM_TEMPERATURE,
                            "num_predict": LLM_MAX_TOKENS,
                        }
                    },
                    timeout=60
                )
                
                if response.status_code == 200:
                    result = response.json().get("response", "").strip()
                    if result:
                        return result
                
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay)
                    
            except Exception:
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay)
                else:
                    raise
        
        raise RuntimeError(f"Failed after {self.max_retries} attempts")
    
    def _extract_hardness_score(self, response: str) -> float:
        """Extract hardness score from LLM response"""
        response_clean = response.strip().upper()
        
        # Try numeric extraction
        try:
            score = float(response_clean)
            if score > 1.0:
                score = score / 100.0
            return min(max(score, 0.0), 1.0)
        except ValueError:
            pass
        
        # Pattern matching
        patterns = [
            (r'(\d+)\s*(?:out of|\/)\s*100', 100),
            (r'(\d+)\s*(?:out of|\/)\s*10', 10),
            (r'score[:\s]*(\d+\.?\d*)', 100),
        ]
        
        for pattern, divisor in patterns:
            match = re.search(pattern, response_clean, re.IGNORECASE)
            if match:
                try:
                    score = float(match.group(1)) / divisor
                    return min(max(score, 0.0), 1.0)
                except ValueError:
                    continue
        
        return 0.5
    
    def classify_negative(self, query: str, gold_passage: str, negative_passage: str) -> Dict:
        """Classify negative with hardness score"""
        query = query[:200]
        gold_passage = gold_passage[:300]
        negative_passage = negative_passage[:300]
        
        prompt = f"""Rate the hardness of this negative passage for training (0-100).

Query: {query}

Gold Passage: {gold_passage}

Negative Passage: {negative_passage}

HARD negatives (70-100):
- Topically related but factually different
- Similar keywords but wrong answer
- Requires understanding to distinguish

EASY negatives (0-30):
- Clearly unrelated topic
- Different domain/keywords

Provide ONLY a number from 0-100:"""
        
        try:
            response = self.call_ollama(prompt)
            hardness_score = self._extract_hardness_score(response)
            is_hard = hardness_score > 0.5
            
            return {
                "is_hard": is_hard,
                "hardness_score": hardness_score,
                "classification": "HARD" if is_hard else "EASY",
                "success": True
            }
        except Exception as e:
            return {
                "is_hard": None,
                "hardness_score": 0.5,
                "classification": "UNKNOWN",
                "success": False
            }
    
    def generate_negatives(self, query: str, positive_passage: str, num_negatives: int = 3) -> List[str]:
        """Generate hard negatives using LLM"""
        prompt = f"""Generate {num_negatives} HARD NEGATIVE passages for this query-passage pair.

Query: {query}

Correct Passage: {positive_passage[:200]}

Requirements:
1. Topically related to query
2. Similar keywords as correct passage
3. BUT does NOT answer the query
4. Each 50-150 words
5. Diverse hard negatives

Output ONLY passages, numbered 1., 2., 3.:"""
        
        try:
            response = self.call_ollama(prompt)
            negatives = self._parse_response(response, num_negatives)
            return negatives
        except Exception as e:
            return []
    
    def _parse_response(self, text: str, num_negatives: int) -> List[str]:
        """Parse LLM response to extract generated negatives"""
        negatives = []
        lines = text.split('\n')
        current = []
        
        for line in lines:
            line = line.strip()
            if line and any(line.startswith(f"{i}.") for i in range(1, 10)):
                if current:
                    negatives.append(' '.join(current))
                    current = []
                line = line.split('.', 1)[1].strip() if '.' in line else line
            if line:
                current.append(line)
        
        if current:
            negatives.append(' '.join(current))
        
        return negatives[:num_negatives] if negatives else []

def check_prerequisites():
    """Check if required files exist"""
    print("\n" + "="*60)
    print("CHECKING PREREQUISITES")
    print("="*60)
    
    msmarco_csv = os.path.join(TRAINING_DATA_DIR, "msmarco", "train.csv")
    
    if not os.path.exists(msmarco_csv):
        print(f"\n❌ Missing: {msmarco_csv}")
        print("\nRun BM25 preprocessing first:")
        print("  python data_processing/prepare_bm25_negatives.py")
        return False
    
    print(f"✓ Training data found: {msmarco_csv}")
    
    # Test Ollama
    if USE_OLLAMA:
        try:
            response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
            if response.status_code == 200:
                print(f"✓ Ollama is running at {OLLAMA_URL}")
            else:
                print("❌ Ollama not responding")
                return False
        except:
            print(f"❌ Cannot connect to Ollama")
            print("Start with: ollama serve")
            return False
    
    return True

def classify_negatives(train_df, classifier):
    """Classify negatives using LLM"""
    print("\n" + "="*60)
    print("STEP 1: LLM CLASSIFICATION")
    print("="*60)
    
    classified_dir = os.path.join(OUTPUT_DIR, "llm_classified_msmarco")
    os.makedirs(classified_dir, exist_ok=True)
    
    output_file = os.path.join(classified_dir, "llm_classified_msmarco_train.jsonl")
    
    # Clear previous
    if os.path.exists(output_file):
        os.remove(output_file)
    
    classified = []
    checkpoint_data = []
    checkpoint_interval = 100
    
    print(f"\nClassifying {len(train_df)} samples...")
    
    for idx, (_, row) in enumerate(tqdm(train_df.iterrows(), total=len(train_df), desc="Classifying"), 1):
        result = classifier.classify_negative(
            row.get('query_text', ''),
            row.get('gold_passage', ''),
            row.get('hard_negative', '')
        )
        
        classified_item = {
            'query_text': row['query_text'],
            'gold_passage': row['gold_passage'],
            'hard_negative': row['hard_negative'],
            'llm_classification': result['classification'],
            'is_hard': result['is_hard'],
            'hardness_score': result['hardness_score']
        }
        
        classified.append(classified_item)
        checkpoint_data.append(classified_item)
        
        if idx % checkpoint_interval == 0:
            with open(output_file, 'a') as f:
                for item in checkpoint_data:
                    f.write(json.dumps(item) + '\n')
            checkpoint_data = []
    
    # Save remaining
    if checkpoint_data:
        with open(output_file, 'a') as f:
            for item in checkpoint_data:
                f.write(json.dumps(item) + '\n')
    
    classified_df = pd.DataFrame(classified)
    csv_file = os.path.join(classified_dir, "llm_classified_msmarco_train.csv")
    classified_df.to_csv(csv_file, index=False)
    
    print(f"\n✓ Classification complete:")
    print(f"  Total: {len(classified_df):,}")
    print(f"  HARD: {classified_df['is_hard'].sum():,}")
    print(f"  EASY: {(~classified_df['is_hard']).sum():,}")
    
    return classified_df

def generate_additional_negatives(train_df, classifier, num_per_query=3):
    """Generate additional hard negatives using LLM"""
    print("\n" + "="*60)
    print("STEP 2: LLM GENERATION")
    print("="*60)
    
    generated_dir = os.path.join(OUTPUT_DIR, "llm_generated_negatives")
    os.makedirs(generated_dir, exist_ok=True)
    
    output_file = os.path.join(generated_dir, "generated_negatives.jsonl")
    
    generated_data = []
    checkpoint_data = []
    checkpoint_interval = 100
    failed_count = 0
    
    sample_size = min(DEV_SAMPLE_SIZE, len(train_df)) if DEV_SAMPLE_SIZE else len(train_df)
    
    print(f"\nGenerating {num_per_query} negatives per query ({sample_size} queries)...")
    
    for idx, (_, row) in enumerate(tqdm(train_df.head(sample_size).iterrows(), total=sample_size, desc="Generating"), 1):
        query = row['query_text']
        positive = row['gold_passage']
        
        hard_negatives = classifier.generate_negatives(query, positive, num_negatives=num_per_query)
        
        if hard_negatives:
            for i, neg in enumerate(hard_negatives, 1):
                item = {
                    'query_text': query,
                    'gold_passage': positive,
                    'hard_negative': neg,
                    'negative_num': i
                }
                generated_data.append(item)
                checkpoint_data.append(item)
        else:
            failed_count += 1
        
        if idx % checkpoint_interval == 0:
            with open(output_file, 'a') as f:
                for item in checkpoint_data:
                    f.write(json.dumps(item) + '\n')
            checkpoint_data = []
    
    # Save remaining
    if checkpoint_data:
        with open(output_file, 'a') as f:
            for item in checkpoint_data:
                f.write(json.dumps(item) + '\n')
    
    generated_df = pd.DataFrame(generated_data)
    csv_file = os.path.join(generated_dir, "generated_negatives.csv")
    generated_df.to_csv(csv_file, index=False)
    
    print(f"\n✓ Generation complete:")
    print(f"  Triplets: {len(generated_df):,}")
    print(f"  Queries: {sample_size}")
    print(f"  Failed: {failed_count}")
    
    return generated_df

def create_final_dataset(classified_df, generated_df):
    """Create final training dataset"""
    print("\n" + "="*60)
    print("STEP 3: FINAL DATASET")
    print("="*60)
    
    # Filter hard negatives from classification
    hard_negatives_df = classified_df[classified_df['is_hard'] == True].copy()
    train_columns = ['query_text', 'gold_passage', 'hard_negative']
    hard_negatives_df = hard_negatives_df[train_columns]
    
    print(f"  From classification: {len(hard_negatives_df):,}")
    print(f"  From generation: {len(generated_df):,}")
    
    # Combine
    final_df = pd.concat([hard_negatives_df, generated_df[train_columns]], ignore_index=True)
    
    # Save
    final_dir = os.path.join(OUTPUT_DIR, "llm_final_training")
    os.makedirs(final_dir, exist_ok=True)
    
    final_csv = os.path.join(final_dir, "hard_negatives_final.csv")
    final_df.to_csv(final_csv, index=False)
    
    final_pkl = os.path.join(final_dir, "hard_negatives_final.pkl")
    with open(final_pkl, 'wb') as f:
        pickle.dump(final_df, f)
    
    print(f"\n✓ Final dataset saved:")
    print(f"  Total: {len(final_df):,} triplets")
    print(f"  Location: {final_csv}")
    
    return final_df

def main():
    print("="*60)
    print("LLM CLASSIFICATION & GENERATION")
    print("="*60)
    
    if not check_prerequisites():
        return False
    
    # Load training data
    msmarco_csv = os.path.join(TRAINING_DATA_DIR, "msmarco", "train.csv")
    print(f"\nLoading: {msmarco_csv}")
    train_df = pd.read_csv(msmarco_csv)
    
    if DEV_MODE and DEV_SAMPLE_SIZE:
        train_df = train_df.head(DEV_SAMPLE_SIZE)
        print(f"✓ DEV MODE: Using {len(train_df)} samples")
    else:
        print(f"✓ Loaded {len(train_df):,} samples")
    
    # Initialize classifier
    print("\n" + "="*60)
    print("INITIALIZING LLM")
    print("="*60)
    
    try:
        classifier = LLMHardNegativeClassifier(
            use_ollama=USE_OLLAMA,
            model=OLLAMA_MODEL,
            max_retries=3
        )
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        return False
    
    # Step 1: Classify existing negatives
    classified_df = classify_negatives(train_df, classifier)
    
    # Step 2: Generate additional negatives
    generated_df = generate_additional_negatives(train_df, classifier, num_per_query=3)
    
    # Step 3: Create final dataset
    final_df = create_final_dataset(classified_df, generated_df)
    
    print("\n" + "="*60)
    print("✅ LLM PROCESSING COMPLETE")
    print("="*60)
    print(f"\nData saved in: {OUTPUT_DIR}/")
    print(f"\nNext: python data_processing/rag_selection.py")
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
