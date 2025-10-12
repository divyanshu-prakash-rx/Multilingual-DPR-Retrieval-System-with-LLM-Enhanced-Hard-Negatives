#!/usr/bin/env python3
"""
Batch Evaluation Script - Evaluate all models on multiple datasets
"""

import subprocess
import sys
from pathlib import Path

def run_batch_evaluation():
    """Run evaluation on all trained models"""

    models = [
        ("ANCE", 0),
        ("DPR-Base", 1), 
        ("DPR-BM25", 2),
        ("ICT-Passage", 3),
        ("ICT-Query", 4),
        ("TAS-Passage", 5),
        ("TAS-Query", 6)
    ]

    datasets = [
        ("beir", 0, "SciFact"),
        ("beir", 1, "ArguAna"),
        # Add more datasets as needed
    ]

    print("🚀 Starting Batch Evaluation...")
    print(f"Models: {len(models)}, Datasets: {len(datasets)}")

    for dataset_collection, dataset_no, dataset_name in datasets:
        print(f"\n📊 Evaluating on {dataset_name}...")

        for model_name, model_no in models:
            print(f"   🔍 Testing {model_name}...")

            try:
                result = subprocess.run([
                    sys.executable, "evaluation_adapter.py", 
                    dataset_collection, str(dataset_no), str(model_no)
                ], capture_output=True, text=True, timeout=300)

                if result.returncode == 0:
                    print(f"   ✅ {model_name}: Success")
                else:
                    print(f"   ❌ {model_name}: Failed")

            except subprocess.TimeoutExpired:
                print(f"   ⏰ {model_name}: Timeout")
            except Exception as e:
                print(f"   ❌ {model_name}: Error - {e}")

    print("\n🎉 Batch evaluation complete!")

if __name__ == "__main__":
    run_batch_evaluation()
