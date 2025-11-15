#!/bin/bash

# Multilingual DPR Training Pipeline
# Trains models progressively through all phases with validation

set -e  # Exit on any error

echo "======================================"
echo "MULTILINGUAL DPR TRAINING PIPELINE"
echo "======================================"

# Phase 1: Preprocess and train BM25 baseline
echo ""
echo "==================================================="
echo "PHASE 1: BM25 PREPROCESSING"
echo "==================================================="
python data_processing/prepare_bm25_negatives.py
if [ $? -ne 0 ]; then
    echo "❌ BM25 preprocessing failed!"
    exit 1
fi

echo ""
echo "==================================================="
echo "PHASE 2: BASELINE TRAINING (BM25 Negatives)"
echo "==================================================="
python train_scripts/train_phase1_bm25_baseline.py

# Phase 2: LLM Classification and Training
echo ""
echo "Phase 2: LLM-Enhanced Model"
echo "---------------------------"
echo "==================================================="
echo "PHASE 3: LLM CLASSIFICATION"
echo "==================================================="
echo "⚠ Ollama must be running: ollama serve"
read -p "Is Ollama running? (y/n): " ollama_ready

if [[ ! "$ollama_ready" =~ ^[Yy]$ ]]; then
    echo "❌ Start Ollama first: ollama serve"
    exit 1
fi

python data_processing/llm_classification.py
if [ $? -ne 0 ]; then
    echo "❌ Phase 3 preprocessing failed"
    exit 1
fi

python train_scripts/train_phase3_llm_enhanced.py
if [ $? -ne 0 ]; then
    echo "❌ Phase 3 training failed"
    exit 1
fi

# Phase 3: RAG Selection and Training
echo ""
echo "==================================================="
echo "PHASE 5: RAG SELECTION"
echo "==================================================="
echo "⚠ This builds FAISS index and scores negatives"
read -p "Run RAG selection now? (y/n): " rag_ready

if [[ ! "$rag_ready" =~ ^[Yy]$ ]]; then
    echo "⚠ Skipping Phase 4 (RAG Enhanced Training)"
    echo "Run later: python data_processing/rag_selection.py"
else
    python data_processing/rag_selection.py
    if [ $? -ne 0 ]; then
        echo "❌ RAG selection failed!"
        exit 1
    fi
    
    echo ""
    echo "==================================================="
    echo "PHASE 6: RAG ENHANCED TRAINING"
    echo "==================================================="
    python train_scripts/train_phase4_rag_enhanced.py
    if [ $? -ne 0 ]; then
        echo "❌ Phase 4 training failed"
        exit 1
    fi
fi

# Phase 5: Multilingual Fine-tuning
echo ""
echo "==================================================="
echo "PHASE 7: MULTILINGUAL FINE-TUNING"
echo "==================================================="
python train_scripts/train_phase5_multilingual_finetune.py
if [ $? -ne 0 ]; then
    echo "❌ Phase 5 fine-tuning failed"
    exit 1
fi

echo ""
echo "======================================"
echo "✅ ALL TRAINING PHASES COMPLETE"
echo "======================================"
echo ""
echo "Trained models saved in ./models/"
echo "Run evaluation with: bash evaluate.sh"
