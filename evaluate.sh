#!/bin/bash

# Model Evaluation Script
# Evaluates all trained models on MS MARCO dev set

set -e  # Exit on any error

echo "======================================"
echo "MODEL EVALUATION"
echo "======================================"

# Check if evaluation data exists
EVAL_DATA="data/msmarco_dev_beir.tsv"
if [ ! -f "$EVAL_DATA" ]; then
    echo "❌ Evaluation data not found: $EVAL_DATA"
    echo ""
    echo "Create evaluation data by running notebook cell that creates BEIR format dev set"
    echo "(Implementation.ipynb cell 17)"
    exit 1
fi

echo "✓ Evaluation data found: $EVAL_DATA"
echo ""

# Check if any models exist
MODEL_DIR="./models"
if [ ! -d "$MODEL_DIR" ] || [ -z "$(ls -A $MODEL_DIR 2>/dev/null)" ]; then
    echo "❌ No models found in $MODEL_DIR"
    echo ""
    echo "Train models first with: bash train.sh"
    exit 1
fi

echo "Models to evaluate:"
for model in "$MODEL_DIR"/dpr_*; do
    if [ -d "$model" ]; then
        echo "  - $(basename $model)"
    fi
done

echo ""
read -p "Press Enter to start evaluation..."

# Run evaluation
echo ""
echo "Running comprehensive evaluation..."
echo "-----------------------------------"
python evaluation_scripts/evaluate_models.py

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Evaluation failed"
    exit 1
fi

echo ""
echo "======================================"
echo "✅ EVALUATION COMPLETE"
echo "======================================"
echo ""
echo "Results saved in:"
echo "  - models/phase5_complete_results.json"