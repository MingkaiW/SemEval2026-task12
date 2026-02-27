#!/bin/bash

# Quick test with only Logistic Regression (fastest classifier)
# Useful for quick iterations and testing

echo "=========================================="
echo "Quick Test: Logistic Regression Only"
echo "Train → Dev"
echo "=========================================="

python baselines/ml_mlp_baseline.py \
    --train_questions train_data/questions.jsonl \
    --train_docs train_data/docs.json \
    --dev_questions dev_data/questions.jsonl \
    --dev_docs dev_data/docs.json \
    --use_dev \
    --classifiers "Logistic Regression" \
    --save_predictions \
    --output_dir ./results/quick_test \
    --batch_size 8

echo ""
echo "=========================================="
echo "Quick test complete!"
echo "=========================================="
