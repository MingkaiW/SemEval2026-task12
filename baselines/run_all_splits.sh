#!/bin/bash

# Run baseline on train, dev, and test splits
# This script trains on train_data and evaluates on dev_data and test_data

echo "=========================================="
echo "Running Multimodal Multi-Label Baseline"
echo "Train → Dev → Test"
echo "=========================================="

python baselines/ml_mlp_baseline.py \
    --train_questions train_data/questions.jsonl \
    --train_docs train_data/docs.json \
    --dev_questions dev_data/questions.jsonl \
    --dev_docs dev_data/docs.json \
    --test_questions test_data/questions.jsonl \
    --test_docs test_data/docs.json \
    --use_dev \
    --use_test \
    --save_predictions \
    --output_dir ./results \
    --batch_size 8

echo ""
echo "=========================================="
echo "Complete! Results saved to ./results/"
echo "=========================================="
echo ""
echo "Files created:"
find ./results -type f | sort
