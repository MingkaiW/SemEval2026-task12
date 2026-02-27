#!/bin/bash

# Example script to demonstrate caching functionality with scikit-learn classifiers

echo "=========================================="
echo "Example 1: First run (creates cache and compares all classifiers)"
echo "=========================================="
python baselines/ml_mlp_baseline.py \
    --train_questions sample_data/questions.jsonl \
    --train_docs sample_data/docs.json \
    --batch_size 4

echo ""
echo "=========================================="
echo "Example 2: Second run (loads from cache - much faster!)"
echo "=========================================="
python baselines/ml_mlp_baseline.py \
    --train_questions sample_data/questions.jsonl \
    --train_docs sample_data/docs.json \
    --batch_size 4 \
    --classifiers "Logistic Regression" "Random Forest"

echo ""
echo "=========================================="
echo "Example 3: With image descriptions (BLIP2)"
echo "=========================================="
python baselines/ml_mlp_baseline.py \
    --train_questions sample_data/questions.jsonl \
    --train_docs sample_data/docs.json \
    --use_image_descriptions \
    --img2txt_model blip2 \
    --blip2_version opt \
    --batch_size 4 \
    --classifiers "Logistic Regression"

echo ""
echo "=========================================="
echo "Example 4: Custom validation split"
echo "=========================================="
python baselines/ml_mlp_baseline.py \
    --train_questions sample_data/questions.jsonl \
    --train_docs sample_data/docs.json \
    --val_split 0.3 \
    --classifiers "MLP"

echo ""
echo "=========================================="
echo "Cache contents:"
echo "=========================================="
find ./cache/original_embeddings -type f -exec ls -lh {} \;
