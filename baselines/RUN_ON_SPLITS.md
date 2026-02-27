# Running on Train/Dev/Test Splits - Quick Reference

## 🚀 Quick Start

### Run on All Splits (One Command)

```bash
./baselines/run_all_splits.sh
```

This trains on `train_data` and evaluates on both `dev_data` and `test_data`.

## 📋 What Gets Evaluated

### Automatic Process

1. **Load train_data** → Compute/cache embeddings
2. **Load dev_data** → Compute/cache embeddings
3. **Load test_data** → Compute/cache embeddings
4. **Train all 4 classifiers** on train_data
5. **Use dev_data** as validation set
6. **Select best classifier** based on dev performance
7. **Evaluate on test_data** with best classifier
8. **Save all results** to `./results/`

### Output Example

```
================================================================================
STEP 1: Loading Training Data
================================================================================
Processing train split...
Loading cached embeddings from cache/original_embeddings/text/xxx.pt
train - Feature shape: (1000, 1536), Labels shape: (1000, 4)

================================================================================
STEP 2: Loading Dev Data
================================================================================
Processing dev split...
Loading cached embeddings from cache/original_embeddings/text/yyy.pt
dev - Feature shape: (200, 1536), Labels shape: (200, 4)

================================================================================
STEP 3: Loading Test Data
================================================================================
Processing test split...
Loading cached embeddings from cache/original_embeddings/text/zzz.pt
test - Feature shape: (200, 1536), Labels shape: (200, 4)

================================================================================
STEP 4: Preparing Training/Validation Split
================================================================================
Using dev set as validation set
Training samples: 1000
Validation samples: 200

================================================================================
STEP 5: Training and Evaluating Classifiers
================================================================================

============================================================
Training Logistic Regression...
============================================================
[Training output...]

============================================================
Training Random Forest...
============================================================
[Training output...]

[... MLP, Gradient Boosting ...]

================================================================================
CLASSIFIER COMPARISON
================================================================================

Classifier               F1-Micro     F1-Macro     Accuracy     Hamming Loss
--------------------------------------------------------------------------------
Logistic Regression      0.7654       0.7321       0.4321       0.1345
Random Forest            0.8123       0.7890       0.5012       0.1123
MLP                      0.8456       0.8234       0.5678       0.0987
Gradient Boosting        0.8234       0.8012       0.5234       0.1056
================================================================================

Best classifier based on f1_micro: MLP (score: 0.8456)

================================================================================
STEP 6: Evaluating on Test Set
================================================================================

Test Set Metrics (MLP):
  hamming_loss: 0.1023
  accuracy: 0.5543
  f1_micro: 0.8321
  f1_macro: 0.8134
  ...

Saved metrics to results/test_MLP_metrics.json
Saved predictions to results/test_MLP_predictions.npy
Saved true labels to results/test_MLP_labels.npy

================================================================================
Final Dev Set Evaluation
================================================================================

Dev Set Metrics (MLP):
  hamming_loss: 0.0987
  accuracy: 0.5678
  f1_micro: 0.8456
  f1_macro: 0.8234
  ...

Saved metrics to results/dev_MLP_metrics.json

================================================================================
Training and Evaluation Complete!
================================================================================
```

## 📁 Results Structure

After running, check `./results/`:

```
results/
├── dev_MLP_metrics.json              # Dev metrics (best model)
├── dev_MLP_predictions.npy           # Dev predictions
├── dev_MLP_labels.npy                # Dev true labels
├── test_MLP_metrics.json             # Test metrics (best model)
├── test_MLP_predictions.npy          # Test predictions
├── test_MLP_labels.npy               # Test true labels
└── all_classifiers_summary.json      # All classifiers comparison
```

## 🎯 Common Commands

### Basic: Train + Dev + Test

```bash
python baselines/ml_mlp_baseline.py \
    --use_dev \
    --use_test \
    --save_predictions
```

### Fast: Only Logistic Regression

```bash
python baselines/ml_mlp_baseline.py \
    --use_dev \
    --use_test \
    --classifiers "Logistic Regression" \
    --save_predictions
```

### Best Performance: All Classifiers

```bash
python baselines/ml_mlp_baseline.py \
    --use_dev \
    --use_test \
    --save_predictions
```

### With Image Descriptions

```bash
python baselines/ml_mlp_baseline.py \
    --use_dev \
    --use_test \
    --use_image_descriptions \
    --img2txt_model blip2 \
    --blip2_version opt \
    --save_predictions
```

## 🔍 Check Results

### View Metrics

```bash
# See all classifier comparisons
cat results/all_classifiers_summary.json

# See test set performance
cat results/test_MLP_metrics.json
```

### Load in Python

```python
import json
import numpy as np

# Load test metrics
with open('results/test_MLP_metrics.json') as f:
    metrics = json.load(f)

print(f"Test F1-Micro: {metrics['f1_micro']:.4f}")
print(f"Test F1-Macro: {metrics['f1_macro']:.4f}")
print(f"Test Accuracy: {metrics['accuracy']:.4f}")

# Load predictions
predictions = np.load('results/test_MLP_predictions.npy')
labels = np.load('results/test_MLP_labels.npy')

print(f"Predictions shape: {predictions.shape}")  # (n_samples, 4)
print(f"Labels shape: {labels.shape}")            # (n_samples, 4)
```

## ⚡ Performance

### Caching Benefits

**First run:**
- Computes embeddings for train/dev/test (~5-15 min)
- Trains classifiers (~5-15 min)
- **Total: ~10-30 min**

**Subsequent runs:**
- Loads cached embeddings (~10-30 sec)
- Trains classifiers (~5-15 min)
- **Total: ~5-15 min** ⚡

### Speed Tips

1. **Quick test**: Use `--classifiers "Logistic Regression"`
2. **Increase batch size**: Use `--batch_size 16` (if memory allows)
3. **Skip test**: Don't use `--use_test` during development

## 🎓 Best Practices

### Development Phase
```bash
# Use dev set only for quick iterations
python baselines/ml_mlp_baseline.py \
    --use_dev \
    --classifiers "Logistic Regression" "Random Forest"
```

### Final Evaluation
```bash
# Run on test set only after development is complete
./baselines/run_all_splits.sh
```

### Avoid Overfitting
- ✅ Use dev set for model selection
- ✅ Use test set only for final evaluation
- ❌ Don't tune hyperparameters on test set!

## 🔧 Troubleshooting

### Missing data files?
```bash
# Check if files exist
ls -lh train_data/ dev_data/ test_data/
```

### Clear cache?
```bash
# Delete all cached embeddings
rm -rf ./cache/original_embeddings/

# Run again to recompute
./baselines/run_all_splits.sh
```

### Out of memory?
```bash
# Reduce batch size
python baselines/ml_mlp_baseline.py \
    --use_dev --use_test \
    --batch_size 4
```

## 📊 Expected Results

Typical performance ranges (may vary by dataset):

| Metric | Range | Description |
|--------|-------|-------------|
| F1-Micro | 0.70-0.90 | Overall performance |
| F1-Macro | 0.65-0.85 | Average across labels |
| Accuracy | 0.40-0.70 | Exact match (strict) |
| Hamming Loss | 0.05-0.20 | Label-wise error rate |

**Best classifier is typically:**
- MLP or Gradient Boosting for accuracy
- Logistic Regression for speed
- Random Forest for balanced performance

For detailed usage, see [USAGE.md](USAGE.md)
