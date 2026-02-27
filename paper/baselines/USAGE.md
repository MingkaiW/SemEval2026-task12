# Usage Guide: Running on Train/Dev/Test Splits

This guide shows how to run the baseline on all data splits (train_data, dev_data, test_data).

## Quick Start

### Option 1: Run on All Splits (Recommended)

Train on `train_data`, evaluate on both `dev_data` and `test_data`:

```bash
./baselines/run_all_splits.sh
```

This will:
- ✅ Train classifiers on train_data
- ✅ Use dev_data as validation set
- ✅ Evaluate best model on test_data
- ✅ Save all predictions and metrics to `./results/`

### Option 2: Quick Test (Logistic Regression only)

For quick iterations and testing:

```bash
./baselines/run_quick_test.sh
```

Uses only Logistic Regression (fastest classifier) for rapid feedback.

## Manual Usage

### Train on train_data, evaluate on dev_data

```bash
python baselines/ml_mlp_baseline.py \
    --use_dev \
    --save_predictions
```

### Train on train_data, evaluate on dev_data AND test_data

```bash
python baselines/ml_mlp_baseline.py \
    --use_dev \
    --use_test \
    --save_predictions
```

### Custom paths

```bash
python baselines/ml_mlp_baseline.py \
    --train_questions path/to/train/questions.jsonl \
    --train_docs path/to/train/docs.json \
    --dev_questions path/to/dev/questions.jsonl \
    --dev_docs path/to/dev/docs.json \
    --test_questions path/to/test/questions.jsonl \
    --test_docs path/to/test/docs.json \
    --use_dev \
    --use_test \
    --save_predictions
```

## Workflow Details

### Default Behavior (No --use_dev/--use_test)
- Loads train_data only
- Splits train_data into 80% train / 20% validation
- Trains all classifiers
- Compares results

### With --use_dev
- Loads train_data + dev_data
- Uses **entire** train_data for training
- Uses dev_data as validation set
- Evaluates best model on dev_data

### With --use_dev --use_test
- Loads train_data + dev_data + test_data
- Trains on train_data
- Uses dev_data as validation set
- Evaluates best model on both dev_data and test_data

## Output Structure

When using `--save_predictions`, results are saved to `--output_dir` (default: `./results/`):

```
results/
├── dev_<best_classifier>_metrics.json        # Dev set metrics
├── dev_<best_classifier>_predictions.npy     # Dev set predictions
├── dev_<best_classifier>_labels.npy          # Dev set true labels
├── test_<best_classifier>_metrics.json       # Test set metrics
├── test_<best_classifier>_predictions.npy    # Test set predictions
├── test_<best_classifier>_labels.npy         # Test set true labels
└── all_classifiers_summary.json              # Summary of all classifiers
```

### Metrics File Format

```json
{
  "hamming_loss": 0.1234,
  "accuracy": 0.4567,
  "f1_micro": 0.7890,
  "f1_macro": 0.7654,
  "f1_samples": 0.7543,
  "precision_micro": 0.8012,
  "recall_micro": 0.7678
}
```

### Predictions Format

Predictions are saved as NumPy arrays:
- Shape: `(n_samples, 4)` for 4 labels (A, B, C, D)
- Values: Probability scores (0.0 to 1.0)
- Load with: `np.load('predictions.npy')`

## Common Scenarios

### 1. Development & Hyperparameter Tuning

Use dev set for quick iterations:

```bash
python baselines/ml_mlp_baseline.py \
    --use_dev \
    --classifiers "Logistic Regression" \
    --hidden_dim 1024
```

### 2. Final Evaluation

Train on full train_data, evaluate on test_data:

```bash
python baselines/ml_mlp_baseline.py \
    --use_dev \
    --use_test \
    --save_predictions \
    --output_dir ./final_results
```

### 3. Specific Classifiers Only

```bash
python baselines/ml_mlp_baseline.py \
    --use_dev \
    --use_test \
    --classifiers "Random Forest" "MLP" \
    --save_predictions
```

### 4. With Image Descriptions

```bash
python baselines/ml_mlp_baseline.py \
    --use_dev \
    --use_test \
    --use_image_descriptions \
    --img2txt_model blip2 \
    --blip2_version opt \
    --save_predictions
```

## Performance Tips

### Speed Up Training

1. **Use fewer classifiers**:
   ```bash
   --classifiers "Logistic Regression"
   ```

2. **Increase batch size** (if you have enough memory):
   ```bash
   --batch_size 16
   ```

3. **Use cached embeddings** (default behavior):
   - First run computes embeddings
   - Subsequent runs load from cache (much faster!)

### Improve Accuracy

1. **Use all classifiers** (default):
   - Compares all 4 approaches automatically

2. **Increase MLP hidden dimensions**:
   ```bash
   --hidden_dim 1024
   ```

3. **Add image descriptions**:
   ```bash
   --use_image_descriptions --img2txt_model blip2
   ```

## Troubleshooting

### "FileNotFoundError: dev_data/questions.jsonl"

Make sure you have the dev_data folder with the required files:
- `dev_data/questions.jsonl`
- `dev_data/docs.json`

Or specify custom paths with `--dev_questions` and `--dev_docs`.

### "Out of memory" during embedding computation

Reduce batch size:
```bash
--batch_size 4
```

### Clear cache to regenerate embeddings

```bash
rm -rf ./cache/original_embeddings/
```

Then run again.

## Example Workflows

### Complete Baseline Run

```bash
# 1. Quick test first (Logistic Regression only)
./baselines/run_quick_test.sh

# 2. If results look good, run all classifiers
./baselines/run_all_splits.sh

# 3. Check results
cat results/all_classifiers_summary.json
```

### Custom Experiment

```bash
# Compare specific classifiers with custom settings
python baselines/ml_mlp_baseline.py \
    --use_dev \
    --use_test \
    --classifiers "Random Forest" "MLP" "Gradient Boosting" \
    --hidden_dim 1024 \
    --batch_size 16 \
    --save_predictions \
    --output_dir ./experiments/exp1

# View results
cat ./experiments/exp1/all_classifiers_summary.json
```

## Reading Results in Python

```python
import json
import numpy as np

# Load metrics
with open('results/dev_MLP_metrics.json') as f:
    metrics = json.load(f)
print(f"Dev F1-Micro: {metrics['f1_micro']:.4f}")

# Load predictions
predictions = np.load('results/dev_MLP_predictions.npy')
true_labels = np.load('results/dev_MLP_labels.npy')

# Apply threshold
pred_binary = (predictions >= 0.5).astype(int)

# Analyze per-label performance
from sklearn.metrics import classification_report
labels = ['A', 'B', 'C', 'D']
print(classification_report(true_labels, pred_binary, target_names=labels))
```

## Advanced: Batch Processing

Run experiments with different configurations:

```bash
#!/bin/bash

# Try different classifiers
for clf in "Logistic Regression" "Random Forest" "MLP"; do
    python baselines/ml_mlp_baseline.py \
        --use_dev --use_test \
        --classifiers "$clf" \
        --save_predictions \
        --output_dir "./experiments/${clf// /_}"
done
```

## Data Split Summary

| Split | Purpose | Size (approx) | Usage |
|-------|---------|---------------|-------|
| train_data | Training | ~70-80% | Train classifiers |
| dev_data | Validation | ~10-15% | Model selection, hyperparameter tuning |
| test_data | Testing | ~10-15% | Final evaluation |

**Best Practice:**
- Use dev_data during development
- Only check test_data for final evaluation
- Don't tune on test_data to avoid overfitting!
