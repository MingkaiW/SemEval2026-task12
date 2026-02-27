# Changes from CuPy to Scikit-learn

## Summary

The baseline has been completely refactored from a CuPy-based custom neural network to a scikit-learn based multi-classifier comparison framework.

## Major Changes

### 1. **Replaced CuPy with Scikit-learn**

**Before:**
- Custom MLP implementation using CuPy
- Manual gradient descent
- Single model training
- Required CUDA/GPU

**After:**
- Multiple scikit-learn classifiers
- Built-in optimization algorithms
- Automatic hyperparameter handling
- CPU-based (no CUDA required)

### 2. **Multiple Classifier Comparison**

The baseline now automatically trains and compares 4 different classifiers:

| Classifier | Type | Speed | Best For |
|------------|------|-------|----------|
| Logistic Regression | Linear | Very Fast | Quick baseline |
| Random Forest | Ensemble | Fast | Non-linear patterns |
| MLP | Neural Network | Moderate | Complex patterns |
| Gradient Boosting | Ensemble | Slow | Best accuracy |

### 3. **Comprehensive Metrics**

**Before:**
- Only binary cross-entropy loss

**After:**
- Hamming Loss
- Accuracy (exact match)
- F1-Score (micro, macro, samples)
- Precision & Recall (micro)

### 4. **Improved Workflow**

**Before:**
```
Load Data → Compute Embeddings → Train for N epochs → Done
```

**After:**
```
Load/Cache Embeddings → Split Data → Train All Classifiers → Compare → Best Model
```

### 5. **Removed Dependencies**

**Removed:**
- CuPy
- CUDA requirement

**Added:**
- scikit-learn (standard ML library)

## Code Architecture Changes

### File Structure

```
baselines/
├── ml_mlp_baseline.py       # Main script (refactored)
├── README.md                 # Updated documentation
├── test_cache.sh            # Updated test script
└── CHANGES.md               # This file
```

### Key Classes

#### Removed
- `CuPyMultiLabelClassifier` - Custom neural network

#### Added
- `MultiLabelClassifierManager` - Manages multiple classifiers
  - `get_classifiers()` - Returns dictionary of classifiers
  - `train_and_evaluate()` - Trains all classifiers
  - `evaluate()` - Computes metrics
  - `print_comparison()` - Shows comparison table
  - `get_best_classifier()` - Returns best performer

#### Kept (with modifications)
- `VisualDescriptionGenerator` - Image-to-text (unchanged)
- `EmbeddingCache` - Caching system (unchanged)
- `MultimodalEmbedder` - Embedding generation (unchanged)
- `AERDataset` - Dataset loading (unchanged)
- `CachedEmbeddingDataset` - Cached embeddings (unchanged)

## API Changes

### Command-Line Arguments

#### Removed
```bash
--epochs          # No longer needed (sklearn handles iterations)
--learning_rate   # Handled internally by each classifier
```

#### Added
```bash
--val_split       # Validation split ratio (default: 0.2)
--classifiers     # Select specific classifiers to train
```

#### Modified
```bash
--batch_size      # Now only for embedding computation
--hidden_dim      # Only affects MLP classifier
```

### Usage Changes

**Before (CuPy):**
```bash
python baselines/ml_mlp_baseline.py \
    --epochs 10 \
    --learning_rate 0.01 \
    --batch_size 8
```

**After (Scikit-learn):**
```bash
# Train all classifiers
python baselines/ml_mlp_baseline.py

# Train specific classifiers
python baselines/ml_mlp_baseline.py \
    --classifiers "Logistic Regression" "MLP"

# Custom validation split
python baselines/ml_mlp_baseline.py \
    --val_split 0.3
```

## Performance Comparison

### Training Speed

| Aspect | CuPy | Scikit-learn |
|--------|------|--------------|
| Embedding | GPU (fast) | GPU (fast) |
| Training | GPU (custom) | CPU (optimized) |
| Iterations | N epochs | Automatic convergence |
| Total Time | ~10-15 min | ~5-15 min (varies by classifier) |

### Advantages of Scikit-learn

✅ **Easier to use** - No manual gradient implementation
✅ **Multiple models** - Compare different approaches
✅ **Better metrics** - Comprehensive evaluation
✅ **No CUDA needed** - Runs on any machine
✅ **Faster iteration** - Quick experimentation
✅ **Production ready** - Well-tested algorithms

### Advantages of CuPy (previous)

✅ **GPU acceleration** - Faster on large datasets
✅ **Custom architecture** - Full control over model
✅ **Memory efficient** - Direct GPU memory management

## Migration Guide

### For Existing Users

If you were using the CuPy version:

1. **Remove CuPy**
   ```bash
   pip uninstall cupy-cuda12x
   ```

2. **Install scikit-learn**
   ```bash
   pip install scikit-learn
   ```

3. **Update your scripts**
   - Remove `--epochs` and `--learning_rate` arguments
   - Add `--classifiers` to select specific models
   - Use `--val_split` instead of manual validation

4. **Cache is compatible**
   - Your existing cached embeddings will work!
   - No need to recompute

### Example Migration

**Old script:**
```bash
#!/bin/bash
for lr in 0.001 0.01 0.1; do
    python baselines/ml_mlp_baseline.py \
        --epochs 20 \
        --learning_rate $lr \
        --batch_size 8
done
```

**New script:**
```bash
#!/bin/bash
# Compare all classifiers with different validation splits
for split in 0.1 0.2 0.3; do
    python baselines/ml_mlp_baseline.py \
        --val_split $split \
        --batch_size 8
done
```

## Output Changes

### Before (CuPy)
```
Epoch 1/10 - Loss: 0.2345
Epoch 2/10 - Loss: 0.2123
...
Epoch 10/10 - Loss: 0.1567
Training complete.
```

### After (Scikit-learn)
```
================================================================================
STEP 1: Computing/Loading Embeddings
================================================================================
Loading cached embeddings...

================================================================================
STEP 2: Splitting Data
================================================================================
Training samples: 800
Validation samples: 200

================================================================================
STEP 3: Training and Evaluating Classifiers
================================================================================

Training Logistic Regression...
  f1_micro: 0.7654
  accuracy: 0.4321
  ...

Training Random Forest...
  f1_micro: 0.8123
  accuracy: 0.5012
  ...

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

Best classifier: MLP (F1-Micro: 0.8456)
```

## Backward Compatibility

### ✅ Compatible
- Cached embeddings (`.pt` files)
- Dataset format (JSONL, JSON)
- Image-to-text functionality
- Cache directory structure

### ❌ Not Compatible
- CuPy model weights (N/A - different paradigm)
- Training loop logic
- Command-line arguments for epochs/learning rate

## Future Enhancements

Possible additions:
- [ ] Cross-validation support
- [ ] Hyperparameter tuning (GridSearchCV)
- [ ] Model ensembling
- [ ] Custom metric weighting
- [ ] Test set evaluation
- [ ] Model persistence (save/load)
- [ ] Confusion matrix visualization

## Questions?

See the [README.md](README.md) for detailed documentation and usage examples.
