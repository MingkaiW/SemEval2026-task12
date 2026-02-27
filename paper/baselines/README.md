# Multimodal Multi-Label Baseline with Scikit-learn

This baseline implements a multimodal multi-label classifier using scikit-learn with multiple classification approaches and efficient embedding caching.

## Features

### 1. **Multiple Classifier Comparison**
Automatically trains and compares multiple scikit-learn classifiers:
- **Logistic Regression** - Fast linear baseline
- **Random Forest** - Ensemble of decision trees
- **MLP (Multi-Layer Perceptron)** - Neural network classifier
- **Gradient Boosting** - Gradient boosted trees

All classifiers are wrapped with `MultiOutputClassifier` for multi-label classification.

### 2. **Comprehensive Evaluation Metrics**
Each classifier is evaluated with:
- **Hamming Loss** - Fraction of wrong labels
- **Accuracy** - Exact match ratio
- **F1-Score** - Micro, Macro, and Samples averaging
- **Precision & Recall** - Micro averaging

### 3. **Embedding Caching System**
- Pre-computes text and image embeddings once
- Saves them to disk in `./cache/original_embeddings/`
- Automatically loads cached embeddings on subsequent runs
- Significantly speeds up experimentation with different classifiers

**Cache Structure:**
```
./cache/original_embeddings/
├── text/
│   └── <cache_key>.pt    # Text embeddings
└── image/
    └── <cache_key>.pt    # Image embeddings
```

### 4. **Image-to-Text Integration**
Generate visual descriptions from images using various vision-language models:
- BLIP2 (flan-t5, opt)
- LLaVA (7b, 13b, 34b)
- Qwen2-VL (2b, 7b, 72b)
- Qwen3-VL (8b, 14b, 72b)
- CLIP Interrogator
- API-based (GPT-4o)

### 5. **Multimodal Embeddings**
- **Text**: Qwen3-Embedding-0.6B
- **Image**: SigLIP (google/siglip-base-patch16-224)
- **Feature Normalization**: StandardScaler for better convergence

## Installation

```bash
pip install torch transformers pillow requests numpy scikit-learn
pip install qwen-vl-utils clip-interrogator
pip install ollama  # For LLaVA and Qwen3 models (optional)
```

## Usage

### Basic Training (Compare All Classifiers)

```bash
python baselines/ml_mlp_baseline.py \
    --train_questions train_data/questions.jsonl \
    --train_docs train_data/docs.json
```

**Output:**
- Trains all 4 classifiers
- Shows metrics for each
- Displays comparison table
- Reports best classifier

### Train Specific Classifiers Only

```bash
# Train only Random Forest and MLP
python baselines/ml_mlp_baseline.py \
    --classifiers "Random Forest" "MLP"

# Train only Logistic Regression
python baselines/ml_mlp_baseline.py \
    --classifiers "Logistic Regression"
```

### Custom Validation Split

```bash
# Use 30% for validation
python baselines/ml_mlp_baseline.py \
    --val_split 0.3
```

### With Image Descriptions

```bash
# Using BLIP2
python baselines/ml_mlp_baseline.py \
    --use_image_descriptions \
    --img2txt_model blip2 \
    --blip2_version opt

# Using Qwen2-VL (quantized)
python baselines/ml_mlp_baseline.py \
    --use_image_descriptions \
    --img2txt_model qwen2-vl \
    --qwen2vl_version 2b \
    --use_quantization

# Using API (GPT-4o)
python baselines/ml_mlp_baseline.py \
    --use_image_descriptions \
    --img2txt_model api \
    --api_key YOUR_API_KEY
```

### Custom Cache Directory

```bash
python baselines/ml_mlp_baseline.py \
    --cache_dir ./my_custom_cache
```

### Disable Caching (for debugging)

```bash
python baselines/ml_mlp_baseline.py \
    --disable_cache
```

## Command-Line Arguments

### Data Paths
- `--train_questions`: Path to training questions JSONL (default: `train_data/questions.jsonl`)
- `--train_docs`: Path to training documents JSON (default: `train_data/docs.json`)

### Training Parameters
- `--batch_size`: Batch size for embedding computation (default: 8)
- `--hidden_dim`: Hidden dimension for MLP classifier (default: 512)
- `--val_split`: Validation split ratio (default: 0.2)
- `--classifiers`: Specific classifiers to use (default: all)
  - Choices: `"Logistic Regression"`, `"Random Forest"`, `"MLP"`, `"Gradient Boosting"`

### Cache Parameters
- `--cache_dir`: Cache directory (default: `./cache/original_embeddings`)
- `--disable_cache`: Disable caching (compute embeddings on-the-fly)

### Output Parameters
- `--output_dir`: Directory for saving results (default: `./results`)
- `--save_predictions`: Save predictions to file
- `--submission_file`: Path to save submission JSONL file (e.g., `submission.jsonl`)

### Submission File Format
The `--submission_file` option generates a file in the required competition format:
```jsonl
{"id": "q-2020", "answer": "A"}
{"id": "q-2021", "answer": "B,D"}
```

Example usage:
```bash
python baselines/ml_mlp_baseline.py \
    --use_test \
    --submission_file submission.jsonl
```

### Image-to-Text Options
- `--use_image_descriptions`: Enable image description generation
- `--img2txt_model`: Model type (`blip2`, `llava`, `qwen2-vl`, `qwen3`, `clip-interrogator`, `api`)

### Model-Specific Parameters
- `--blip2_version`: BLIP2 version (`flan-t5`, `opt`)
- `--llava_version`: LLaVA version (`7b`, `13b`, `34b`)
- `--llava_port`: LLaVA Ollama port (default: 11434)
- `--qwen2vl_version`: Qwen2-VL version (`2b`, `7b`, `72b`)
- `--use_quantization`: Enable 4-bit quantization
- `--qwen3_version`: Qwen3 version (`8b`, `14b`, `72b`)
- `--qwen3_port`: Qwen3 Ollama port (default: 11434)
- `--clip_model`: CLIP model (default: `ViT-L-14/openai`)
- `--api_key`: API key for API-based models
- `--api_url`: API URL (default: OpenAI endpoint)

## Example Output

```
================================================================================
STEP 1: Computing/Loading Embeddings
================================================================================
Loading cached text embeddings from ./cache/original_embeddings/text/abc123.pt
Loading cached image embeddings from ./cache/original_embeddings/image/abc123.pt
Feature shape: (1000, 1536)
Labels shape: (1000, 4)

================================================================================
STEP 2: Splitting Data
================================================================================
Training samples: 800
Validation samples: 200

================================================================================
STEP 3: Training and Evaluating Classifiers
================================================================================

============================================================
Training Logistic Regression...
============================================================
Logistic Regression training completed.

Training Metrics for Logistic Regression:
  hamming_loss: 0.1234
  accuracy: 0.4567
  f1_micro: 0.7890
  f1_macro: 0.7654
  ...

Validation Metrics for Logistic Regression:
  hamming_loss: 0.1345
  accuracy: 0.4321
  f1_micro: 0.7654
  ...

[Similar output for other classifiers...]

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
Training complete!
================================================================================
```

## Workflow

### 1. **Embedding Computation** (First Run)
   - Loads dataset
   - Computes text embeddings using Qwen3
   - Computes image embeddings using SigLIP
   - Saves embeddings to cache
   - Takes ~5-10 minutes depending on dataset size

### 2. **Embedding Loading** (Subsequent Runs)
   - Loads pre-computed embeddings from cache
   - Takes ~5-10 seconds
   - Allows rapid experimentation with different classifiers

### 3. **Data Splitting**
   - Randomly splits data into train/validation
   - Default: 80% train, 20% validation
   - Uses same split for all classifiers for fair comparison

### 4. **Classifier Training**
   - Normalizes features using StandardScaler
   - Trains each classifier independently
   - Shows progress and metrics in real-time

### 5. **Evaluation & Comparison**
   - Computes comprehensive metrics for each classifier
   - Displays comparison table
   - Identifies best performing classifier

## Classifier Details

### Logistic Regression
- **Type**: Linear model with logistic function
- **Speed**: Very fast (seconds)
- **Use Case**: Quick baseline, interpretable results
- **Parameters**: `max_iter=1000`, L2 regularization

### Random Forest
- **Type**: Ensemble of 100 decision trees
- **Speed**: Fast to moderate (~1-2 minutes)
- **Use Case**: Non-linear patterns, robust to outliers
- **Parameters**: `n_estimators=100`, parallel execution

### MLP (Multi-Layer Perceptron)
- **Type**: Neural network with 2 hidden layers
- **Speed**: Moderate (~2-5 minutes)
- **Use Case**: Complex non-linear patterns
- **Parameters**: Hidden sizes `(512, 256)`, early stopping enabled

### Gradient Boosting
- **Type**: Sequential ensemble of decision trees
- **Speed**: Slow (~5-10 minutes)
- **Use Case**: Best accuracy, careful tuning needed
- **Parameters**: `n_estimators=100`, gradient boosting

## Performance Tips

### For Speed
```bash
# Use only fast classifiers
python baselines/ml_mlp_baseline.py \
    --classifiers "Logistic Regression" "Random Forest"
```

### For Accuracy
```bash
# Use MLP or Gradient Boosting with larger hidden dimension
python baselines/ml_mlp_baseline.py \
    --classifiers "MLP" "Gradient Boosting" \
    --hidden_dim 1024
```

### For Development
```bash
# Quick iteration with cached embeddings
python baselines/ml_mlp_baseline.py \
    --classifiers "Logistic Regression"
```

## Caching Behavior

### When to Clear Cache
Clear the cache if you:
- Modify the dataset files
- Change image description settings
- Change embedding models
- Want to regenerate embeddings

```bash
rm -rf ./cache/original_embeddings/
```

### Cache Performance
- **First run**: Compute embeddings (~5-10 min) + Train classifiers (~5-15 min)
- **Subsequent runs**: Load embeddings (~10 sec) + Train classifiers (~5-15 min)
- **Storage**: ~100-500 MB per dataset depending on size

## Multi-Label Classification Notes

- **Multi-Output Strategy**: Each label (A, B, C, D) is predicted independently
- **Threshold**: 0.5 probability threshold for binary predictions
- **Metrics**: Designed for multi-label (not multi-class) scenarios
- **Label Combinations**: Supports multiple labels per sample (e.g., A+B, A+C+D)

## Troubleshooting

### Out of Memory
Reduce batch size:
```bash
python baselines/ml_mlp_baseline.py --batch_size 4
```

### Slow Training
Use faster classifiers or smaller datasets:
```bash
python baselines/ml_mlp_baseline.py --classifiers "Logistic Regression"
```

### CUDA Out of Memory (Embeddings)
The embedding models run on GPU by default. If you encounter CUDA OOM:
- The embeddings are computed in batches
- Reduce `--batch_size`
- Or compute embeddings on CPU (automatic fallback)

## Advanced Usage

### Save Best Model

```python
# After training, access the best classifier
best_name, best_result = classifier_manager.get_best_classifier()
best_clf = best_result['classifier']

# Save using joblib or pickle
import joblib
joblib.dump(best_clf, 'best_classifier.pkl')
joblib.dump(classifier_manager.scaler, 'scaler.pkl')
```

### Custom Classifier Configuration

Modify the `MultiLabelClassifierManager.get_classifiers()` method to add custom classifiers or tune hyperparameters.

## Requirements

```
torch>=2.0.0
transformers>=4.30.0
pillow>=9.0.0
requests>=2.28.0
numpy>=1.23.0
scikit-learn>=1.2.0
qwen-vl-utils
clip-interrogator
```

Optional:
```
ollama  # For LLaVA and Qwen3 models
```
