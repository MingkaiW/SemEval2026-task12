import os
import sys
import json
import base64
import glob
import logging
import argparse
import hashlib
from io import BytesIO
from typing import List, Dict, Tuple, Optional
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.submission_utils import save_submission, convert_numpy_predictions

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import requests
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import hamming_loss, accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

from transformers import (
    AutoProcessor,
    AutoModel,
    AutoTokenizer,
    SiglipModel,
    SiglipProcessor,
    Blip2ForConditionalGeneration,
    AutoModelForImageTextToText,
    BitsAndBytesConfig
)
from qwen_vl_utils import process_vision_info
from clip_interrogator import Config, Interrogator

try:
    import ollama
except ImportError:
    # Allow environments without ollama installed
    pass

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==========================================
# 1. Visual Description Generator
# ==========================================

class VisualDescriptionGenerator:
    """
    Generates textual descriptions from images using various vision-language models.
    Adapted from Image_to_Text.py for integration with the baseline.
    """
    def __init__(self, model_type='blip2', device='cuda' if torch.cuda.is_available() else 'cpu', **kwargs):
        self.model_type = model_type
        self.device = device
        self.kwargs = kwargs

        self.prompt = """
        Please provide a detailed visual description of this image.
        Include key objects, their spatial relationships,
        notable visual features, and any observable actions or events.
        Respond in clear, structured English paragraphs.
        """

        if self.model_type == 'api':
            self._validate_api_credentials()
        elif self.model_type == 'blip2':
            self.processor, self.model = self._init_BLIP2()
        elif self.model_type == 'llava':
            self.client = self._init_LLaVa()
        elif self.model_type == 'qwen2-vl':
            self.processor, self.model = self._init_Qwen2_VL()
        elif self.model_type == 'qwen3':
            self.client = self._init_Qwen3()
        elif self.model_type == 'clip-interrogator':
            self.ci = self._init_CLIP_Interrogator()

    def _validate_api_credentials(self):
        if 'api_key' not in self.kwargs or not self.kwargs['api_key']:
            raise ValueError("API key is required for API model type.")

    def _init_BLIP2(self):
        model_map = {
            'flan-t5': "Salesforce/blip2-flan-t5-xl",
            'opt': "Salesforce/blip2-opt-2.7b"
        }
        version = self.kwargs.get('blip2_version', 'opt')
        logger.info(f"Loading BLIP2 ({version})...")
        processor = AutoProcessor.from_pretrained(model_map[version])
        model = Blip2ForConditionalGeneration.from_pretrained(
            model_map[version]
        ).to(self.device)
        model.eval()
        return processor, model

    def _init_LLaVa(self):
        port = self.kwargs.get('llava_port', 11434)
        return ollama.Client(host=f"http://localhost:{port}")

    def _init_Qwen3(self):
        port = self.kwargs.get('qwen3_port', 11434)
        return ollama.Client(host=f"http://localhost:{port}")

    def _init_Qwen2_VL(self):
        model_map = {
            '2b': "Qwen/Qwen2-VL-2B-Instruct",
            '7b': "Qwen/Qwen2-VL-7B-Instruct",
            '72b': "Qwen/Qwen2-VL-72B-Instruct"
        }
        version = self.kwargs.get('qwen2vl_version', '2b')
        use_quantization = self.kwargs.get('use_quantization', False)

        logger.info(f"Loading Qwen2-VL ({version})...")

        if use_quantization:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4"
            )
        else:
            quant_config = None

        processor = AutoProcessor.from_pretrained(
            model_map[version],
            trust_remote_code=True
        )
        model = AutoModelForImageTextToText.from_pretrained(
            model_map[version],
            torch_dtype=torch.float16,
            quantization_config=quant_config,
            trust_remote_code=True
        ).to(self.device)
        model.eval()
        return processor, model

    def _init_CLIP_Interrogator(self):
        logger.info("Loading CLIP Interrogator...")
        config = Config()
        clip_model = self.kwargs.get('clip_model', 'ViT-L-14/openai')
        config.clip_model_name = clip_model
        return Interrogator(config)

    def generate_description(self, image: Image.Image) -> Optional[str]:
        """Generate description for a PIL Image object."""
        try:
            if self.model_type == 'api':
                return self._generate_API_description(image)
            elif self.model_type == 'blip2':
                return self._generate_BLIP2_description(image)
            elif self.model_type == 'llava':
                return self._generate_LLaVa_description(image)
            elif self.model_type == 'qwen2-vl':
                return self._generate_Qwen2_VL_description(image)
            elif self.model_type == 'qwen3':
                return self._generate_Qwen3_description(image)
            elif self.model_type == 'clip-interrogator':
                return self._generate_CLIP_Interrogator_description(image)
        except Exception as e:
            logger.warning(f"Failed to generate description: {str(e)}")
            return None

    def _generate_API_description(self, image: Image.Image):
        base64_image = self._image_to_base64(image)
        api_key = self.kwargs.get('api_key')
        api_url = self.kwargs.get('api_url', "https://api.openai.com/v1/chat/completions")

        headers = {"Authorization": f"Bearer {api_key}"}
        data = {
            "model": "gpt-4o",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": self.prompt},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"}
                    }
                ]
            }]
        }
        response = requests.post(api_url, headers=headers, json=data)
        return response.json()['choices'][0]['message']['content'].strip()

    def _generate_BLIP2_description(self, image: Image.Image):
        image = image.convert('RGB')
        inputs = self.processor(
            images=image,
            text=self.prompt,
            return_tensors="pt"
        ).to(self.device)

        generated_ids = self.model.generate(
            **inputs,
            max_length=300,
            num_beams=5,
            temperature=0.7
        )
        return self.processor.decode(generated_ids[0], skip_special_tokens=True)

    def _generate_LLaVa_description(self, image: Image.Image):
        base64_image = self._image_to_base64(image)
        version = self.kwargs.get('llava_version', '7b')
        response = self.client.chat(
            model=f"llava:{version}",
            messages=[{
                'role': 'user',
                'content': self.prompt,
                'images': [base64_image]
            }]
        )
        return response['message']['content'].strip()

    def _generate_Qwen2_VL_description(self, image: Image.Image):
        base64_image = self._image_to_base64(image)
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": self.prompt},
                {"type": "image", "image": f"data:image;base64,{base64_image}"}
            ]
        }]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=1500,
            temperature=0.7
        )
        return self.processor.decode(generated_ids[0], skip_special_tokens=True)

    def _generate_Qwen3_description(self, image: Image.Image):
        base64_image = self._image_to_base64(image)
        version = self.kwargs.get('qwen3_version', '8b')
        response = self.client.chat(
            model=f"qwen3-vl:{version}",
            messages=[{
                'role': 'user',
                'content': self.prompt,
                'images': [base64_image]
            }]
        )
        return response['message']['content'].strip()

    def _generate_CLIP_Interrogator_description(self, image: Image.Image):
        return self.ci.interrogate(image.convert('RGB'))

    @staticmethod
    def _image_to_base64(image: Image.Image):
        buffered = BytesIO()
        image.convert("RGB").save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

# ==========================================
# 2. Embedding Cache Manager
# ==========================================

class EmbeddingCache:
    """Manages caching of text and image embeddings to disk."""

    def __init__(self, cache_dir: str = "./cache/original_embeddings"):
        self.cache_dir = Path(cache_dir)
        self.text_cache_dir = self.cache_dir / "text"
        self.image_cache_dir = self.cache_dir / "image"

        # Create cache directories
        self.text_cache_dir.mkdir(parents=True, exist_ok=True)
        self.image_cache_dir.mkdir(parents=True, exist_ok=True)

    def get_cache_key(self, data_path: str, split: str = "train") -> str:
        """Generate cache key based on data path and split."""
        # Use the absolute path and split to create a unique identifier
        path_str = f"{os.path.abspath(data_path)}_{split}"
        return hashlib.md5(path_str.encode()).hexdigest()

    def get_text_cache_path(self, cache_key: str) -> Path:
        """Get path for cached text embeddings."""
        return self.text_cache_dir / f"{cache_key}.pt"

    def get_image_cache_path(self, cache_key: str) -> Path:
        """Get path for cached image embeddings."""
        return self.image_cache_dir / f"{cache_key}.pt"

    def save_text_embeddings(self, cache_key: str, embeddings: torch.Tensor):
        """Save text embeddings to cache."""
        cache_path = self.get_text_cache_path(cache_key)
        torch.save(embeddings.cpu(), cache_path)
        logger.info(f"Saved text embeddings to {cache_path}")

    def save_image_embeddings(self, cache_key: str, embeddings: torch.Tensor):
        """Save image embeddings to cache."""
        cache_path = self.get_image_cache_path(cache_key)
        torch.save(embeddings.cpu(), cache_path)
        logger.info(f"Saved image embeddings to {cache_path}")

    def load_text_embeddings(self, cache_key: str) -> Optional[torch.Tensor]:
        """Load text embeddings from cache if available."""
        cache_path = self.get_text_cache_path(cache_key)
        if cache_path.exists():
            logger.info(f"Loading cached text embeddings from {cache_path}")
            return torch.load(cache_path)
        return None

    def load_image_embeddings(self, cache_key: str) -> Optional[torch.Tensor]:
        """Load image embeddings from cache if available."""
        cache_path = self.get_image_cache_path(cache_key)
        if cache_path.exists():
            logger.info(f"Loading cached image embeddings from {cache_path}")
            return torch.load(cache_path)
        return None

    def cache_exists(self, cache_key: str) -> Tuple[bool, bool]:
        """Check if both text and image caches exist."""
        text_exists = self.get_text_cache_path(cache_key).exists()
        image_exists = self.get_image_cache_path(cache_key).exists()
        return text_exists, image_exists

# ==========================================
# 3. Embedding Modules
# ==========================================

class MultimodalEmbedder:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device

        # --- Text Embedder (Qwen3) ---
        logger.info("Loading Qwen3 Text Embedder...")
        self.text_model_id = "Qwen/Qwen3-Embedding-0.6B"
        try:
            self.text_tokenizer = AutoTokenizer.from_pretrained(self.text_model_id, trust_remote_code=True)
            self.text_model = AutoModel.from_pretrained(self.text_model_id, trust_remote_code=True).to(self.device)
            self.text_model.eval()
        except Exception as e:
            logger.error(f"Failed to load Qwen3: {e}. Ensure the model ID is correct and accessible.")
            raise e

        # --- Image Embedder (SigLIP) ---
        logger.info("Loading SigLIP Image Embedder...")
        self.vision_model_id = "google/siglip-base-patch16-224"
        try:
            self.vision_processor = SiglipProcessor.from_pretrained(self.vision_model_id)
            self.vision_model = SiglipModel.from_pretrained(self.vision_model_id).to(self.device)
            self.vision_model.eval()
        except Exception as e:
            logger.error(f"Failed to load SigLIP: {e}")
            raise e

    def embed_text(self, texts: List[str]) -> torch.Tensor:
        with torch.no_grad():
            inputs = self.text_tokenizer(
                texts, padding=True, truncation=True, return_tensors="pt", max_length=512
            ).to(self.device)
            outputs = self.text_model(**inputs)
            embeddings = self.mean_pooling(outputs, inputs['attention_mask'])
            return embeddings

    def embed_image(self, images: List[Image.Image]) -> torch.Tensor:
        valid_images = [img for img in images if img is not None]
        if not valid_images:
            return torch.zeros((len(images), self.vision_model.config.vision_config.hidden_size)).to(self.device)

        with torch.no_grad():
            inputs = self.vision_processor(images=valid_images, return_tensors="pt").to(self.device)
            outputs = self.vision_model.get_image_features(**inputs)
            return outputs

    @staticmethod
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# ==========================================
# 4. Dataset & Dataloader
# ==========================================

class AERDataset(Dataset):
    def __init__(self, questions_path: str, docs_path: str,
                 use_image_descriptions: bool = False,
                 description_generator: Optional[VisualDescriptionGenerator] = None):
        self.samples = []
        self.topic_docs = {}
        self.use_image_descriptions = use_image_descriptions
        self.description_generator = description_generator

        # Load Docs
        if os.path.exists(docs_path):
            with open(docs_path, 'r', encoding='utf-8') as f:
                docs_data = json.load(f)
                for entry in docs_data:
                    self.topic_docs[entry['topic_id']] = entry['docs']

        # Load Questions
        with open(questions_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        topic_id = item['topic_id']

        # 1. Construct Text Input
        related_docs = self.topic_docs.get(topic_id, [])
        context_text = " ".join([d.get('snippet', '') for d in related_docs[:3]])

        full_text = f"Event: {item['target_event']} Context: {context_text} " \
                    f"Option A: {item['option_A']} Option B: {item['option_B']} " \
                    f"Option C: {item['option_C']} Option D: {item['option_D']}"

        # 2. Retrieve Image and optionally generate description
        image = None
        image_description = ""

        for doc in related_docs:
            if doc.get('imageUrl'):
                img_data = self.load_image(doc['imageUrl'])
                if img_data:
                    image = img_data

                    # Generate image description if enabled
                    if self.use_image_descriptions and self.description_generator:
                        desc = self.description_generator.generate_description(image)
                        if desc:
                            image_description = f" Image Description: {desc}"
                    break

        # Append image description to text if available
        if image_description:
            full_text += image_description

        # 3. Process Labels
        label_vector = np.zeros(4, dtype=np.float32)
        golden = item.get('golden_answer', '')
        if 'A' in golden: label_vector[0] = 1.0
        if 'B' in golden: label_vector[1] = 1.0
        if 'C' in golden: label_vector[2] = 1.0
        if 'D' in golden: label_vector[3] = 1.0

        return {
            "id": item.get('id', f"sample_{idx}"),
            "text": full_text,
            "image": image,
            "labels": label_vector
        }

    def load_image(self, source):
        try:
            if source.startswith('data:image'):
                header, encoded = source.split(',', 1)
                return Image.open(BytesIO(base64.b64decode(encoded))).convert("RGB")
            elif source.startswith('http'):
                resp = requests.get(source, timeout=3)
                if resp.status_code == 200:
                    return Image.open(BytesIO(resp.content)).convert("RGB")
        except Exception:
            return None
        return None

def collate_fn(batch):
    ids = [item['id'] for item in batch]
    texts = [item['text'] for item in batch]
    images = [item['image'] for item in batch]
    labels = torch.tensor(np.array([item['labels'] for item in batch]))
    return ids, texts, images, labels

class CachedEmbeddingDataset(Dataset):
    """Dataset that uses pre-computed cached embeddings."""

    def __init__(self, text_embeddings: torch.Tensor, image_embeddings: torch.Tensor,
                 labels: torch.Tensor):
        self.text_embeddings = text_embeddings
        self.image_embeddings = image_embeddings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'text_emb': self.text_embeddings[idx],
            'image_emb': self.image_embeddings[idx],
            'label': self.labels[idx]
        }

def cached_collate_fn(batch):
    """Collate function for cached embeddings."""
    text_embs = torch.stack([item['text_emb'] for item in batch])
    image_embs = torch.stack([item['image_emb'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    return text_embs, image_embs, labels

# ==========================================
# 5. Scikit-learn Classifier Manager
# ==========================================

class MultiLabelClassifierManager:
    """
    Manages multiple scikit-learn classifiers for multi-label classification.
    """
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.classifiers = {}
        self.results = {}

    def get_classifiers(self, hidden_dim=512):
        """
        Returns a dictionary of classifiers to compare.
        """
        classifiers = {
            'Logistic Regression': MultiOutputClassifier(
                LogisticRegression(max_iter=1000, random_state=self.random_state, n_jobs=-1)
            ),
            'Random Forest': MultiOutputClassifier(
                RandomForestClassifier(n_estimators=100, random_state=self.random_state, n_jobs=-1)
            ),
            'MLP': MultiOutputClassifier(
                MLPClassifier(hidden_layer_sizes=(hidden_dim, hidden_dim // 2),
                             max_iter=100, random_state=self.random_state,
                             early_stopping=True, validation_fraction=0.1)
            ),
            'Gradient Boosting': MultiOutputClassifier(
                GradientBoostingClassifier(n_estimators=100, random_state=self.random_state)
            ),
        }
        return classifiers

    def evaluate(self, y_true, y_pred, classifier_name):
        """
        Evaluate multi-label classification performance.
        """
        # Convert predictions to binary
        y_pred_binary = (y_pred >= 0.5).astype(int)

        metrics = {
            'hamming_loss': hamming_loss(y_true, y_pred_binary),
            'accuracy': accuracy_score(y_true, y_pred_binary),
            'f1_micro': f1_score(y_true, y_pred_binary, average='micro', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred_binary, average='macro', zero_division=0),
            'f1_samples': f1_score(y_true, y_pred_binary, average='samples', zero_division=0),
            'precision_micro': precision_score(y_true, y_pred_binary, average='micro', zero_division=0),
            'recall_micro': recall_score(y_true, y_pred_binary, average='micro', zero_division=0),
        }

        return metrics

    def train_and_evaluate(self, X_train, y_train, X_val=None, y_val=None, classifiers_to_use=None):
        """
        Train and evaluate all classifiers.
        """
        # Normalize features
        logger.info("Normalizing features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        if X_val is not None:
            X_val_scaled = self.scaler.transform(X_val)

        # Get classifiers
        all_classifiers = self.get_classifiers()
        if classifiers_to_use:
            classifiers = {k: v for k, v in all_classifiers.items() if k in classifiers_to_use}
        else:
            classifiers = all_classifiers

        results = {}

        for name, clf in classifiers.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"Training {name}...")
            logger.info(f"{'='*60}")

            try:
                # Train
                clf.fit(X_train_scaled, y_train)
                logger.info(f"{name} training completed.")

                # Predict on training set
                y_train_pred = clf.predict_proba(X_train_scaled) if hasattr(clf, 'predict_proba') else clf.predict(X_train_scaled)
                if isinstance(y_train_pred, list):  # MultiOutputClassifier returns list of arrays
                    y_train_pred = np.column_stack([pred[:, 1] if pred.ndim > 1 else pred for pred in y_train_pred])

                train_metrics = self.evaluate(y_train, y_train_pred, name)
                logger.info(f"\nTraining Metrics for {name}:")
                for metric, value in train_metrics.items():
                    logger.info(f"  {metric}: {value:.4f}")

                results[name] = {
                    'classifier': clf,
                    'train_metrics': train_metrics
                }

                # Evaluate on validation set if provided
                if X_val is not None and y_val is not None:
                    y_val_pred = clf.predict_proba(X_val_scaled) if hasattr(clf, 'predict_proba') else clf.predict(X_val_scaled)
                    if isinstance(y_val_pred, list):
                        y_val_pred = np.column_stack([pred[:, 1] if pred.ndim > 1 else pred for pred in y_val_pred])

                    val_metrics = self.evaluate(y_val, y_val_pred, name)
                    logger.info(f"\nValidation Metrics for {name}:")
                    for metric, value in val_metrics.items():
                        logger.info(f"  {metric}: {value:.4f}")

                    results[name]['val_metrics'] = val_metrics

            except Exception as e:
                logger.error(f"Error training {name}: {str(e)}")
                continue

        self.results = results
        return results

    def print_comparison(self):
        """
        Print comparison table of all classifiers.
        """
        if not self.results:
            logger.warning("No results to compare. Train models first.")
            return

        logger.info("\n" + "="*80)
        logger.info("CLASSIFIER COMPARISON")
        logger.info("="*80)

        # Print header
        logger.info(f"\n{'Classifier':<25} {'F1-Micro':<12} {'F1-Macro':<12} {'Accuracy':<12} {'Hamming Loss':<12}")
        logger.info("-" * 80)

        # Print results for each classifier
        for name, result in self.results.items():
            metrics = result.get('val_metrics', result.get('train_metrics'))
            logger.info(f"{name:<25} {metrics['f1_micro']:<12.4f} {metrics['f1_macro']:<12.4f} "
                       f"{metrics['accuracy']:<12.4f} {metrics['hamming_loss']:<12.4f}")

        logger.info("="*80)

    def get_best_classifier(self, metric='f1_micro'):
        """
        Get the best performing classifier based on a metric.
        """
        if not self.results:
            return None

        best_name = None
        best_score = -float('inf')

        for name, result in self.results.items():
            metrics = result.get('val_metrics', result.get('train_metrics'))
            score = metrics.get(metric, 0)

            if metric == 'hamming_loss':
                score = -score  # Lower is better

            if score > best_score:
                best_score = score
                best_name = name

        logger.info(f"\nBest classifier based on {metric}: {best_name} (score: {abs(best_score):.4f})")
        return best_name, self.results[best_name]

# ==========================================
# 6. Embedding Pre-computation Functions
# ==========================================

def precompute_embeddings(dataset: AERDataset, embedder: MultimodalEmbedder,
                         cache: EmbeddingCache, cache_key: str,
                         batch_size: int = 8) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
    """
    Pre-compute all embeddings for the dataset and cache them.
    Returns: (text_embeddings, image_embeddings, labels, sample_ids)
    """
    # Check if cache exists
    text_cache_exists, image_cache_exists = cache.cache_exists(cache_key)

    if text_cache_exists and image_cache_exists:
        logger.info("Loading embeddings from cache...")
        text_embeddings = cache.load_text_embeddings(cache_key)
        image_embeddings = cache.load_image_embeddings(cache_key)

        # Collect labels and IDs from dataset
        logger.info("Collecting labels and IDs...")
        labels_list = []
        ids_list = []
        for i in range(len(dataset)):
            sample = dataset[i]
            labels_list.append(sample['labels'])
            ids_list.append(sample['id'])
        labels = torch.tensor(np.array(labels_list))

        return text_embeddings, image_embeddings, labels, ids_list

    # Need to compute embeddings
    logger.info("Computing embeddings (this may take a while)...")

    text_emb_list = []
    image_emb_list = []
    labels_list = []
    ids_list = []

    # Create a temporary dataloader for batch processing
    temp_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    for batch_idx, (batch_ids, batch_texts, batch_images, batch_labels) in enumerate(temp_loader):
        if (batch_idx + 1) % 10 == 0:
            logger.info(f"Processing batch {batch_idx + 1}/{len(temp_loader)}")

        # Collect IDs
        ids_list.extend(batch_ids)

        # Generate text embeddings
        text_emb = embedder.embed_text(batch_texts)
        text_emb_list.append(text_emb.cpu())

        # Generate image embeddings
        processed_imgs = []
        for img in batch_images:
            if img is None:
                processed_imgs.append(Image.new('RGB', (224, 224), color='black'))
            else:
                processed_imgs.append(img)

        img_emb = embedder.embed_image(processed_imgs)
        image_emb_list.append(img_emb.cpu())

        labels_list.append(batch_labels)

    # Concatenate all batches
    text_embeddings = torch.cat(text_emb_list, dim=0)
    image_embeddings = torch.cat(image_emb_list, dim=0)
    labels = torch.cat(labels_list, dim=0)

    # Save to cache
    logger.info("Saving embeddings to cache...")
    cache.save_text_embeddings(cache_key, text_embeddings)
    cache.save_image_embeddings(cache_key, image_embeddings)

    return text_embeddings, image_embeddings, labels, ids_list

# ==========================================
# 7. Utilities for Evaluation and Saving Results
# ==========================================

def save_results(output_dir: str, split_name: str, metrics: Dict, predictions: Optional[np.ndarray] = None,
                true_labels: Optional[np.ndarray] = None, classifier_name: str = "best"):
    """Save evaluation results and predictions to files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save metrics
    metrics_file = output_path / f"{split_name}_{classifier_name}_metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics to {metrics_file}")

    # Save predictions if provided
    if predictions is not None:
        pred_file = output_path / f"{split_name}_{classifier_name}_predictions.npy"
        np.save(pred_file, predictions)
        logger.info(f"Saved predictions to {pred_file}")

    # Save true labels if provided
    if true_labels is not None:
        labels_file = output_path / f"{split_name}_{classifier_name}_labels.npy"
        np.save(labels_file, true_labels)
        logger.info(f"Saved true labels to {labels_file}")

def load_and_prepare_split(questions_path: str, docs_path: str, embedder: MultimodalEmbedder,
                          cache: EmbeddingCache, split_name: str, batch_size: int = 8,
                          use_image_descriptions: bool = False,
                          description_generator: Optional[VisualDescriptionGenerator] = None,
                          disable_cache: bool = False) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load and prepare embeddings for a data split.

    Returns: (features, labels, sample_ids)
    """
    logger.info(f"\nProcessing {split_name} split...")

    # Create dataset
    dataset = AERDataset(
        questions_path,
        docs_path,
        use_image_descriptions=use_image_descriptions,
        description_generator=description_generator
    )

    # Get cache key
    cache_key = cache.get_cache_key(questions_path, split=split_name)

    # Pre-compute or load embeddings
    if not disable_cache:
        text_embeddings, image_embeddings, labels, sample_ids = precompute_embeddings(
            dataset, embedder, cache, cache_key, batch_size=batch_size
        )
    else:
        # Compute without caching
        temp_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

        text_emb_list = []
        image_emb_list = []
        labels_list = []
        sample_ids = []

        for batch_ids, batch_texts, batch_images, batch_labels in temp_loader:
            sample_ids.extend(batch_ids)

            text_emb = embedder.embed_text(batch_texts)
            text_emb_list.append(text_emb.cpu())

            processed_imgs = []
            for img in batch_images:
                if img is None:
                    processed_imgs.append(Image.new('RGB', (224, 224), color='black'))
                else:
                    processed_imgs.append(img)

            img_emb = embedder.embed_image(processed_imgs)
            image_emb_list.append(img_emb.cpu())
            labels_list.append(batch_labels)

        text_embeddings = torch.cat(text_emb_list, dim=0)
        image_embeddings = torch.cat(image_emb_list, dim=0)
        labels = torch.cat(labels_list, dim=0)

    # Combine embeddings
    features = torch.cat((text_embeddings, image_embeddings), dim=1).numpy()
    labels = labels.numpy()

    logger.info(f"{split_name} - Feature shape: {features.shape}, Labels shape: {labels.shape}")

    return features, labels, sample_ids

# ==========================================
# 8. Main Execution Pipeline
# ==========================================

def parse_args():
    parser = argparse.ArgumentParser(description="Multimodal Multi-Label Classifier with Optional Image-to-Text")

    # Data paths
    parser.add_argument('--train_questions', type=str, default="train_data/questions.jsonl",
                       help="Path to training questions JSONL file")
    parser.add_argument('--train_docs', type=str, default="train_data/docs.json",
                       help="Path to training documents JSON file")
    parser.add_argument('--dev_questions', type=str, default="dev_data/questions.jsonl",
                       help="Path to dev questions JSONL file")
    parser.add_argument('--dev_docs', type=str, default="dev_data/docs.json",
                       help="Path to dev documents JSON file")
    parser.add_argument('--test_questions', type=str, default="test_data/questions.jsonl",
                       help="Path to test questions JSONL file")
    parser.add_argument('--test_docs', type=str, default="test_data/docs.json",
                       help="Path to test documents JSON file")
    parser.add_argument('--use_dev', action='store_true',
                       help="Use dev set for evaluation")
    parser.add_argument('--use_test', action='store_true',
                       help="Use test set for evaluation")

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size for embedding computation")
    parser.add_argument('--hidden_dim', type=int, default=512, help="Hidden dimension for MLP")
    parser.add_argument('--val_split', type=float, default=0.2, help="Validation split ratio")
    parser.add_argument('--classifiers', type=str, nargs='+',
                       choices=['Logistic Regression', 'Random Forest', 'MLP', 'Gradient Boosting'],
                       help="Specific classifiers to use (default: all)")

    # Cache parameters
    parser.add_argument('--cache_dir', type=str, default="./cache/original_embeddings",
                       help="Directory for caching embeddings")
    parser.add_argument('--disable_cache', action='store_true',
                       help="Disable embedding caching (recompute every time)")

    # Output parameters
    parser.add_argument('--output_dir', type=str, default="./results",
                       help="Directory for saving results")
    parser.add_argument('--save_predictions', action='store_true',
                       help="Save predictions to file")
    parser.add_argument('--submission_file', type=str, default=None,
                       help="Path to save submission JSONL file (e.g., submission.jsonl)")

    # Image-to-text options
    parser.add_argument('--use_image_descriptions', action='store_true',
                       help="Enable image description generation")
    parser.add_argument('--img2txt_model', type=str, choices=['blip2', 'llava', 'qwen2-vl',
                                                                'qwen3', 'clip-interrogator', 'api'],
                       default='blip2', help="Image-to-text model type")

    # Model-specific parameters
    parser.add_argument('--blip2_version', type=str, choices=['flan-t5', 'opt'], default='opt')
    parser.add_argument('--llava_version', type=str, choices=['7b', '13b', '34b'], default='7b')
    parser.add_argument('--llava_port', type=int, default=11434)
    parser.add_argument('--qwen2vl_version', type=str, choices=['2b', '7b', '72b'], default='2b')
    parser.add_argument('--use_quantization', action='store_true')
    parser.add_argument('--qwen3_version', type=str, choices=['8b', '14b', '72b'], default='8b')
    parser.add_argument('--qwen3_port', type=int, default=11434)
    parser.add_argument('--clip_model', type=str, default="ViT-L-14/openai")
    parser.add_argument('--api_key', type=str, help="API key for API-based models")
    parser.add_argument('--api_url', type=str, default="https://api.openai.com/v1/chat/completions")

    return parser.parse_args()

def main():
    args = parse_args()

    # Fallback to sample data if training data doesn't exist
    if not os.path.exists(args.train_questions):
        logger.warning(f"Training data not found at {args.train_questions}, using sample data")
        args.train_questions = "sample_data/questions.jsonl"
        args.train_docs = "sample_data/docs.json"

    # 1. Setup Image Description Generator (if enabled)
    description_generator = None
    if args.use_image_descriptions:
        logger.info(f"Initializing {args.img2txt_model} for image description generation...")
        img2txt_kwargs = {
            'blip2_version': args.blip2_version,
            'llava_version': args.llava_version,
            'llava_port': args.llava_port,
            'qwen2vl_version': args.qwen2vl_version,
            'use_quantization': args.use_quantization,
            'qwen3_version': args.qwen3_version,
            'qwen3_port': args.qwen3_port,
            'clip_model': args.clip_model,
            'api_key': args.api_key,
            'api_url': args.api_url
        }
        description_generator = VisualDescriptionGenerator(
            model_type=args.img2txt_model,
            **img2txt_kwargs
        )

    # 2. Setup Cache Manager and Embedder
    cache = EmbeddingCache(cache_dir=args.cache_dir)
    embedder = MultimodalEmbedder()

    # 3. Load Training Data
    logger.info("\n" + "="*80)
    logger.info("STEP 1: Loading Training Data")
    logger.info("="*80)

    X_train, y_train, train_ids = load_and_prepare_split(
        args.train_questions,
        args.train_docs,
        embedder,
        cache,
        "train",
        batch_size=args.batch_size,
        use_image_descriptions=args.use_image_descriptions,
        description_generator=description_generator,
        disable_cache=args.disable_cache
    )

    # 4. Load Dev/Test Data if requested
    X_dev, y_dev, dev_ids = None, None, None
    X_test, y_test, test_ids = None, None, None

    if args.use_dev and os.path.exists(args.dev_questions):
        logger.info("\n" + "="*80)
        logger.info("STEP 2: Loading Dev Data")
        logger.info("="*80)
        X_dev, y_dev, dev_ids = load_and_prepare_split(
            args.dev_questions,
            args.dev_docs,
            embedder,
            cache,
            "dev",
            batch_size=args.batch_size,
            use_image_descriptions=args.use_image_descriptions,
            description_generator=description_generator,
            disable_cache=args.disable_cache
        )

    if args.use_test and os.path.exists(args.test_questions):
        logger.info("\n" + "="*80)
        logger.info(f"STEP {3 if args.use_dev else 2}: Loading Test Data")
        logger.info("="*80)
        X_test, y_test, test_ids = load_and_prepare_split(
            args.test_questions,
            args.test_docs,
            embedder,
            cache,
            "test",
            batch_size=args.batch_size,
            use_image_descriptions=args.use_image_descriptions,
            description_generator=description_generator,
            disable_cache=args.disable_cache
        )

    # 5. Prepare validation set (use dev if available, else split train)
    logger.info("\n" + "="*80)
    logger.info(f"STEP {4 if args.use_test else 3 if args.use_dev else 2}: Preparing Training/Validation Split")
    logger.info("="*80)

    if X_dev is not None:
        # Use dev set as validation
        X_train_final, y_train_final = X_train, y_train
        X_val, y_val = X_dev, y_dev
        logger.info("Using dev set as validation set")
    else:
        # Split training set
        n_samples = X_train.shape[0]
        n_val = int(n_samples * args.val_split)
        indices = np.random.permutation(n_samples)

        val_indices = indices[:n_val]
        train_indices = indices[n_val:]

        X_train_final = X_train[train_indices]
        y_train_final = y_train[train_indices]
        X_val = X_train[val_indices]
        y_val = y_train[val_indices]
        logger.info(f"Split training set: {len(X_train_final)} train, {len(X_val)} val")

    logger.info(f"Training samples: {len(X_train_final)}")
    logger.info(f"Validation samples: {len(X_val)}")

    # 6. Train and Evaluate Classifiers
    logger.info("\n" + "="*80)
    logger.info(f"STEP {5 if args.use_test else 4 if args.use_dev else 3}: Training and Evaluating Classifiers")
    logger.info("="*80)

    classifier_manager = MultiLabelClassifierManager()
    classifier_manager.train_and_evaluate(
        X_train_final, y_train_final, X_val, y_val,
        classifiers_to_use=args.classifiers
    )

    # 7. Print Comparison
    classifier_manager.print_comparison()

    # 8. Get Best Classifier
    best_name, best_result = classifier_manager.get_best_classifier(metric='f1_micro')
    best_clf = best_result['classifier']

    # 9. Evaluate on Test Set if available
    if X_test is not None:
        logger.info("\n" + "="*80)
        logger.info("STEP 6: Evaluating on Test Set")
        logger.info("="*80)

        X_test_scaled = classifier_manager.scaler.transform(X_test)
        y_test_pred = best_clf.predict_proba(X_test_scaled) if hasattr(best_clf, 'predict_proba') else best_clf.predict(X_test_scaled)

        if isinstance(y_test_pred, list):
            y_test_pred = np.column_stack([pred[:, 1] if pred.ndim > 1 else pred for pred in y_test_pred])

        test_metrics = classifier_manager.evaluate(y_test, y_test_pred, best_name)

        logger.info(f"\nTest Set Metrics ({best_name}):")
        for metric, value in test_metrics.items():
            logger.info(f"  {metric}: {value:.4f}")

        # Save test results
        if args.save_predictions:
            save_results(args.output_dir, "test", test_metrics, y_test_pred, y_test, best_name)

        # Save submission file for test set
        if args.submission_file and test_ids:
            submission_preds = convert_numpy_predictions(y_test_pred, test_ids, threshold=0.5)
            save_submission(submission_preds, args.submission_file)
            logger.info(f"Saved submission to {args.submission_file}")

    # 10. Evaluate best model on dev set
    if X_dev is not None:
        logger.info("\n" + "="*80)
        logger.info("Final Dev Set Evaluation")
        logger.info("="*80)

        X_dev_scaled = classifier_manager.scaler.transform(X_dev)
        y_dev_pred = best_clf.predict_proba(X_dev_scaled) if hasattr(best_clf, 'predict_proba') else best_clf.predict(X_dev_scaled)

        if isinstance(y_dev_pred, list):
            y_dev_pred = np.column_stack([pred[:, 1] if pred.ndim > 1 else pred for pred in y_dev_pred])

        dev_metrics = classifier_manager.evaluate(y_dev, y_dev_pred, best_name)

        logger.info(f"\nDev Set Metrics ({best_name}):")
        for metric, value in dev_metrics.items():
            logger.info(f"  {metric}: {value:.4f}")

        # Save dev results
        if args.save_predictions:
            save_results(args.output_dir, "dev", dev_metrics, y_dev_pred, y_dev, best_name)

        # Save submission file for dev set if no test set
        if args.submission_file and dev_ids and X_test is None:
            submission_preds = convert_numpy_predictions(y_dev_pred, dev_ids, threshold=0.5)
            save_submission(submission_preds, args.submission_file)
            logger.info(f"Saved submission to {args.submission_file}")

    # 11. Save all results
    if args.save_predictions:
        # Save all classifier results
        results_summary = {}
        for name, result in classifier_manager.results.items():
            results_summary[name] = {
                'train_metrics': result['train_metrics'],
                'val_metrics': result.get('val_metrics', {})
            }

        summary_file = Path(args.output_dir) / "all_classifiers_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(results_summary, f, indent=2)
        logger.info(f"\nSaved all results to {summary_file}")

    logger.info("\n" + "="*80)
    logger.info("Training and Evaluation Complete!")
    logger.info("="*80)

if __name__ == "__main__":
    main()
