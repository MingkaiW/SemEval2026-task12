# Baseline 4: Hyperbolic KG–Enhanced LLM QA

This folder implements **Baseline 4**, a hyperbolic extension of Baseline 3 (KG-Enhanced LLM QA).
It reuses the knowledge graph construction and LLM QA pipeline from `single_modality/baseline3`,
while switching the KG embedding model to **Lorentz / hyperbolic embeddings** (e.g., RotH, LorentzKG).

- Knowledge source: COMET-ATOMIC 2020 (recommended)
- KG embeddings: RotH / LorentzKG on Lorentz manifold (dim=32 by default)
- LLM: DeepSeek-chat (via OpenAI-compatible API), or other OpenAI / Anthropic models

## 1. Environment

Baseline 4 shares the same environment as Baseline 3:

- Python 3.10
- PyTorch 2.10.0
- transformers 4.36.2
- geoopt 0.5.1 (required for Lorentz/hyperbolic embeddings)
- openai 2.15.0, anthropic 0.76.0

Make sure `single_modality` is your working directory root so that `baseline` and `baseline3`
can be imported correctly.

## 2. Build Hyperbolic KG + Train Embeddings

Example command (train RotH, dim=32) on the official train split:

```bash
cd single_modality/baseline4

python run_baseline4.py build-kg \
    --data-path ../../train_data \
    --output-dir ./kg_output_hyper \
    --use-comet \
    --train-embedding \
    --kg-model RotH \
    --embedding-dim 32 \
    --epochs 100 \
    --batch-size 256
```

This will:

1. Load `../../train_data/questions.jsonl` and `docs.json` via `baseline.data_loader.AERDataLoader`.
2. Collect all target events and options as seed nodes.
3. Use COMET (if `--use-comet` is set) to generate causal triples and build a `CausalKnowledgeGraph`.
4. Train a hyperbolic KG embedding model (RotH or LorentzKG) using `KGEmbeddingTrainer` from `baseline3.kg_embedding`.
5. Save the graph and embeddings under `kg_output_hyper/`.

## 3. Run Hyperbolic KG–Enhanced LLM QA

Run QA on the dev split using DeepSeek-chat and prompt fusion:

```bash
python run_baseline4.py qa \
    --data-path ../../dev_data \
    --fusion prompt \
    --kg-path ./kg_output_hyper \
    --llm-type openai \
    --llm-model deepseek-chat \
    --api-base https://api.deepseek.com \
    --api-key YOUR_API_KEY \
    --use-comet \
    --output results_baseline4_dev.json \
    --submission-file submission_baseline4_dev.jsonl
```

For the test split, simply change `--data-path` to `../../test_data` and adjust output paths accordingly.
The QA pipeline is handled by `baseline3.kg_llm_qa.run_kg_llm_baseline`, which loads the
knowledge graph and embeddings from `--kg-path` and augments LLM prompts with hyperbolic KG context.

## 4. Notes

- If `geoopt` is not installed, RotH/LorentzKG will not be available; install with:
  `pip install geoopt`
- Baseline 4 is conceptually aligned with the description in `baseline4.md` at the project root.
  Hyperparameters such as embedding dimension and training epochs should match those described there
  for reproducibility.
