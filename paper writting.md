## Introduction

Understanding why events happen is crucial not only for humans making sense of the world, but also for intelligent systems that aim to interpret it. While large language models (LLMs) have achieved strong performance on tasks such as event extraction, summarization, and even forecasting, they still struggle with abductive reasoning: inferring the most plausible cause of an observed outcome from incomplete, noisy, or distributed evidence.

SemEval-2026 Task 12: Abductive Event Reasoning (AER) is designed to probe this ability. Given a real-world event and a set of retrieved documents, systems must identify the most plausible and direct cause among multiple natural-language candidates. Compared to standard information extraction or summarization, AER demands structured, context-grounded causal reasoning that combines textual evidence with prior knowledge. Progress on this task has direct implications for transparency, explainability, and decision making, with potential applications in journalism, analysis, and public information systems.

In this paper we study AER in a single-modality setting using three families of baselines: (i) a classical embedding-based multi-label classifier built on frozen representations and Logistic Regression, (ii) a strong discriminative multiple-choice model based on DeBERTa-v3, and (iii) knowledge-graph–augmented LLMs that leverage a COMET-enhanced causal KG in either Euclidean (RotatE) or hyperbolic (RotH) geometry. On the development set, these systems obtain scores of approximately 0.73 (DeBERTa-v3), 0.61 (Euclidean KG–LLM), and 0.61 (hyperbolic KG–LLM). The results show that the discriminative DeBERTa-v3 baseline remains the strongest overall, COMET-based KG–LLM methods substantially narrow the gap, and moving from Euclidean to hyperbolic KG embeddings yields only marginal improvements under our current training regime.

# Methodology: Single-Modality Baselines

## Baseline 1: Classical Embedding-Based Classifier

Baseline 1 uses fixed text (and optionally image) embeddings with a shallow multi-label classifier. For each AER instance, we compute frozen encoder embeddings for the event and its associated documents (with image features concatenated when available), aggregate them into a single feature vector, and train a scikit-learn multi-output Logistic Regression model to predict the set of correct options. This provides a lightweight, non-LLM baseline against which the stronger neural baselines can be compared.

## Baseline 2: UnifiedQA + DeBERTa-v3 (Discriminative MCQA)

Baseline 2 uses a two-branch pipeline on a shared preprocessed representation. For each AER instance, we aggregate a small set of retrieved documents, concatenate title and body into a context string, and form a natural-language question with four options (A–D). This is serialized both as a UnifiedQA-style text-to-text input and as a SWAG-style multiple-choice record with `sent1`, `ending0`–`ending3`, a single primary label, and a full multi-label set.

The UnifiedQA branch treats AER as text generation: a T5-based UnifiedQA model encodes the unified input and directly decodes a string of option letters (e.g., “A,B”) without task-specific fine-tuning. The DeBERTa-v3 branch formulates AER as multiple-choice classification: a `microsoft/deberta-v3-base` encoder scores each (question, option) pair and is fine-tuned with cross-entropy over the primary label index, while multi-label information is preserved for evaluation. Both branches produce option sets that are scored with the official SemEval metric.

## Baseline 3: Knowledge-Graph–Enhanced LLM QA

Baseline 3 augments LLMs with a causal knowledge graph built from the training split. Target events and candidate options are collected as seed nodes, and a COMET-ATOMIC 2020 model (or a rule-based fallback) generates causal hypotheses (`Causes`, `isBefore`, `xEffect`, `oEffect`, `isAfter`), which are stored as directed triples in a multi-relational `CausalKnowledgeGraph`. A RotatE embedding model is trained on these triples to obtain dense entity representations for later retrieval.

At inference time, the system answers questions by combining KG-derived knowledge with an LLM such as DeepSeek-chat. For each instance, we retrieve relevant triples by lexical overlap and/or nearest neighbours in embedding space, convert them into short causal statements, and use a `KGAugmentedPrompt` to concatenate: (i) the causal context, (ii) a compressed document view, and (iii) the multiple-choice question with options. This prompt is fed to the LLM, which is instructed to output only option labels; the decoded labels are mapped back to sets and evaluated with the official metric.

## Baseline 4: Hyperbolic KG–Enhanced LLM QA

Baseline 4 keeps the same COMET-based causal graph construction as Baseline 3 but replaces Euclidean RotatE embeddings with hyperbolic representations better suited to hierarchical structure. Entities are embedded on a Lorentz-model hyperboloid and relations are parameterized as transformations in this space; training uses negative sampling and a ranking-style loss defined with hyperbolic geodesic distance, implemented via Geoopt.

The downstream QA pipeline mirrors Baseline 3 and treats COMET-generated causal text as the primary knowledge signal. For each instance, we first obtain COMET-based causal snippets for the event and options; optionally, we query the hyperbolic embeddings via k-nearest-neighbour search to retrieve nearby entities and short causal paths and append them to the COMET text. The resulting knowledge-augmented prompt, together with document snippets and the multiple-choice question, is passed to DeepSeek-chat, whose option predictions are again scored with the official SemEval metric.

## Experimental Setup

### Data Splits and Usage

We follow the official SemEval-2026 Task 12 data splits: sample (200 instances, 10 topics), train (1,819 instances, 36 topics), dev (400 instances, 36 topics), and test (612 instances, 20 topics). All baselines consume the organizer-provided JSONL/JSON files (`questions.jsonl` and `docs.json`) without changing labels or topics.

Baseline 1 trains and validates the classical classifier on the train and dev splits, then applies the selected model once to test to generate a submission file. For Baseline 2, UnifiedQA is applied zero-shot to the preprocessed dev and test splits; we report metrics on dev and submit test predictions to Codabench. The DeBERTa-v3 branch trains on the processed train split and uses dev for model selection; the selected checkpoint is then run once on test to produce the final submission. For Baseline 3, we use the full train split to build the causal KG (events and options as entities) and run the KG–LLM QA pipeline on dev and test. As for Baseline 2, dev is used for analysis, while test remains unlabeled and is only scored via Codabench.

### Preprocessing and Hyperparameters

For Baseline 1, we pre-compute fixed Qwen-style text embeddings (and SigLIP image embeddings when used), standardize the concatenated features with `StandardScaler`, and fit a scikit-learn multi-output Logistic Regression classifier using the default regularization and solver settings.

For Baseline 2, a preprocessor converts question–document pairs to UnifiedQA and MCQA formats. Each instance aggregates up to five retrieved documents (`max_doc_count = 5`); titles and contents are concatenated and truncated to 2,048 characters (`max_context_length = 2048`). The output includes (i) UnifiedQA-style text-to-text inputs (context + question + four options) and (ii) SWAG-style records with `sent1`, `ending0`–`ending3`, `label`, and `labels_all`. UnifiedQA models (e.g., `allenai/unifiedqa-v2-t5-base-1363200`) use standard text-to-text generation settings, maximum input length 512–768 tokens, and batch size 4–8. The DeBERTa branch fine-tunes `microsoft/deberta-v3-base` with max sequence length 256 per (question, option) pair, batch size 4, 3 epochs, learning rate 2×10⁻⁵, warmup ratio 0.1, gradient clipping 1.0, AdamW optimizer, and a linear schedule.

For Baseline 3, we load `questions.jsonl` and `docs.json` as above and additionally build a causal KG from the train split. When COMET-ATOMIC 2020 is available, we use a BART-based COMET model (`model_name = "mismayil/comet-bart-ai2"`, `max_length = 64`, `num_beams = 5`, `num_return_sequences = 5`) to generate `Causes`, `isBefore`, `xEffect`, `oEffect`, and `isAfter` relations, and convert them into triples (cause, relation, effect). We train a RotatE KGE model with embedding dimension 256, learning rate 1×10⁻³, batch size 256, 100 epochs, 10 negative samples per positive triple, and a margin close to 9.0, optimizing a self-adversarial negative-sampling loss with Adam. The resulting embeddings and checkpoints are reused across QA runs. During QA we limit retrieved KG triples per instance (tens of triples) and truncate KG-derived text to about 400–500 tokens so that KG context, document snippets, and the multiple-choice question fit within the LLM context window.

For Baseline 4, we keep the same KG construction but replace Euclidean embeddings with Lorentz-model hyperbolic embeddings implemented via Geoopt. In the reported configuration we use a RotH model that embeds entities on a Lorentzian hyperboloid with dimension 32 and Lorentz geodesic distance \(d_L(u,v) = \operatorname{arcosh}(-\langle u,v \rangle_L)\). Training runs for up to 60 epochs with the RiemannianAdam optimizer (learning rate 5×10⁻⁴, batch size 256, 10 negative samples per positive triple), using the same negative-sampling configuration as RotatE and an early-stopping safeguard when the loss becomes numerically unstable. For the Euclidean vs. hyperbolic retrieval ablation in the Results section, we additionally train a 32-dimensional RotatE model and a matching RotH model for 30 epochs each on the same COMET-enhanced KG.

### Libraries and Tooling

Experiments run in a Conda environment based on Python 3.10 (`dl310`) on a Windows 11 machine with a single NVIDIA GPU. Neural models use PyTorch 2.10.0 and Transformers 4.36.2 for encoder–decoder (UnifiedQA, COMET) and encoder-only (DeBERTa-v3, RoBERTa) architectures. UnifiedQA and DeBERTa-v3 are loaded from the Hugging Face Model Hub (https://huggingface.co), and COMET-ATOMIC 2020 from `mismayil/comet-bart-ai2`. KGE models in Baseline 3 and 4 are implemented in PyTorch; hyperbolic embeddings rely on Geoopt 0.5.1 (https://github.com/geoopt/geoopt) and its Lorentz manifold and RiemannianAdam optimizer. LLM calls use the official Python SDKs for OpenAI (openai 2.15.0) and Anthropic (anthropic 0.76.0), plus huggingface-hub 0.36.0 where needed. The full dependency list is recorded in `pip_freeze_main.txt`.

### Evaluation Metrics

We use the official instance-level metric from SemEval-2026 Task 12. Let G be the set of gold options and P the predicted set. The instance score is 1.0 if P = G, 0.5 if P is a non-empty proper subset of G, and 0.0 otherwise (including empty P or any incorrect option). The final score is the average over all instances. We also report exact-match, partial-match, and error rates; no alternative metrics are used.

## Results

### Dev-Set Comparison: Baseline 2 vs Baseline 3

Table 1 reports the main results on the official development split (`dev_data`, 400 instances) for Baseline 2 and Baseline 3.

| System                                   | Backbone / LLM      | Split                | Score  | Exact Match | Partial Match | Error Rate |
|------------------------------------------|---------------------|----------------------|--------|------------:|-------------:|----------:|
| **Baseline 2 (MCQA)**                   | DeBERTa v3 base     | dev_data (N = 400)   | **0.7250** |  **49.5%** |   **46.0%** | **4.5%**  |
| Baseline 3 (KG + Haiku, no COMET)       | claude-3-haiku      | dev_data (N = 400)   | 0.4625 |      33.25% |       26.0%  |   40.75%  |
| Baseline 3 (KG + DeepSeek + COMET, best)| DeepSeek-chat       | dev_data (N = 400)   | 0.6088 |      46.25% |       29.25% |   24.50%  |

The DeBERTa v3 base classifier (Baseline 2) achieves the best dev-set score, with high exact and partial match rates and a very low error rate. Among the KG–LLM systems, switching from Haiku to DeepSeek-chat and enabling COMET-based KG expansion yields a strong gain over the Haiku + KG configuration, and substantially narrows the gap to Baseline 2.

### Baseline 3 Ablations

For Baseline 3 we perform controlled comparisons on `dev_data` (400 instances) to study the effect of different KG and knowledge sources. Table 2 summarizes the key variants.

| Variant ID | LLM           | KG Source / Embedding          | Dev Score | Exact Match | Partial Match | Error Rate |
|-----------:|---------------|---------------------------------|----------:|-----------:|-------------:|----------:|
| (a)        | DeepSeek-chat | Simple KG (rule-based), no COMET | 0.5713   |   40.75%   |    32.75%    |  26.50%   |
| (b)        | DeepSeek-chat | RotatE, Simple KG only         | 0.5787   |   42.00%   |    31.75%    |  26.25%   |
| (c)        | DeepSeek-chat | RotatE, COMET-enhanced KG      | **0.6088** | **46.25%** |   29.25%    | **24.50%** |

Variant (b) shows that adding RotatE embeddings on top of the simple KG brings a modest gain over the pure symbolic KG in (a). The COMET-enhanced KG in (c) further increases the score from 0.5787 to 0.6088 (+5.2% relative), raises exact match from 42.00% to 46.25%, and reduces the error rate from 26.25% to 24.50%, confirming that denser commonsense knowledge and learned embeddings are both important. The DeepSeek-chat + KG + COMET configuration in row (c) corresponds to the “historical best” setup reported in the baseline code; it was obtained after fixing a COMET model-loading issue by enabling `use_safetensors=True` and should be regarded as a post-submission variant.

### Baseline 4 Results

We next evaluate Baseline 4, which keeps the COMET-enhanced causal KG and DeepSeek-chat LLM but replaces the Euclidean RotatE embeddings of Baseline 3 with hyperbolic RotH embeddings trained on the same graph. At inference time we use COMET-based causal text as the primary KG signal (prompt fusion) and treat the hyperbolic KG mainly as a structural backbone that could support retrieval-style ablations.

Table 3 compares the best Baseline 3 configuration with the final Baseline 4 system on the dev split.

| System                                   | Backbone / LLM      | Split                | Score  | Exact Match | Partial Match | Error Rate |
|------------------------------------------|---------------------|----------------------|--------|------------:|-------------:|----------:|
| Baseline 3 (KG + DeepSeek + COMET, best)| DeepSeek-chat       | dev_data (N = 400)   | 0.6088 |      46.25% |       29.25% |   24.50%  |
| **Baseline 4 (Hyperbolic KG + DeepSeek)** | DeepSeek-chat     | dev_data (N = 400)   | **0.6100** |   **46.75%** |     28.50%  | **24.75%** |

Baseline 4 slightly improves the overall dev score over the best Baseline 3 configuration (0.6100 vs. 0.6088), with a marginally higher exact-match rate and a very similar error rate. This suggests that, once a strong COMET-enhanced prompt is in place, switching the underlying KG representation from Euclidean RotatE to hyperbolic RotH has at best a modest effect on end-task accuracy under the current training budget.

#### Euclidean vs. Hyperbolic KGE (Retrieval Ablation)

To more directly probe the effect of the embedding geometry, we also run a retrieval-only KG–LLM QA variant in which COMET is disabled and the KG context is constructed purely from nearest-neighbour entities in the learned embeddings. Using the same COMET-enhanced KG built from `train_data`, we train a Euclidean RotatE model and a hyperbolic RotH model with matched dimensionality (32) and batch size, and then evaluate DeepSeek-chat with retrieval-style fusion on `dev_data`.

| Variant ID | LLM           | Fusion      | KG Embedding | Dev Score | Exact Match | Partial Match | Error Rate |
|-----------:|---------------|------------|-------------|----------:|-----------:|-------------:|----------:|
| (d)        | DeepSeek-chat | retrieval  | RotatE (Euclidean) | 0.3150   |   17.25%   |    28.50%    |  54.25%   |
| (e)        | DeepSeek-chat | retrieval  | RotH (Hyperbolic)  | 0.2988   |   16.75%   |    26.25%    |  57.00%   |

Both geometries perform substantially worse than the prompt-based COMET variants in Tables 1 and 3, confirming that, in this setup, high-quality textualized causal knowledge is far more important than the choice between Euclidean and hyperbolic embedding spaces. The gap between RotatE and RotH in rows (d)–(e) is small, and does not provide strong evidence that hyperbolic retrieval alone can close the performance gap to the COMET-augmented systems.

### Competition Submission and Ranking

For both Baseline 2 and Baseline 3, the trained models are used to generate predictions on the official test split (`test_data`, 612 instances), which are submitted to the Codabench competition server for scoring. As the repository does not record a snapshot of the final Codabench leaderboard, we do not report an exact rank here and instead focus on the dev-set results in Tables 1–2. In all experiments discussed above, dev-set metrics are computed on `dev_data` (400 instances) with the official SemEval-2026 Task 12 scoring rule.

## Conclusion

We investigated abductive event reasoning in a single-modality setting using three progressively stronger baselines. A classical embedding-based classifier (Baseline 1) provides a lightweight reference point but is clearly outperformed by a fine-tuned DeBERTa-v3 multiple-choice model (Baseline 2), which achieves the best overall dev-set score. Building a COMET-enhanced causal knowledge graph and prompting LLMs with structured causal context (Baseline 3) substantially narrows the gap to the discriminative baseline, demonstrating the value of external commonsense knowledge for this task. Extending the KG with hyperbolic RotH embeddings (Baseline 4) yields only a marginal improvement over the best Euclidean KG–LLM variant, and retrieval-only KGE ablations perform far below COMET-based prompts. Overall, our results suggest that high-quality textualized causal knowledge and strong discriminative MCQA backbones remain the primary drivers of AER performance, while more expressive KG geometries offer limited gains under current training budgets.
