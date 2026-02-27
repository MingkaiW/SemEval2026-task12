# Baseline 4: Hyperbolic KG–Enhanced LLM QA

This baseline extends the KG–enhanced LLM framework of Baseline 3 by replacing the Euclidean knowledge graph (KG) embeddings with hyperbolic representations. The goal is to better capture hierarchical and multi-hop causal structure in the event graph, and to exploit the geometry of hyperbolic space when retrieving and aggregating causal evidence for abductive event reasoning.

---

## Motivation

The causal knowledge graph constructed in Baseline 3 is inherently hierarchical: high-level macro events branch into more specific sub-events and fine-grained consequences. Euclidean embeddings such as TransE, ComplEx, or RotatE can approximate this structure, but they struggle to represent trees and DAGs with low distortion in moderate dimensions. Hyperbolic spaces (e.g., Lorentz or Poincaré models) are known to embed hierarchies with exponentially expanding volume, allowing them to represent multi-level causality more compactly. Baseline 4 therefore keeps the same KG construction and LLM interface as Baseline 3, but replaces the KG encoder with a hyperbolic embedding model and makes retrieval decisions in hyperbolic space.

---

## Model Overview

Baseline 4 follows the same two-phase structure as Baseline 3:

1. **Offline KG construction and embedding**
   - Build a directed causal KG over events and options using COMET-ATOMIC 2020 and/or a rule-based causal knowledge base.
   - Embed entities and relations into a hyperbolic manifold (e.g., Lorentz model) using a dedicated KG embedding objective.

2. **Online KG-augmented LLM question answering**
   - For each AER instance at evaluation time, retrieve hyperbolically nearest neighbours and causal paths in the KG.
   - Convert the retrieved hyperbolic knowledge into natural-language snippets and integrate them into the LLM prompt, as in Baseline 3.
   - Let the LLM perform multiple-choice abductive reasoning and map the output back to option labels.

The key difference from Baseline 3 is that similarity and neighbourhoods are computed using hyperbolic distances and geodesics instead of Euclidean norms, which biases retrieval towards nodes that are hierarchically close in the causal tree.

---

## Phase 1: Causal KG Construction (Shared with Baseline 3)

The KG construction procedure is identical to Baseline 3:

- **Event collection**: gather all target events and candidate options from the training split, treating each unique text as a candidate KG entity.
- **Commonsense expansion**: for each event, query COMET-ATOMIC 2020 (when available) to generate causal relations such as `Causes`, `isBefore`, `xEffect`, `oEffect`, and `isAfter`. When COMET cannot be used, fall back to a rule-based simple causal knowledge base that maps domain-specific patterns (economic, political, disaster, etc.) to template-based causes and effects.
- **Triple generation**: wrap the generated knowledge in `KnowledgeEnhancedEvent` objects and add directed triples to a `CausalKnowledgeGraph`. For example, if COMET suggests that "economic recession" is a possible cause of "unemployment rises", we insert the triple ("economic recession", `causes`, "unemployment rises"). Likewise, predicted effects form forward causal edges.
- **Graph statistics**: record the number of entities, relations, and triples, and optionally compare a small “simple KG” against a COMET-expanded KG to quantify how much the causal space is densified.

The output of this phase is a multi-relational KG `G = (V, R, E)` where `V` is the set of events, `R` is the set of relation types (e.g., `causes`, `is_caused_by`, `precondition_of`), and `E` is the set of directed triples.

---

## Phase 2: Hyperbolic KG Embedding

In Baseline 4 we embed the causal KG into a hyperbolic manifold instead of a Euclidean vector space. We consider Lorentz-model hyperbolic embeddings, but the design is compatible with other hyperbolic parametrizations.

### 2.1 Hyperbolic Entity and Relation Representations

- **Entity embeddings**: each node `v ∈ V` is mapped to a point `x_v` on the hyperboloid in `R^{d+1}` satisfying the Lorentz norm constraint. This representation allocates more volume near the boundary, which is suitable for representing leaves and deep branches of a causal hierarchy.
- **Relation embeddings**: each relation type `r ∈ R` is represented as a transformation on the manifold (e.g., a Lorentz isometry or a combination of translation and scaling), analogous to how TransE or RotatE treat relations in Euclidean space but now respecting the curvature.

### 2.2 Training Objective

Training proceeds over mini-batches of triples `(h, r, t)` sampled from `E`:

1. **Negative sampling**: for each positive triple, generate corrupted triples by replacing the head or tail with a randomly selected entity while keeping the relation fixed.
2. **Hyperbolic scoring**: define a score function based on the geodesic distance between the transformed head and the tail. For example, we can encourage `d_H(f_r(x_h), x_t)` to be small for true triples and large for corrupted ones, where `d_H` is the Lorentzian geodesic distance and `f_r` is the relation transformation.
3. **Loss function**: use a margin-based ranking loss or a self-adversarial negative sampling loss adapted to hyperbolic distances. The objective encourages the model to place causally related events close in hyperbolic space along appropriate directions while pushing unrelated events apart.
4. **Optimization**: train with Riemannian stochastic gradient descent or an equivalent optimizer that respects the manifold constraints, periodically projecting updated embeddings back onto the hyperboloid.

After training, we export hyperbolic entity embeddings and (optionally) relation parameters. These embeddings are later used to measure hyperbolic similarity between events and to drive structure-aware retrieval.

---

## Phase 3: Hyperbolic KG–Augmented QA with LLMs

The online QA phase mirrors Baseline 3 but replaces Euclidean retrieval with hyperbolic neighbourhood queries.

### 3.1 Hyperbolic Retrieval

For each evaluation instance consisting of a target event and four candidate options:

1. **Locate nodes**: map the textual event and each option to their corresponding KG entities (creating on-the-fly nodes if necessary and approximating embeddings when unseen).
2. **Neighbourhood search**: using the hyperbolic embeddings, perform k-nearest-neighbour search around the target event, the options, or both. Distances are computed with the hyperbolic metric `d_H`, and we retain the top-k most relevant neighbours.
3. **Path extraction**: for the retrieved neighbours, inspect the underlying graph structure to extract short causal paths such as `candidate_cause → ... → target_event` or `target_event → ... → downstream_effect`. These paths highlight potential direct causes and multi-hop causal chains.

### 3.2 Prompt Construction

We then convert the retrieved hyperbolic knowledge into natural language and feed it to a large language model:

- **Causal context rendering**: summarize the nearest neighbours and paths into textual statements, for example: "Hyperbolic KG suggests that X is a typical cause of events like Y" or "Events of type A often precede events of type B". The aim is to surface local causal structure discovered by the hyperbolic model.
- **Document integration**: as in Baseline 3, we compress the original retrieved documents into a short context paragraph containing the most salient sentences.
- **Multiple-choice question**: we append the formatted AER question and its four answer options, and include explicit instructions asking the LLM to choose the most plausible *direct* cause(s) based on both the textual documents and the KG-derived causal hints.

The resulting prompt is sent to an LLM (e.g., GPT-4o-mini, DeepSeek-chat, or another API-compatible model). The model’s textual output is parsed into a set of option letters {A, B, C, D}, which is then evaluated using the official 0 / 0.5 / 1.0 scoring scheme.

---

## Evaluation Protocol and Expected Benefits

Baseline 4 is evaluated on the same development and test splits and with the same instance-level metric as the previous baselines. Experimental comparisons focus on three aspects:

1. **Effect of hyperbolic geometry**: compare hyperbolic KG embeddings against Euclidean TransE/ComplEx/RotatE variants under identical KG construction and LLM settings, isolating the benefit of better hierarchical modelling.
2. **Retrieval quality**: measure how often hyperbolic neighbourhoods surface truly causal nodes or paths that connect the correct option to the target event, relative to Euclidean kNN.
3. **End-to-end AER performance**: report improvements (or regressions) in overall Score, exact match rate, and partial match rate when replacing Euclidean KG embeddings with hyperbolic ones in the KG-augmented prompt pipeline.

We expect hyperbolic embeddings to more faithfully encode the tree-like and multi-level nature of the causal KG, yielding more informative neighbourhoods and causal chains. When these signals are surfaced to the LLM via carefully crafted prompts, they should further strengthen the model’s abductive event reasoning beyond what is achieved with Euclidean KG embeddings alone.
