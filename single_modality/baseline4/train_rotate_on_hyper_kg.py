"""Train a RotatE (Euclidean) KG embedding on the existing COMET-based KG
produced by Baseline 4 (kg_output_hyper/knowledge_graph.json).

This is used to fairly compare Euclidean (RotatE) vs Hyperbolic (RotH)
embeddings under the same KG, data, and LLM settings.
"""

import os
import sys
from pathlib import Path

# Reuse Baseline 3's KG embedding module by adding its directory to sys.path
PROJECT_ROOT = Path(__file__).parent.parent.parent
SINGLE_MODALITY_ROOT = Path(__file__).parent.parent
BASELINE3_ROOT = SINGLE_MODALITY_ROOT / "baseline3"

sys.path.append(str(BASELINE3_ROOT))

from kg_embedding import CausalKnowledgeGraph, KGEmbeddingTrainer, KGEConfig


def main():
    """Train RotatE embeddings with configurable hyperparameters.

    默认参数保持与最初实验一致：dim=32, epochs=30, batch_size=256。
    可以通过命令行覆盖这些参数，以便做更大规模的对比实验。
    """
    import argparse

    parser = argparse.ArgumentParser(description="Train RotatE on Baseline 4 COMET KG")
    parser.add_argument("--kg-dir", type=str, default="./kg_output_hyper", help="输入 KG 目录 (包含 knowledge_graph.json)")
    parser.add_argument("--out-dir", type=str, default="./kg_output_euclid", help="RotatE 输出目录")
    parser.add_argument("--embedding-dim", type=int, default=32, help="RotatE 嵌入维度")
    parser.add_argument("--epochs", type=int, default=30, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=256, help="batch 大小")

    args = parser.parse_args()

    root = Path(__file__).parent
    # 相对路径以当前脚本所在目录为基准
    hyper_dir = (root / args.kg_dir) if not os.path.isabs(args.kg_dir) else Path(args.kg_dir)
    euclid_dir = (root / args.out_dir) if not os.path.isabs(args.out_dir) else Path(args.out_dir)

    kg_path = hyper_dir / "knowledge_graph.json"
    if not kg_path.exists():
        raise FileNotFoundError(f"knowledge_graph.json not found in {hyper_dir}")

    euclid_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Training RotatE (Euclidean) embeddings on Baseline 4 COMET KG")
    print("=" * 60)
    print(f"Loading KG from: {kg_path}")

    kg = CausalKnowledgeGraph()
    kg.load(str(kg_path))

    print(f"KG: {kg.num_entities} entities, {kg.num_relations} relations, {len(kg.triples)} triples")

    config = KGEConfig(
        embedding_dim=args.embedding_dim,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
    )

    trainer = KGEmbeddingTrainer(kg, model_type="RotatE", config=config)
    results = trainer.train()

    print("\nTraining complete.")
    print(f"  Final loss: {results['final_loss']:.4f}")

    # Save KG (copy) and embeddings/model into euclidean output dir
    out_kg_path = euclid_dir / "knowledge_graph.json"
    kg.save(str(out_kg_path))
    print(f"Knowledge graph copied to: {out_kg_path}")

    model_path = euclid_dir / "kg_model.pt"
    emb_path = euclid_dir / "embeddings.npz"

    trainer.save_model(str(model_path))
    trainer.save_embeddings(str(emb_path))

    print(f"Model saved to: {model_path}")
    print(f"Embeddings saved to: {emb_path}")


if __name__ == "__main__":
    main()
