"""
SemEval 2026 Task 12: Abductive Event Reasoning
Baseline 4: Hyperbolic KG–Enhanced LLM QA 统一运行脚本

本脚本在 Baseline 3 的基础上，固定使用 COMET 构建因果知识图谱，
并采用 Lorentz/Hyperbolic KG Embedding（如 RotH, LorentzKG），
以 DeepSeek-chat 作为默认 LLM，实现 Baseline 4 全流程：

1. 构建 + 训练双曲 KG 嵌入:
   python run_baseline4.py build-kg \
       --data-path ../../train_data \
       --output-dir ./kg_output_hyper \
       --kg-model RotH --train-embedding --use-comet

2. 在 dev/test 上运行 Hyperbolic KG + LLM QA:
   python run_baseline4.py qa \
       --data-path ../../dev_data \
       --fusion prompt \
       --kg-path ./kg_output_hyper \
       --llm-type openai \
       --llm-model deepseek-chat \
       --api-base https://api.deepseek.com \
       --api-key YOUR_API_KEY \
       --output results_baseline4_dev.json \
       --submission-file submission_baseline4_dev.jsonl
"""

import argparse
import os
import sys
from pathlib import Path

# 将 single_modality 及 baseline3 目录加入 sys.path，便于复用现有模块
PROJECT_ROOT = Path(__file__).parent.parent.parent
SINGLE_MODALITY_ROOT = Path(__file__).parent.parent
BASELINE3_ROOT = SINGLE_MODALITY_ROOT / "baseline3"

sys.path.append(str(SINGLE_MODALITY_ROOT))
sys.path.append(str(BASELINE3_ROOT))


def build_knowledge_graph(args):
    """构建双曲 KG 并训练 Hyperbolic/Lorentz 嵌入 (RotH / LorentzKG 等)"""
    from baseline.data_loader import AERDataLoader
    from kg_embedding import KGEmbeddingTrainer, KGEConfig
    from comet_knowledge import build_event_knowledge_graph

    print("=" * 60)
    print("Building Hyperbolic Causal Knowledge Graph (Baseline 4)")
    print("=" * 60)

    # 加载数据
    loader = AERDataLoader(args.data_path)
    instances = loader.load()

    if args.max_samples:
        instances = instances[: args.max_samples]

    print(f"Loaded {len(instances)} instances")

    # 收集所有事件（目标事件 + 选项）
    events = set()
    for inst in instances:
        events.add(inst.target_event)
        for opt_text in inst.options.values():
            events.add(opt_text)

    events = list(events)
    print(f"Collected {len(events)} unique events")

    # 使用 COMET 构建因果知识图谱（Baseline 4 默认使用 COMET）
    enhanced_events, kg = build_event_knowledge_graph(
        events,
        use_comet=args.use_comet,
    )

    print("\nKnowledge Graph Statistics:")
    print(f"  Entities: {kg.num_entities}")
    print(f"  Relations: {kg.num_relations}")
    print(f"  Triples: {len(kg.triples)}")

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 保存知识图谱
    kg_path = os.path.join(args.output_dir, "knowledge_graph.json")
    kg.save(kg_path)
    print(f"\nKnowledge graph saved to: {kg_path}")

    # 训练 Hyperbolic/Lorentz KG embedding
    if args.train_embedding and len(kg.triples) > 0:
        print(f"\nTraining hyperbolic KG embeddings with model: {args.kg_model} ...")
        print(f"  embedding_dim={args.embedding_dim}, epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}")

        # Baseline 4 默认使用低维双曲空间，比如 dim=32，学习率适当调小以提升数值稳定性
        config = KGEConfig(
            embedding_dim=args.embedding_dim,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
        )

        trainer = KGEmbeddingTrainer(kg, args.kg_model, config)
        results = trainer.train()

        # 保存模型和嵌入
        model_path = os.path.join(args.output_dir, "kg_model.pt")
        trainer.save_model(model_path)

        emb_path = os.path.join(args.output_dir, "embeddings.npz")
        trainer.save_embeddings(emb_path)

        print("\nTraining complete!")
        print(f"  Final loss: {results['final_loss']:.4f}")
        print(f"  Model saved: {model_path}")
        print(f"  Embeddings saved: {emb_path}")
    else:
        if len(kg.triples) == 0:
            print("\nNo triples to train on. Skipping embedding training.")
        else:
            print("\nSkipping embedding training (use --train-embedding to enable)")

    return kg


def run_qa(args):
    """运行 Baseline 4 的 KG-Enhanced LLM QA 评估"""
    from kg_llm_qa import run_kg_llm_baseline

    print("=" * 60)
    print("Hyperbolic KG–Enhanced LLM QA Evaluation (Baseline 4)")
    print("=" * 60)

    # Baseline 4 默认使用 COMET 增强 KG
    results = run_kg_llm_baseline(
        data_path=args.data_path,
        fusion_method=args.fusion,
        llm_type=args.llm_type,
        llm_model=args.llm_model,
        use_comet=args.use_comet,
        output_path=args.output,
        submission_path=args.submission_file,
        max_samples=args.max_samples,
        api_base=args.api_base,
        api_key=args.api_key,
    )

    return results


def run_kge_qa(args):
    """使用预训练好的 KG Embedding (Euclidean / Hyperbolic) 做检索式 QA，用于公平对比。

    要求 args.kg_dir 下至少包含:
      - knowledge_graph.json
      - embeddings.npz (由 RotatE 或 RotH 等模型训练得到)
    """
    from baseline.data_loader import AERDataLoader
    from kg_embedding import CausalKnowledgeGraph
    from kg_llm_qa import KGLLMConfig, KGLLMQA
    import numpy as np

    print("=" * 60)
    print("KG-Embedding Retrieval QA (Baseline 3/4 Fair Comparison)")
    print("=" * 60)

    # 1. 加载数据
    loader = AERDataLoader(args.data_path)
    instances = loader.load()

    if args.max_samples:
        instances = instances[: args.max_samples]

    print(f"Loaded {len(instances)} instances from {args.data_path}")

    # 2. 加载 KG
    kg_dir = args.kg_dir
    kg_path = os.path.join(kg_dir, "knowledge_graph.json")
    emb_path = os.path.join(kg_dir, "embeddings.npz")

    if not os.path.exists(kg_path):
        raise FileNotFoundError(f"knowledge_graph.json not found in {kg_dir}")
    if not os.path.exists(emb_path):
        raise FileNotFoundError(f"embeddings.npz not found in {kg_dir}")

    kg = CausalKnowledgeGraph()
    kg.load(kg_path)
    print(f"Loaded KG from {kg_path}: {kg.num_entities} entities, {kg.num_relations} relations, {len(kg.triples)} triples")

    emb_npz = np.load(emb_path)
    kg_embeddings = {k: emb_npz[k] for k in emb_npz.files}
    example_key = next(iter(kg_embeddings))
    emb_dim = kg_embeddings[example_key].shape[-1]
    print(f"Loaded KG embeddings from {emb_path}: {len(kg_embeddings)} entities, dim={emb_dim}")

    # 3. 配置 DeepSeek-chat LLM + KG Embedding Retrieval
    config = KGLLMConfig(
        fusion_method="retrieval",         # 只使用检索式融合
        llm_type="openai",
        llm_model=args.llm_model,
        # COMET 知识已经编码进 KG 中，这里不再额外调用 COMET 生成
        use_comet=False,
        use_kg_embedding=True,
        kg_embedding_dim=emb_dim,
        kg_model_type=args.kg_model_type,
        api_base=args.api_base,
        api_key=args.api_key,
    )

    qa_system = KGLLMQA(config)
    qa_system.setup(kg=kg, kg_embeddings=kg_embeddings)

    # 4. 评估
    results = qa_system.evaluate(instances, args.output, args.submission_file)

    print("\n=== KG-Embedding Retrieval QA Results ===")
    print(f"KG Model Type: {args.kg_model_type}")
    print(f"LLM: {args.llm_model} (provider: {config.llm_type}, api_base={config.api_base})")
    print(f"Score: {results['score']:.4f}")
    print(f"Exact Match: {results['exact_match_rate']:.4f}")
    print(f"Partial Match: {results['partial_match_rate']:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Baseline 4: Hyperbolic KG–Enhanced LLM QA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:

1. 构建双曲知识图谱 (使用 COMET + RotH, dim=32):
   python run_baseline4.py build-kg \\
       --data-path ../../train_data \\
       --output-dir ./kg_output_hyper \\
       --use-comet \\
       --train-embedding \\
       --kg-model RotH \\
       --embedding-dim 32

2. 在 dev 集上运行 QA (DeepSeek-chat + Hyperbolic KG, prompt 融合):
   python run_baseline4.py qa \\
       --data-path ../../dev_data \\
       --fusion prompt \\
       --kg-path ./kg_output_hyper \\
       --llm-type openai \\
       --llm-model deepseek-chat \\
       --api-base https://api.deepseek.com \\
       --api-key YOUR_API_KEY \\
       --output results_baseline4_dev.json \\
       --submission-file submission_baseline4_dev.jsonl
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # build-kg 命令（专注于双曲 / Lorentz KG 模型）
    build_parser = subparsers.add_parser("build-kg", help="构建双曲知识图谱并训练KG embedding")
    build_parser.add_argument("--data-path", type=str, required=True, help="数据目录路径 (train_data)")
    build_parser.add_argument("--output-dir", type=str, default="./kg_output_hyper", help="输出目录")
    build_parser.add_argument("--use-comet", action="store_true", help="使用 COMET 生成知识 (Baseline 4 推荐开启)")
    build_parser.add_argument("--train-embedding", action="store_true", help="训练 KG embedding")
    build_parser.add_argument(
        "--kg-model",
        type=str,
        default="RotH",
        choices=["RotH", "LorentzKG"],
        help="Hyperbolic/Lorentz KG embedding 模型 (RotH 或 LorentzKG)",
    )
    build_parser.add_argument("--embedding-dim", type=int, default=32, help="嵌入维度 (双曲模型推荐 32)")
    build_parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    build_parser.add_argument("--batch-size", type=int, default=256, help="batch 大小")
    build_parser.add_argument("--lr", type=float, default=5e-4, help="KG embedding 学习率 (双曲模型推荐 5e-4 或更小)")
    build_parser.add_argument("--max-samples", type=int, default=None, help="最多使用多少训练样本 (调试用)")

    # qa 命令
    qa_parser = subparsers.add_parser("qa", help="运行 Hyperbolic KG–Enhanced LLM QA 评估")
    qa_parser.add_argument("--data-path", type=str, required=True, help="数据目录路径 (dev_data 或 test_data)")
    qa_parser.add_argument(
        "--fusion",
        type=str,
        default="prompt",
        choices=["prompt", "retrieval"],
        help="KG 与文档信息的融合方式",
    )
    qa_parser.add_argument(
        "--llm-type",
        type=str,
        default="openai",
        choices=["openai", "anthropic"],
        help="LLM 提供方类型",
    )
    # 默认使用 DeepSeek-chat，方便复现实验
    qa_parser.add_argument("--llm-model", type=str, default="deepseek-chat", help="LLM 模型名称")
    qa_parser.add_argument("--use-comet", action="store_true", help="是否在 QA 阶段启用 COMET 知识增强 (与 build-kg 对齐)")
    qa_parser.add_argument("--kg-path", type=str, default=None, help="KG 输出目录 (包含 knowledge_graph.json / kg_model.pt / embeddings.npz)")
    qa_parser.add_argument("--output", type=str, default=None, help="评估结果保存路径 (JSON)")
    qa_parser.add_argument("--max-samples", type=int, default=None, help="最多评估多少样本 (调试用)")
    qa_parser.add_argument("--api-base", type=str, default=None, help="自定义 API 端点 (如 DeepSeek: https://api.deepseek.com)")
    qa_parser.add_argument("--api-key", type=str, default=None, help="API 密钥 (优先于环境变量)")
    qa_parser.add_argument("--submission-file", type=str, default=None, help="Codabench 提交文件保存路径 (JSONL)")

    # kge-qa 命令: 使用同一 KG、不同嵌入 (RotatE vs RotH) 做检索式 QA，对比 Euclidean vs Hyperbolic
    kge_parser = subparsers.add_parser("kge-qa", help="使用 KG Embedding Retrieval 的 DeepSeek-chat QA (对比 Euclidean vs Hyperbolic)")
    kge_parser.add_argument("--data-path", type=str, required=True, help="数据目录路径 (dev_data 或 test_data)")
    kge_parser.add_argument("--kg-dir", type=str, required=True, help="KG 目录 (包含 knowledge_graph.json / embeddings.npz)")
    kge_parser.add_argument(
        "--kg-model-type",
        type=str,
        default="RotatE",
        choices=["TransE", "ComplEx", "RotatE", "RotH", "LorentzKG"],
        help="训练该 embeddings 的 KG 模型类型，用于结果记录",
    )
    kge_parser.add_argument("--llm-model", type=str, default="deepseek-chat", help="LLM 模型名称 (默认 DeepSeek-chat)")
    kge_parser.add_argument(
        "--api-base",
        type=str,
        default="https://api.deepseek.com",
        help="DeepSeek 或其他 OpenAI-兼容端点 (默认 https://api.deepseek.com)",
    )
    kge_parser.add_argument("--api-key", type=str, default=None, help="API 密钥 (优先于环境变量)")
    kge_parser.add_argument("--output", type=str, default=None, help="评估结果保存路径 (JSON)")
    kge_parser.add_argument("--submission-file", type=str, default=None, help="Codabench 提交文件保存路径 (JSONL)")
    kge_parser.add_argument("--max-samples", type=int, default=None, help="最多评估多少样本 (调试用)")

    args = parser.parse_args()

    if args.command == "build-kg":
        build_knowledge_graph(args)
    elif args.command == "qa":
        run_qa(args)
    elif args.command == "kge-qa":
        run_kge_qa(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
