"""
SemEval 2026 Task 12: Abductive Event Reasoning
Training: GRPO (Group Relative Policy Optimization) using TRL

This script performs GRPO training on a SFT-pretrained model for causal reasoning.
Based on DeepSeek-R1's approach adapted for the causal reasoning task.

Usage:
    # Single GPU
    python run_grpo.py \
        --model ./output/causal_sft \
        --data ./data/causal_grpo_data.jsonl \
        --output ./output/causal_grpo

    # Multi-GPU with accelerate
    accelerate launch run_grpo.py \
        --model ./output/causal_sft \
        --data ./data/causal_grpo_data.jsonl \
        --output ./output/causal_grpo

Requirements:
    pip install trl>=0.8.0 peft transformers accelerate
"""

import os
import json
import argparse
import torch
from pathlib import Path
from typing import List, Dict, Any
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOTrainer, GRPOConfig


def load_grpo_dataset(data_path: str, max_samples: int = None) -> Dataset:
    """Load GRPO dataset from JSONL file"""
    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    if max_samples:
        data = data[:max_samples]

    return Dataset.from_list(data)


def causal_reasoning_reward(
    completions: List[str],
    prompts: List[str] = None,
    **kwargs
) -> List[float]:
    """
    Reward function for causal reasoning

    Evaluates model completions against golden answers.

    Args:
        completions: List of model-generated completions
        prompts: List of original prompts
        **kwargs: Additional metadata including golden_causes, golden_answers

    Returns:
        List of reward scores
    """
    rewards = []

    # Get metadata from kwargs
    golden_causes = kwargs.get('golden_causes', [])
    golden_answers = kwargs.get('golden_answers', [])

    for i, completion in enumerate(completions):
        completion_lower = completion.lower().strip()
        reward = 0.0

        # Check for MCQA format answers (A, B, C, D)
        if golden_answers and i < len(golden_answers):
            answers = golden_answers[i] if isinstance(golden_answers[i], list) else [golden_answers[i]]
            # Check if any golden answer is in completion
            for ans in answers:
                if ans.upper() in completion.upper()[:10]:  # Check first 10 chars
                    reward = 1.0
                    break

        # Check for cause text matching
        if reward == 0.0 and golden_causes and i < len(golden_causes):
            causes = golden_causes[i] if isinstance(golden_causes[i], list) else [golden_causes[i]]
            for cause in causes:
                cause_lower = cause.lower()
                # Full match
                if cause_lower in completion_lower:
                    reward = 1.0
                    break
                # Partial match (first 50 chars)
                elif cause_lower[:50] in completion_lower:
                    reward = 0.5
                # Keyword match
                else:
                    keywords = cause_lower.split()[:5]
                    matched = sum(1 for kw in keywords if kw in completion_lower and len(kw) > 4)
                    if matched >= 3:
                        reward = 0.3

        # Penalize empty or very short responses
        if len(completion.strip()) < 5:
            reward = -1.0

        rewards.append(reward)

    return rewards


def causal_mcqa_reward(
    completions: List[str],
    prompts: List[str] = None,
    golden_answers: List[List[str]] = None,
    **kwargs
) -> List[float]:
    """
    Reward function for MCQA format causal reasoning

    Args:
        completions: Model completions
        prompts: Original prompts
        golden_answers: List of lists of correct answer letters

    Returns:
        List of rewards
    """
    rewards = []

    if golden_answers is None:
        golden_answers = kwargs.get('golden_answers', [])

    for i, completion in enumerate(completions):
        completion_clean = completion.strip().upper()
        reward = 0.0

        if i < len(golden_answers):
            golden = golden_answers[i]
            if isinstance(golden, str):
                golden = [golden]

            # Extract predicted answers from completion
            predicted = set()
            for char in "ABCD":
                # Check if letter appears as answer
                if char in completion_clean[:20]:  # Check first 20 chars
                    predicted.add(char)

            golden_set = set(golden)

            if predicted == golden_set:
                reward = 1.0  # Exact match
            elif predicted and predicted.issubset(golden_set):
                reward = 0.5  # Partial match (subset)
            elif predicted & golden_set:
                reward = 0.3  # Some overlap
            else:
                reward = -0.5  # Wrong answer

        # Penalize empty responses
        if len(completion.strip()) < 1:
            reward = -1.0

        rewards.append(reward)

    return rewards


def main():
    parser = argparse.ArgumentParser(description="GRPO Training for Causal Reasoning")

    # Model arguments
    parser.add_argument("--model", type=str, required=True,
                        help="Path to SFT-trained model or base model")
    parser.add_argument("--tokenizer", type=str, default=None,
                        help="Tokenizer path (defaults to model path)")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to GRPO training data (JSONL)")
    parser.add_argument("--output", type=str, default="./output/causal_grpo",
                        help="Output directory")

    # Training arguments
    parser.add_argument("--epochs", type=int, default=2,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Per-device batch size")
    parser.add_argument("--num-generations", type=int, default=4,
                        help="Number of generations per prompt")
    parser.add_argument("--max-completion-length", type=int, default=256,
                        help="Maximum completion length")
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Learning rate")
    parser.add_argument("--beta", type=float, default=0.1,
                        help="KL penalty coefficient")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Maximum training samples")

    # GRPO specific
    parser.add_argument("--loss-type", type=str, default="grpo",
                        choices=["grpo", "dapo", "dr_grpo"],
                        help="GRPO loss variant")
    parser.add_argument("--scale-rewards", type=str, default="group",
                        choices=["group", "batch", "none"],
                        help="Reward scaling strategy")
    parser.add_argument("--reward-type", type=str, default="mcqa",
                        choices=["mcqa", "generation"],
                        help="Reward function type")

    args = parser.parse_args()

    # Setup output directory
    Path(args.output).mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("GRPO Training for Causal Reasoning")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Data: {args.data}")
    print(f"Output: {args.output}")
    print(f"Epochs: {args.epochs}")
    print(f"Num generations: {args.num_generations}")
    print(f"Beta (KL penalty): {args.beta}")
    print(f"Loss type: {args.loss_type}")

    # Load tokenizer
    tokenizer_path = args.tokenizer or args.model
    print(f"\nLoading tokenizer from {tokenizer_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    print(f"Loading model from {args.model}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # Load dataset
    print(f"\nLoading dataset from {args.data}...")
    dataset = load_grpo_dataset(args.data, args.max_samples)
    print(f"Dataset size: {len(dataset)}")

    # Select reward function
    if args.reward_type == "mcqa":
        reward_func = causal_mcqa_reward
        print("Using MCQA reward function")
    else:
        reward_func = causal_reasoning_reward
        print("Using generation reward function")

    # GRPO config
    grpo_config = GRPOConfig(
        output_dir=args.output,
        per_device_train_batch_size=args.batch_size,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        beta=args.beta,
        loss_type=args.loss_type,
        scale_rewards=args.scale_rewards if args.scale_rewards != "none" else False,
        logging_steps=10,
        save_steps=500,
        save_total_limit=3,
        bf16=True,
        report_to="none",
        remove_unused_columns=False,
    )

    # Create trainer
    print("\nInitializing GRPO trainer...")
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        reward_funcs=reward_func,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    # Train
    print("\nStarting GRPO training...")
    trainer.train()

    # Save
    print(f"\nSaving model to {args.output}...")
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)

    # Save training args
    with open(Path(args.output) / "training_args.json", 'w') as f:
        json.dump(vars(args), f, indent=2)

    print("\nGRPO training complete!")
    print(f"Model saved to: {args.output}")


if __name__ == "__main__":
    main()
