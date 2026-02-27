"""
SemEval 2026 Task 12: Abductive Event Reasoning
Training: SFT (Supervised Fine-Tuning) using TRL

This script fine-tunes a small language model (Qwen3-0.5B) for causal reasoning
using the TRL SFTTrainer with LoRA.

Usage:
    # Single GPU
    python run_sft.py \
        --model Qwen/Qwen3-0.5B-Instruct \
        --data ./data/causal_sft_data.jsonl \
        --output ./output/causal_sft

    # Multi-GPU with accelerate
    accelerate launch run_sft.py \
        --model Qwen/Qwen3-0.5B-Instruct \
        --data ./data/causal_sft_data.jsonl \
        --output ./output/causal_sft

Requirements:
    pip install trl peft transformers accelerate bitsandbytes
"""

import os
import json
import argparse
import torch
from pathlib import Path
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer


def load_sft_dataset(data_path: str, max_samples: int = None) -> Dataset:
    """Load SFT dataset from JSONL file"""
    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    if max_samples:
        data = data[:max_samples]

    return Dataset.from_list(data)


def format_chat_template(example, tokenizer):
    """Format example using chat template"""
    messages = example.get("messages", [])
    if not messages:
        return {"text": ""}

    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        return {"text": text}
    except Exception as e:
        # Fallback: simple concatenation
        text = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            text += f"<|{role}|>\n{content}\n"
        return {"text": text}


def main():
    parser = argparse.ArgumentParser(description="SFT Training for Causal Reasoning")

    # Model arguments
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.5B-Instruct",
                        help="Base model name or path")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to SFT training data (JSONL)")
    parser.add_argument("--output", type=str, default="./output/causal_sft",
                        help="Output directory")

    # Training arguments
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Per-device batch size")
    parser.add_argument("--grad-accum", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Learning rate")
    parser.add_argument("--max-seq-length", type=int, default=512,
                        help="Maximum sequence length")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Maximum training samples")

    # LoRA arguments
    parser.add_argument("--lora-r", type=int, default=16,
                        help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32,
                        help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.05,
                        help="LoRA dropout")

    # Quantization
    parser.add_argument("--use-4bit", action="store_true",
                        help="Use 4-bit quantization")
    parser.add_argument("--use-8bit", action="store_true",
                        help="Use 8-bit quantization")

    args = parser.parse_args()

    # Setup output directory
    Path(args.output).mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("SFT Training for Causal Reasoning")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Data: {args.data}")
    print(f"Output: {args.output}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"LoRA rank: {args.lora_r}")

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Quantization config
    bnb_config = None
    if args.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    elif args.use_8bit:
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    # Load model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # LoRA config
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Load dataset
    print(f"\nLoading dataset from {args.data}...")
    dataset = load_sft_dataset(args.data, args.max_samples)
    print(f"Dataset size: {len(dataset)}")

    # Format dataset
    dataset = dataset.map(
        lambda x: format_chat_template(x, tokenizer),
        remove_columns=dataset.column_names
    )

    # SFT config
    sft_config = SFTConfig(
        output_dir=args.output,
        max_length=args.max_seq_length,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        logging_steps=10,
        save_steps=500,
        save_total_limit=3,
        bf16=True,
        optim="adamw_torch",
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        report_to="none",
        dataset_text_field="text",
    )

    # Create trainer
    print("\nInitializing trainer...")
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    # Train
    print("\nStarting training...")
    trainer.train()

    # Save
    print(f"\nSaving model to {args.output}...")
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)

    # Save training args
    with open(Path(args.output) / "training_args.json", 'w') as f:
        json.dump(vars(args), f, indent=2)

    print("\nTraining complete!")
    print(f"Model saved to: {args.output}")


if __name__ == "__main__":
    main()
