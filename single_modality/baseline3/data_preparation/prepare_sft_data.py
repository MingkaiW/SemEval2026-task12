"""
SemEval 2026 Task 12: Abductive Event Reasoning
Data Preparation: Prepare SFT Training Data

Converts causal triples to conversation format for SFT training with TRL.

Usage:
    python prepare_sft_data.py \
        --input ./data/causal_triples.jsonl \
        --output ./data/causal_sft_data.jsonl \
        --format chat
"""

import json
import argparse
import random
from pathlib import Path
from typing import List, Dict, Any


SYSTEM_PROMPT_CAUSAL_GEN = """You are an expert in causal reasoning and event analysis. Given a target event, identify and explain the most likely direct causes based on temporal order and causal relationships."""

SYSTEM_PROMPT_CAUSAL_JUDGE = """You are an expert in causal reasoning. Given a potential cause and an effect, determine whether the cause directly leads to the effect."""


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load JSONL file"""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def save_jsonl(data: List[Dict], path: str):
    """Save data to JSONL file"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"Saved {len(data)} items to {path}")


def create_causal_generation_sample(triple: Dict) -> Dict:
    """
    Create a causal generation sample (only for positive triples)

    Format: Given effect, generate cause
    """
    user_content = f"""Event: {triple['effect']}

What is the most likely direct cause of this event? Explain the causal relationship."""

    assistant_content = f"""The direct cause is: {triple['cause']}

This is a direct cause because the event "{triple['cause'][:100]}..." temporally precedes and directly leads to "{triple['effect'][:100]}..."."""

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_CAUSAL_GEN},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content}
        ]
    }


def create_causal_judgment_sample(triple: Dict) -> Dict:
    """
    Create a causal judgment sample (for both positive and negative)

    Format: Given cause and effect, judge if causal
    """
    user_content = f"""Potential Cause: {triple['cause']}

Effect: {triple['effect']}

Is the potential cause a direct cause of the effect? Answer with Yes or No and explain briefly."""

    if triple['label'] == 1:
        assistant_content = f"""Yes, this is a direct cause.

The event "{triple['cause'][:80]}..." directly leads to "{triple['effect'][:80]}..." because it creates the necessary conditions or triggers the effect through a clear causal chain."""
    else:
        assistant_content = f"""No, this is not a direct cause.

While "{triple['cause'][:80]}..." may be related to "{triple['effect'][:80]}...", it does not directly cause the effect. It may be a background factor, consequence, or unrelated event."""

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_CAUSAL_JUDGE},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content}
        ]
    }


def create_mcqa_sample(triples_by_question: Dict[str, List[Dict]]) -> List[Dict]:
    """
    Create multiple-choice QA samples grouped by question

    Format: Given effect and options, select the correct cause(s)
    """
    samples = []

    for question_id, triples in triples_by_question.items():
        if not triples:
            continue

        effect = triples[0]['effect']
        options = {}
        correct_answers = []

        for i, t in enumerate(triples[:4]):  # Max 4 options
            opt_label = chr(65 + i)  # A, B, C, D
            options[opt_label] = t['cause']
            if t['label'] == 1:
                correct_answers.append(opt_label)

        if not correct_answers:
            continue

        options_text = "\n".join([f"{k}. {v}" for k, v in options.items()])
        answer_text = ", ".join(sorted(correct_answers))

        user_content = f"""Target Event: {effect}

Which of the following is the most likely direct cause of this event?

{options_text}

Select all correct answers."""

        assistant_content = f"""{answer_text}

The correct answer(s) is/are {answer_text} because {"these events" if len(correct_answers) > 1 else "this event"} directly lead to the target event through clear causal relationships."""

        samples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT_CAUSAL_GEN},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content}
            ]
        })

    return samples


def prepare_sft_data(
    triples: List[Dict],
    format_type: str = "all",
    max_samples: int = None,
    shuffle: bool = True
) -> List[Dict]:
    """
    Prepare SFT training data

    Args:
        triples: List of causal triples
        format_type: "generation", "judgment", "mcqa", or "all"
        max_samples: Maximum number of samples to generate
        shuffle: Whether to shuffle the output

    Returns:
        List of conversation samples
    """
    samples = []

    if format_type in ["generation", "all"]:
        # Only positive triples for generation
        positive_triples = [t for t in triples if t['label'] == 1]
        for t in positive_triples:
            samples.append(create_causal_generation_sample(t))

    if format_type in ["judgment", "all"]:
        # All triples for judgment
        for t in triples:
            samples.append(create_causal_judgment_sample(t))

    if format_type in ["mcqa", "all"]:
        # Group by question for MCQA
        triples_by_question = {}
        for t in triples:
            qid = t.get('question_id', 'unknown')
            if qid not in triples_by_question:
                triples_by_question[qid] = []
            triples_by_question[qid].append(t)

        samples.extend(create_mcqa_sample(triples_by_question))

    if shuffle:
        random.shuffle(samples)

    if max_samples:
        samples = samples[:max_samples]

    return samples


def main():
    parser = argparse.ArgumentParser(
        description="Prepare SFT training data from causal triples"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to causal_triples.jsonl"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./data/causal_sft_data.jsonl",
        help="Output path (default: ./data/causal_sft_data.jsonl)"
    )
    parser.add_argument(
        "--format", "-f",
        type=str,
        choices=["generation", "judgment", "mcqa", "all"],
        default="all",
        help="Data format type (default: all)"
    )
    parser.add_argument(
        "--max-samples", "-m",
        type=int,
        default=None,
        help="Maximum number of samples"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling"
    )

    args = parser.parse_args()

    random.seed(args.seed)

    print(f"Loading triples from: {args.input}")
    triples = load_jsonl(args.input)
    print(f"Loaded {len(triples)} triples")

    print(f"\nPreparing SFT data with format: {args.format}")
    samples = prepare_sft_data(
        triples,
        format_type=args.format,
        max_samples=args.max_samples
    )

    print(f"Generated {len(samples)} SFT samples")
    save_jsonl(samples, args.output)

    # Show sample
    if samples:
        print("\n=== Sample Output ===")
        sample = samples[0]
        for msg in sample["messages"]:
            role = msg["role"].upper()
            content = msg["content"][:200] + "..." if len(msg["content"]) > 200 else msg["content"]
            print(f"\n[{role}]")
            print(content)

    print("\nDone!")


if __name__ == "__main__":
    main()
