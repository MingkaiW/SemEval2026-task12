"""
SemEval 2026 Task 12: Abductive Event Reasoning
Data Preparation: Prepare GRPO Training Data

Converts causal triples to prompt format for GRPO training with TRL.

Usage:
    python prepare_grpo_data.py \
        --input ./data/causal_triples.jsonl \
        --output ./data/causal_grpo_data.jsonl
"""

import json
import argparse
import random
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict


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


def group_triples_by_question(triples: List[Dict]) -> Dict[str, List[Dict]]:
    """Group triples by question_id"""
    grouped = defaultdict(list)
    for t in triples:
        qid = t.get('question_id', 'unknown')
        grouped[qid].append(t)
    return dict(grouped)


def create_grpo_sample(question_id: str, triples: List[Dict]) -> Dict:
    """
    Create a GRPO training sample from grouped triples

    GRPO format requires:
    - prompt: The question/context
    - golden_causes: List of correct causes for reward computation
    - target_event: The effect event
    - metadata: Additional info for reward function

    Args:
        question_id: Question ID
        triples: List of triples for this question

    Returns:
        GRPO format sample
    """
    if not triples:
        return None

    # Get target event (effect) - same for all triples in a question
    target_event = triples[0]['effect']

    # Separate positive and negative causes
    golden_causes = [t['cause'] for t in triples if t['label'] == 1]
    negative_causes = [t['cause'] for t in triples if t['label'] == 0]

    if not golden_causes:
        return None

    # Create prompt
    prompt = f"""You are an expert in causal reasoning. Given the following event, identify the most likely direct cause.

Target Event: {target_event}

What is the direct cause of this event? Provide a clear and specific answer."""

    return {
        "prompt": prompt,
        "golden_causes": golden_causes,
        "negative_causes": negative_causes,
        "target_event": target_event,
        "question_id": question_id,
        "topic_id": triples[0].get('topic_id', 0),
        "num_correct": len(golden_causes),
        "num_options": len(triples)
    }


def create_mcqa_grpo_sample(question_id: str, triples: List[Dict]) -> Dict:
    """
    Create a GRPO sample in MCQA format

    This format is closer to the original task format
    """
    if not triples:
        return None

    target_event = triples[0]['effect']
    options = {}
    golden_answers = []

    for i, t in enumerate(triples[:4]):
        opt_label = chr(65 + i)  # A, B, C, D
        options[opt_label] = t['cause']
        if t['label'] == 1:
            golden_answers.append(opt_label)

    if not golden_answers:
        return None

    options_text = "\n".join([f"{k}. {v}" for k, v in options.items()])

    prompt = f"""You are an expert in causal reasoning. Given the target event and options, select the direct cause(s).

Target Event: {target_event}

Options:
{options_text}

Which option(s) represent the direct cause? Answer with only the letter(s)."""

    return {
        "prompt": prompt,
        "golden_answers": golden_answers,
        "golden_causes": [options[a] for a in golden_answers],
        "options": options,
        "target_event": target_event,
        "question_id": question_id,
        "topic_id": triples[0].get('topic_id', 0)
    }


def prepare_grpo_data(
    triples: List[Dict],
    format_type: str = "generation",
    shuffle: bool = True
) -> List[Dict]:
    """
    Prepare GRPO training data

    Args:
        triples: List of causal triples
        format_type: "generation" or "mcqa"
        shuffle: Whether to shuffle output

    Returns:
        List of GRPO samples
    """
    # Group by question
    grouped = group_triples_by_question(triples)

    samples = []
    for qid, q_triples in grouped.items():
        if format_type == "mcqa":
            sample = create_mcqa_grpo_sample(qid, q_triples)
        else:
            sample = create_grpo_sample(qid, q_triples)

        if sample:
            samples.append(sample)

    if shuffle:
        random.shuffle(samples)

    return samples


def main():
    parser = argparse.ArgumentParser(
        description="Prepare GRPO training data from causal triples"
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
        default="./data/causal_grpo_data.jsonl",
        help="Output path (default: ./data/causal_grpo_data.jsonl)"
    )
    parser.add_argument(
        "--format", "-f",
        type=str,
        choices=["generation", "mcqa"],
        default="mcqa",
        help="Output format (default: mcqa)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    args = parser.parse_args()
    random.seed(args.seed)

    print(f"Loading triples from: {args.input}")
    triples = load_jsonl(args.input)
    print(f"Loaded {len(triples)} triples")

    print(f"\nPreparing GRPO data with format: {args.format}")
    samples = prepare_grpo_data(triples, format_type=args.format)

    print(f"Generated {len(samples)} GRPO samples")

    # Statistics
    if samples:
        avg_correct = sum(
            s.get('num_correct', len(s.get('golden_answers', [])))
            for s in samples
        ) / len(samples)
        print(f"Average correct answers per sample: {avg_correct:.2f}")

    save_jsonl(samples, args.output)

    # Show sample
    if samples:
        print("\n=== Sample Output ===")
        sample = samples[0]
        print(f"\n[PROMPT]\n{sample['prompt'][:500]}...")
        if 'golden_answers' in sample:
            print(f"\n[GOLDEN ANSWERS]: {sample['golden_answers']}")
        if 'golden_causes' in sample:
            for i, cause in enumerate(sample['golden_causes'][:2]):
                print(f"\n[GOLDEN CAUSE {i+1}]: {cause[:200]}...")

    print("\nDone!")


if __name__ == "__main__":
    main()
