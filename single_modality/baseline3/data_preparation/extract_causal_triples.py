"""
SemEval 2026 Task 12: Abductive Event Reasoning
Data Preparation: Extract Causal Triples from Training Data

This script extracts causal relation triples from the SemEval training data
for use in SFT and GRPO training.

Usage:
    python extract_causal_triples.py \
        --input /path/to/train_data/questions.jsonl \
        --output ./data/causal_triples.jsonl
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass, asdict


@dataclass
class CausalTriple:
    """Represents a causal relationship triple"""
    cause: str           # The cause event
    effect: str          # The effect event (target_event)
    relation: str        # Relation type (always "causes" for now)
    label: int           # 1 for positive (correct cause), 0 for negative
    question_id: str     # Original question ID
    topic_id: int        # Topic ID for grouping


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


def extract_triples_from_questions(questions_path: str) -> List[CausalTriple]:
    """
    Extract causal triples from questions.jsonl

    For each question:
    - Correct answers (golden_answer) are positive samples (label=1)
    - Incorrect answers are negative samples (label=0)
    - "None of the others are correct causes." is excluded

    Args:
        questions_path: Path to questions.jsonl

    Returns:
        List of CausalTriple objects
    """
    questions = load_jsonl(questions_path)
    triples = []

    for q in questions:
        target = q["target_event"]
        golden = q["golden_answer"]  # "A" or "B,D"
        question_id = q["id"]
        topic_id = q["topic_id"]

        # Parse golden answers
        golden_options = set(ans.strip() for ans in golden.split(","))

        # Process all options
        for opt in ["A", "B", "C", "D"]:
            option_key = f"option_{opt}"
            if option_key not in q:
                continue

            option_text = q[option_key]

            # Skip "None of the others are correct causes."
            if "None of the others" in option_text:
                continue

            # Determine label
            is_correct = opt in golden_options
            label = 1 if is_correct else 0

            triple = CausalTriple(
                cause=option_text,
                effect=target,
                relation="causes",
                label=label,
                question_id=question_id,
                topic_id=topic_id
            )
            triples.append(triple)

    return triples


def analyze_triples(triples: List[CausalTriple]) -> Dict[str, Any]:
    """Analyze the extracted triples"""
    positive = sum(1 for t in triples if t.label == 1)
    negative = sum(1 for t in triples if t.label == 0)
    topics = len(set(t.topic_id for t in triples))
    questions = len(set(t.question_id for t in triples))

    return {
        "total": len(triples),
        "positive": positive,
        "negative": negative,
        "positive_ratio": positive / len(triples) if triples else 0,
        "topics": topics,
        "questions": questions
    }


def main():
    parser = argparse.ArgumentParser(
        description="Extract causal triples from SemEval training data"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to questions.jsonl"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./data/causal_triples.jsonl",
        help="Output path for triples (default: ./data/causal_triples.jsonl)"
    )
    parser.add_argument(
        "--split-output", "-s",
        action="store_true",
        help="Also output separate positive/negative files"
    )

    args = parser.parse_args()

    print(f"Loading questions from: {args.input}")
    triples = extract_triples_from_questions(args.input)

    # Analyze
    stats = analyze_triples(triples)
    print("\n=== Extraction Statistics ===")
    print(f"Total triples: {stats['total']}")
    print(f"Positive (correct causes): {stats['positive']}")
    print(f"Negative (incorrect causes): {stats['negative']}")
    print(f"Positive ratio: {stats['positive_ratio']:.2%}")
    print(f"Unique topics: {stats['topics']}")
    print(f"Unique questions: {stats['questions']}")

    # Save all triples
    save_jsonl([asdict(t) for t in triples], args.output)

    # Optionally save split files
    if args.split_output:
        output_dir = Path(args.output).parent

        positive_triples = [asdict(t) for t in triples if t.label == 1]
        negative_triples = [asdict(t) for t in triples if t.label == 0]

        save_jsonl(positive_triples, output_dir / "positive_triples.jsonl")
        save_jsonl(negative_triples, output_dir / "negative_triples.jsonl")

    print("\nDone!")


if __name__ == "__main__":
    main()
