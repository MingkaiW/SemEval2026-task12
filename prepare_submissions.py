#!/usr/bin/env python3
"""
Prepare submission packages for all test results.

Creates submission.zip files for Codabench upload from test result JSONs.
"""

import json
import os
import shutil
import zipfile
from datetime import datetime
from pathlib import Path

# Add utils to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from utils.submission_utils import format_answer, validate_submission


# Configuration: test result files to process (baseline_num, model_name, result_path, approach_description)
BASELINES = [
    {
        "baseline_num": 1,
        "model_name": "qwen3",
        "result_path": "single_modality/baseline/results_qwen3_test.json",
        "approach": """## Approach: Qwen3 LLM Baseline

**Model**: Qwen3 1.7B (via Ollama)
**Method**: Direct prompting with context
**Configuration**:
- Model type: Ollama local inference
- Model name: qwen3:1.7b
- Use context: Yes
- Task: Multi-label causal relation identification

**Description**:
This baseline uses the Qwen3 1.7B language model to directly predict causal relations
between events. The model receives the target event, candidate options (A-D), and
relevant context documents, then outputs which options are valid causes of the target event.
"""
    },
    {
        "baseline_num": 2,
        "model_name": "unifiedqa",
        "result_path": "single_modality/baseline2/results_unifiedqa_test.json",
        "approach": """## Approach: UnifiedQA Baseline

**Model**: allenai/unifiedqa-t5-small
**Method**: Question-answering format
**Configuration**:
- Model name: allenai/unifiedqa-t5-small
- Input format: QA-style with context

**Description**:
This baseline uses the UnifiedQA model (T5-small variant) which is trained on multiple
QA datasets. The causal identification task is reformatted as a question-answering problem
where the model must identify which events caused the target event.
"""
    },
    {
        "baseline_num": 3,
        "model_name": "deepseek",
        "result_path": "single_modality/baseline3/results_deepseek_test.json",
        "approach": """## Approach: DeepSeek LLM Baseline

**Model**: DeepSeek Chat
**Method**: Prompt-based fusion
**Configuration**:
- LLM model: deepseek-chat
- Fusion method: prompt
- Use COMET: No

**Description**:
This baseline uses the DeepSeek Chat model with prompt-based fusion. The model receives
the target event and candidate options, then predicts causal relations through
natural language reasoning without external commonsense knowledge (COMET disabled).
"""
    },
    {
        "baseline_num": 4,
        "model_name": "kg_rotateE",
        "result_path": "single_modality/baseline3/results_kg_rotateE_test.json",
        "approach": """## Approach: Knowledge Graph + RotatE Baseline

**Model**: DeepSeek Chat + Knowledge Graph Embeddings
**Method**: Prompt-based fusion with KG enhancement
**Configuration**:
- LLM model: deepseek-chat
- Fusion method: prompt
- KG embedding: RotatE

**Description**:
This baseline combines DeepSeek Chat with knowledge graph embeddings using the RotatE
model. The approach leverages structured knowledge from knowledge graphs to enhance
causal relation identification. RotatE embeddings provide relational reasoning
capabilities that complement the LLM's language understanding.
"""
    },
]

TEST_QUESTIONS_PATH = "test_data/questions.jsonl"
OUTPUT_DIR = "submissions"


def load_question_ids(questions_path: str) -> list:
    """Load question IDs from test questions file in order."""
    ids = []
    with open(questions_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                ids.append(data["id"])
    return ids


def load_predictions(result_path: str) -> list:
    """Load predictions from result JSON file."""
    with open(result_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("predictions", [])


def convert_to_submission(predictions: list, id_mapping: dict = None) -> list:
    """
    Convert predictions to submission format.

    Args:
        predictions: List of prediction dicts with 'id' and 'prediction' keys
        id_mapping: Optional dict to map original IDs to submission IDs

    Returns:
        List of dicts with 'id' and 'answer' keys
    """
    submission = []
    for pred in predictions:
        original_id = pred.get("id", "")

        # Apply ID mapping if provided
        if id_mapping is not None:
            submission_id = id_mapping.get(str(original_id), original_id)
        else:
            submission_id = original_id

        # Get prediction and format as answer string
        prediction = pred.get("prediction", [])
        answer = format_answer(prediction)

        submission.append({
            "id": submission_id,
            "answer": answer
        })

    return submission


def save_submission_jsonl(submission: list, output_path: str):
    """Save submission as JSONL file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in submission:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def create_submission_zip(submission_dir: str, zip_path: str):
    """Create submission.zip from submission directory."""
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(submission_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, os.path.dirname(submission_dir))
                zf.write(file_path, arcname)


def create_readme(baseline_info: dict, output_path: str, num_entries: int):
    """Create README.md for the submission."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    readme_content = f"""# Baseline {baseline_info['baseline_num']}: {baseline_info['model_name']}

## Submission Info

- **Submission Time**: {timestamp}
- **Number of Entries**: {num_entries}
- **Source File**: `{baseline_info['result_path']}`

{baseline_info['approach']}

## Files

- `submission.zip` - Submission package for Codabench
  - Contains `submission/submission.jsonl`

## How to Submit

1. Upload `submission.zip` to Codabench: https://www.codabench.org/competitions/12440/
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(readme_content)


def process_baseline(baseline_info: dict, question_ids: list, output_base: str):
    """Process a single baseline's results into a submission package."""
    baseline_num = baseline_info["baseline_num"]
    model_name = baseline_info["model_name"]
    result_path = baseline_info["result_path"]

    folder_name = f"baseline{baseline_num}"
    print(f"\nProcessing {folder_name} ({model_name})...")

    # Load predictions
    predictions = load_predictions(result_path)
    print(f"  Loaded {len(predictions)} predictions")

    # Create ID mapping for UnifiedQA (uses numeric IDs)
    id_mapping = None
    if model_name == "unifiedqa":
        id_mapping = {str(i): qid for i, qid in enumerate(question_ids)}
        print(f"  Created ID mapping for {len(id_mapping)} questions")

    # Convert to submission format
    submission = convert_to_submission(predictions, id_mapping)
    print(f"  Converted to {len(submission)} submission entries")

    # Create output directory structure
    baseline_dir = os.path.join(output_base, folder_name)
    submission_folder = os.path.join(baseline_dir, "submission")
    os.makedirs(submission_folder, exist_ok=True)

    # Save submission.jsonl
    jsonl_path = os.path.join(submission_folder, "submission.jsonl")
    save_submission_jsonl(submission, jsonl_path)
    print(f"  Saved: {jsonl_path}")

    # Validate submission
    is_valid, errors = validate_submission(jsonl_path)
    if is_valid:
        print(f"  Validation: PASSED ({len(submission)} entries)")
    else:
        print(f"  Validation: FAILED")
        for error in errors[:5]:  # Show first 5 errors
            print(f"    - {error}")
        if len(errors) > 5:
            print(f"    ... and {len(errors) - 5} more errors")

    # Create README.md
    readme_path = os.path.join(baseline_dir, "README.md")
    create_readme(baseline_info, readme_path, len(submission))
    print(f"  Created: {readme_path}")

    # Create submission.zip (zip the submission folder)
    zip_path = os.path.join(baseline_dir, "submission.zip")
    create_submission_zip(submission_folder, zip_path)
    print(f"  Created: {zip_path}")

    # Keep submission.jsonl file visible (copy to baseline dir for easy access)
    visible_jsonl = os.path.join(baseline_dir, "submission.jsonl")
    shutil.copy(jsonl_path, visible_jsonl)
    print(f"  Copied: {visible_jsonl}")

    # Clean up submission folder (keep the visible jsonl, README, and zip)
    shutil.rmtree(submission_folder)

    return is_valid, folder_name, model_name


def main():
    base_dir = Path(__file__).parent
    os.chdir(base_dir)

    print("=" * 60)
    print("Preparing Submission Packages for Codabench")
    print("=" * 60)

    # Load question IDs for ID mapping
    print(f"\nLoading question IDs from {TEST_QUESTIONS_PATH}...")
    question_ids = load_question_ids(TEST_QUESTIONS_PATH)
    print(f"  Loaded {len(question_ids)} question IDs")

    # Clean and create output directory
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Process each baseline
    results = []
    for baseline_info in BASELINES:
        result_path = baseline_info["result_path"]
        if not os.path.exists(result_path):
            print(f"\nSkipping baseline{baseline_info['baseline_num']}: {result_path} not found")
            results.append((None, f"baseline{baseline_info['baseline_num']}", baseline_info["model_name"]))
            continue

        result = process_baseline(baseline_info, question_ids, OUTPUT_DIR)
        results.append(result)

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    for is_valid, folder_name, model_name in results:
        if is_valid is None:
            status = "SKIPPED (file not found)"
        elif is_valid:
            status = "SUCCESS"
        else:
            status = "FAILED (validation errors)"

        print(f"  {folder_name} ({model_name}): {status}")
        if is_valid:
            zip_path = os.path.join(OUTPUT_DIR, folder_name, "submission.zip")
            readme_path = os.path.join(OUTPUT_DIR, folder_name, "README.md")
            print(f"    -> {zip_path}")
            print(f"    -> {readme_path}")

    print(f"\nOutput directory: {OUTPUT_DIR}/")
    print("Upload submission.zip files to Codabench.")


if __name__ == "__main__":
    main()
