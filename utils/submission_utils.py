"""
Submission utilities for SemEval 2026 Task 12.

This module provides functions for saving predictions in the required submission format:
- JSONL format with one JSON object per line
- Each object has "id" and "answer" fields
- Answer format: comma-separated uppercase letters (e.g., "A" or "A,B")

Example submission.jsonl:
    {"id": "q-2020", "answer": "A"}
    {"id": "q-2021", "answer": "B,D"}
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


VALID_OPTIONS = {"A", "B", "C", "D"}


def format_answer(answer: Union[str, List[str], Set[str], None]) -> str:
    """
    Convert answer to comma-separated uppercase string.

    Args:
        answer: Answer in various formats:
            - str: "A", "a", "A,B", "a, b"
            - List[str]: ["A", "B"], ["a", "b"]
            - Set[str]: {"A", "B"}
            - None: returns empty string

    Returns:
        Comma-separated uppercase string, e.g., "A,B"
        Options are sorted alphabetically.

    Examples:
        >>> format_answer(["a", "B"])
        "A,B"
        >>> format_answer({"C"})
        "C"
        >>> format_answer("a,b")
        "A,B"
        >>> format_answer(None)
        ""
    """
    if answer is None:
        return ""

    # Convert to set of uppercase letters
    if isinstance(answer, str):
        # Split by comma and clean up
        options = {opt.strip().upper() for opt in answer.split(",") if opt.strip()}
    elif isinstance(answer, (list, set, frozenset)):
        options = {str(opt).strip().upper() for opt in answer if opt}
    else:
        options = {str(answer).strip().upper()}

    # Filter to valid options only
    valid = options & VALID_OPTIONS

    # Sort and join
    return ",".join(sorted(valid))


def save_submission(
    predictions: List[Dict[str, Any]],
    output_path: Union[str, Path],
    validate: bool = True
) -> None:
    """
    Save predictions as JSONL submission file.

    Args:
        predictions: List of dicts with "id" and "answer" keys.
            The "answer" can be in any format accepted by format_answer().
        output_path: Path to output JSONL file.
        validate: If True, validate after saving.

    Raises:
        ValueError: If predictions is empty or missing required keys.

    Example:
        >>> predictions = [
        ...     {"id": "q-001", "answer": ["A"]},
        ...     {"id": "q-002", "answer": {"B", "C"}},
        ... ]
        >>> save_submission(predictions, "submission.jsonl")
    """
    if not predictions:
        raise ValueError("Predictions list cannot be empty")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for pred in predictions:
            if "id" not in pred:
                raise ValueError(f"Prediction missing 'id' key: {pred}")
            if "answer" not in pred:
                raise ValueError(f"Prediction missing 'answer' key: {pred}")

            entry = {
                "id": pred["id"],
                "answer": format_answer(pred["answer"])
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    if validate:
        is_valid, errors = validate_submission(output_path)
        if not is_valid:
            print(f"Warning: Submission has validation errors: {errors}")


def load_submission(file_path: Union[str, Path]) -> List[Dict[str, str]]:
    """
    Load a submission JSONL file.

    Args:
        file_path: Path to JSONL file.

    Returns:
        List of dicts with "id" and "answer" keys.

    Raises:
        FileNotFoundError: If file doesn't exist.
        json.JSONDecodeError: If file contains invalid JSON.
    """
    file_path = Path(file_path)
    predictions = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                predictions.append(entry)
            except json.JSONDecodeError as e:
                raise json.JSONDecodeError(
                    f"Invalid JSON at line {line_num}: {e.msg}",
                    e.doc, e.pos
                )

    return predictions


def validate_submission(file_path: Union[str, Path]) -> Tuple[bool, List[str]]:
    """
    Validate a submission file.

    Checks:
        - File exists and is readable
        - Each line is valid JSON
        - Each entry has "id" and "answer" fields
        - Answer contains only valid options (A, B, C, D)
        - No duplicate IDs

    Args:
        file_path: Path to submission JSONL file.

    Returns:
        Tuple of (is_valid, list_of_error_messages)

    Example:
        >>> is_valid, errors = validate_submission("submission.jsonl")
        >>> if not is_valid:
        ...     for error in errors:
        ...         print(error)
    """
    errors = []
    file_path = Path(file_path)

    if not file_path.exists():
        return False, [f"File not found: {file_path}"]

    seen_ids = set()

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                # Check JSON validity
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError as e:
                    errors.append(f"Line {line_num}: Invalid JSON - {e.msg}")
                    continue

                # Check required fields
                if "id" not in entry:
                    errors.append(f"Line {line_num}: Missing 'id' field")
                else:
                    # Check for duplicate IDs
                    if entry["id"] in seen_ids:
                        errors.append(f"Line {line_num}: Duplicate ID '{entry['id']}'")
                    seen_ids.add(entry["id"])

                if "answer" not in entry:
                    errors.append(f"Line {line_num}: Missing 'answer' field")
                else:
                    # Validate answer format
                    answer = entry["answer"]
                    if answer:  # Empty answer is allowed (scored as 0)
                        options = {opt.strip() for opt in answer.split(",")}
                        invalid = options - VALID_OPTIONS
                        if invalid:
                            errors.append(
                                f"Line {line_num}: Invalid options {invalid} in answer '{answer}'"
                            )
                        # Check for lowercase
                        if any(opt != opt.upper() for opt in answer.replace(",", "")):
                            errors.append(
                                f"Line {line_num}: Answer must be uppercase, got '{answer}'"
                            )

    except Exception as e:
        errors.append(f"Error reading file: {e}")

    return len(errors) == 0, errors


def convert_numpy_predictions(
    predictions: "np.ndarray",
    ids: List[str],
    threshold: float = 0.5,
    labels: Optional[List[str]] = None
) -> List[Dict[str, str]]:
    """
    Convert numpy array predictions to submission format.

    Args:
        predictions: Array of shape (n_samples, n_labels) with probabilities or binary values.
            Each row represents predictions for one sample.
            Each column represents one label (A, B, C, D by default).
        ids: List of sample IDs, must match length of predictions.
        threshold: Threshold for converting probabilities to binary (default 0.5).
        labels: Label names, defaults to ["A", "B", "C", "D"].

    Returns:
        List of dicts with "id" and "answer" keys.

    Raises:
        ImportError: If numpy is not installed.
        ValueError: If predictions and ids have mismatched lengths.

    Example:
        >>> import numpy as np
        >>> preds = np.array([[0.9, 0.1, 0.2, 0.3], [0.1, 0.8, 0.7, 0.2]])
        >>> ids = ["q-001", "q-002"]
        >>> result = convert_numpy_predictions(preds, ids, threshold=0.5)
        >>> # result: [{"id": "q-001", "answer": "A"}, {"id": "q-002", "answer": "B,C"}]
    """
    if not HAS_NUMPY:
        raise ImportError("numpy is required for convert_numpy_predictions")

    if len(predictions) != len(ids):
        raise ValueError(
            f"Predictions length ({len(predictions)}) doesn't match IDs length ({len(ids)})"
        )

    if labels is None:
        labels = ["A", "B", "C", "D"]

    # Ensure predictions is a numpy array
    predictions = np.asarray(predictions)

    # Convert to binary if not already
    binary_preds = (predictions >= threshold).astype(int)

    result = []
    for i, sample_id in enumerate(ids):
        # Get indices where prediction is 1
        selected_indices = np.where(binary_preds[i] == 1)[0]

        # Convert indices to labels
        selected_labels = [labels[idx] for idx in selected_indices if idx < len(labels)]

        result.append({
            "id": sample_id,
            "answer": format_answer(selected_labels)
        })

    return result


if __name__ == "__main__":
    # Basic tests
    print("Testing format_answer...")
    assert format_answer(["a", "B"]) == "A,B"
    assert format_answer({"C"}) == "C"
    assert format_answer("a,b") == "A,B"
    assert format_answer("A") == "A"
    assert format_answer(None) == ""
    assert format_answer(["D", "A", "B"]) == "A,B,D"  # Sorted
    assert format_answer(["a", "X", "b"]) == "A,B"  # Invalid option filtered
    print("  All format_answer tests passed!")

    # Test save and load
    print("Testing save_submission and load_submission...")
    import tempfile
    import os

    test_predictions = [
        {"id": "q-001", "answer": ["A"]},
        {"id": "q-002", "answer": {"B", "D"}},
        {"id": "q-003", "answer": "C"},
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        temp_path = f.name

    try:
        save_submission(test_predictions, temp_path)
        loaded = load_submission(temp_path)

        assert len(loaded) == 3
        assert loaded[0] == {"id": "q-001", "answer": "A"}
        assert loaded[1] == {"id": "q-002", "answer": "B,D"}
        assert loaded[2] == {"id": "q-003", "answer": "C"}
        print("  All save/load tests passed!")

        # Test validation
        print("Testing validate_submission...")
        is_valid, errors = validate_submission(temp_path)
        assert is_valid, f"Validation failed: {errors}"
        print("  Validation test passed!")

    finally:
        os.unlink(temp_path)

    # Test numpy conversion if available
    if HAS_NUMPY:
        print("Testing convert_numpy_predictions...")
        preds = np.array([
            [0.9, 0.1, 0.2, 0.3],
            [0.1, 0.8, 0.7, 0.2],
            [0.1, 0.1, 0.1, 0.1],
        ])
        ids = ["q-001", "q-002", "q-003"]
        result = convert_numpy_predictions(preds, ids, threshold=0.5)

        assert result[0] == {"id": "q-001", "answer": "A"}
        assert result[1] == {"id": "q-002", "answer": "B,C"}
        assert result[2] == {"id": "q-003", "answer": ""}  # No prediction above threshold
        print("  All numpy conversion tests passed!")

    print("\nAll tests passed!")
