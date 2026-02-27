# SemEval 2026 Task 12

This repository contains the code, data processing scripts, and results for our participation in **SemEval 2026 Task 12**.

## Project Structure

```
SemEval2026-task12/
├── src/                  # Source code
│   ├── data_processing/  # Scripts for data loading and preprocessing
│   ├── models/           # Model definitions
│   └── evaluation/       # Evaluation scripts
├── data/
│   ├── train/            # Training data
│   ├── dev/              # Development/validation data
│   └── test/             # Test data
├── results/              # Experiment results and predictions
├── models/               # Saved model checkpoints (not tracked by git)
├── requirements.txt      # Python dependencies
└── README.md
```

## Requirements

Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Usage

### Data Preprocessing

```bash
python src/data_processing/preprocess.py --input data/train/ --output data/train/processed/
```

### Training

```bash
python src/train.py --config configs/config.yaml
```

### Evaluation

```bash
python src/evaluation/evaluate.py --predictions results/predictions.json --gold data/test/gold.json
```

## Results

| Model | Metric | Score |
|-------|--------|-------|
| Baseline | - | - |

## Citation

If you use this code, please cite our paper:

```bibtex
@inproceedings{semeval2026task12,
  title={},
  author={},
  booktitle={Proceedings of SemEval 2026},
  year={2026}
}
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
