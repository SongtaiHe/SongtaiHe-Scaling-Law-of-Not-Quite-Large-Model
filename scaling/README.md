# Scaling Package Overview

This package groups submodules for analysis, configuration management, data preparation, tokenization, and utility helpers supporting scaling-law research.

## Module layout

- `analysis/`: Experiment tracking utilities, plotting helpers, and data exploration notebooks/scripts.
- `configs/`: Default configuration files for experiments, including model definitions and training hyperparameters.
- `data_scripts/`: Data preprocessing scripts for dataset curation and preparation pipelines.
- `models/`: Model definitions and training loops for scaling-law experiments.
- `sweeps/`: Experiment sweep specifications and automation helpers.
- `tokenization/`: Tokenizer training and loading utilities shared across experiments.
- `utils/`: Shared helper functions and common infrastructure used across modules.

## Environment

The package is designed for Python 3.11+. Install dependencies with `pip install -r requirements.txt` or manage development dependencies via Poetry using `poetry install`.
