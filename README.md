
# QR2MSE_ICMR2025_llm4rul_phm08

## Project Overview

This project implements the paper: **"Pre-trained LLM-based Remaining Useful Life Prediction of Aircraft Engines"** by Tan Q, Yang L, Zhu F, et al., presented at the 15th International Conference on Quality, Reliability, Risk, Maintenance, and Safety Engineering (QR2MSE 2025), IET, 2025, pp. 1016-1024.

The project focuses on predicting the Remaining Useful Life (RUL) of aircraft engines using pre-trained Large Language Models (LLMs) and advanced deep learning techniques on the CMAPSS (Commercial Modular Aero-Propulsion System Simulation) dataset.

## Project Structure

```
QR2MSE_ICMR2025_llm4rul_phm08/
├── main.py                          # Main training and testing script
├── data_provider/                   # Data loading and processing
│   ├── __init__.py
│   ├── data_loader.py              # Data loading utilities for PHM08 dataset
│   ├── cluster_demo.py             # Clustering demonstrations
│   ├── plot_demo.py                # Plotting utilities
│   └── score_metric.py             # RUL evaluation metrics
├── models/                          # Deep learning model implementations
│   ├── __init__.py
│   ├── GPT4TS.py                   # GPT4TS model (LLM-based RUL prediction)
│   ├── embed.py                    # Embedding layers
│   └── traditional_models.py       # Traditional models (LSTM, CNN, Transformer, etc.)
├── utils/                           # Utility functions
│   ├── __init__.py
│   ├── tools.py                    # Statistical and plotting tools
│   └── train_and_test.py           # Training and testing functions
└── CMAPSSData/                      # CMAPSS dataset directory
```

## Key Features

- **LLM-based Approach**: Implements pre-trained language models (GPT4TS) for RUL prediction
- **CMAPSS Dataset**: Trained and evaluated on PHM08 (FD001-FD004) datasets
- **Comprehensive Evaluation**: RUL prediction with RMSE and scoring metrics
- **Visualization**: Single-unit and all-unit result plotting with prediction zones (Early/Timely/Late)

## Main Components

### Data Provider (`data_provider/`)
- Loads CMAPSS PHM08 datasets
- Processes time-series engine sensor data
- Generates train/validation/test splits
- Calculates RUL ground truth labels

### Models (`models/`)
- **GPT4TS**: Pre-trained LLM-based model for time series RUL prediction
- **Traditional Models**: Baseline deep learning architectures
- **Embedding**: Custom embedding layers for sensor data

### Utilities (`utils/`)
- Model training and evaluation pipelines
- Statistical analysis and result aggregation
- Visualization of predictions vs. actual RUL

## Usage

```bash
python main.py --model_name GPT4TS --dataset_uno FD004 --num_epochs 100 --batch_size 128
```

## Configuration Parameters

- `model_name`: Model architecture selection
- `seq_len`: Input sequence length (default: 64)
- `pre_len`: Prediction horizon (default: 1)
- `batch_size`: Training batch size (default: 128)
- `num_epochs`: Number of training epochs (default: 100)
- `learning_rate`: Learning rate (default: 0.005)
