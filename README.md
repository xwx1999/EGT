# EGT: Enhanced Genomic Transformer for Animal Quantitative Trait Prediction

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

EGT (Enhanced Genomic Transformer) is a deep learning model designed for predicting animal quantitative traits from Single Nucleotide Polymorphism (SNP) data. The model leverages an autoencoder for dimensionality reduction of the genomic data, combined with a Transformer-based architecture to capture complex patterns and improve prediction accuracy.

This repository contains the full implementation of the EGT model, data preprocessing scripts, and an experiment runner to reproduce results on the provided datasets.

## Features

- **End-to-End Workflow**: A single script to automate preprocessing, training, and evaluation for different datasets.
- **Enhanced Transformer Model**: Utilizes an autoencoder for feature extraction followed by a multi-head self-attention Transformer network.
- **Experiment Tracking**: Integrated with [Weights & Biases](https://wandb.ai/) for seamless logging of metrics, configurations, and results.
- **Flexible Configuration**: Model architecture, training parameters, and data paths are managed via YAML configuration files.

## Project Structure

The repository is organized as follows:

```
EGT/
├── configs/              # Configuration files
│   └── default_config.yaml
├── data/                 # Data directory
│   ├── raw/              # Raw datasets (AQT_I, AQT_II, AQT_III)
|   └── processed/        # Processed datasets (AQT_I, AQT_II, AQT_III)
├── models/               # Model implementations
│   ├── egt.py            # EGT model
│   ├── attention.py      # Self-attention module
│   └── autoencoder.py    # Autoencoder module
├── scripts/              # Standalone scripts
│   └── preprocess_data.py
├── utils/                # Utility functions
│   ├── evaluation.py     # Evaluation metrics
│   └── preprocessing.py  # Data loading and splitting
├── evaluate.py           # Evaluation script
├── run_experiment.py     # Main experiment runner
├── setup.py              # Installation configuration
├── train.py              # Training script
├── requirements.txt      # Project dependencies
└── README.md
```

## Getting Started

Follow these instructions to set up the environment and run an experiment.

### 1. Installation

First, clone the repository to your local machine:

```bash
git clone https://github.com/xwx1999/EGT.git
cd EGT
```

Next, it is highly recommended to create a virtual environment to manage dependencies:

```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Linux/macOS
source venv/bin/activate
```

Install the required Python packages from `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 2. Weights & Biases Setup

This project uses Weights & Biases (W&B) to log experiment results. You will need a free account.

Log in to W&B from your terminal. You will be prompted to enter your API key.

```bash
wandb login
```

### 3. Running an Experiment

The entire workflow (data preprocessing, model training, and evaluation) is handled by the `run_experiment.py` script.

To run an experiment for a specific dataset, use the following command:

```bash
python run_experiment.py --dataset <DATASET_NAME>
```

Replace `<DATASET_NAME>` with one of the available datasets:
*   `AQT_I`
*   `AQT_II`
*   `AQT_III`

For example, to run the complete pipeline for the `AQT_I` dataset:

```bash
python run_experiment.py --dataset AQT_I
```

The script will first preprocess the raw data, then train the model, and finally evaluate it. All results, including the final performance metrics, will be logged to your W&B dashboard.

## Citation

If you use this code in your research, please consider citing:

```
@article{xiang2025egt,
  title={Enhanced Genomic Transformer for Animal Quantitative Trait Prediction},
  author={Weixi Xiang},
  journal={GitHub repository},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

- Author: Weixi Xiang
- Email: xiangweixi@hotmail.com
- Project Link: [https://github.com/xwx1999/EGT](https://github.com/xwx1999/EGT)

## Running an Experiment

To run a full experiment (preprocessing, training, and evaluation) for a specific dataset, use the `run_experiment.py` script. This script will automate the entire workflow and log results to Weights & Biases.

### Prerequisites

1.  Make sure you have installed all the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

2.  Log in to Weights & Biases. You can do this by running `wandb login` in your terminal and pasting your API key, or by providing the key as an argument to the script.

### Usage

Run the following command from the root of the project:

```bash
python run_experiment.py --dataset <DATASET_NAME>
```

Replace `<DATASET_NAME>` with one of the following:
*   `AQT_I`
*   `AQT_II`
*   `AQT_III`

For example, to run the experiment on the `AQT_I` dataset:

```bash
python run_experiment.py --dataset AQT_I
```

If you need to provide your `wandb` API key directly, you can use the `--wandb_login_key` argument:

```bash
python run_experiment.py --dataset AQT_I --wandb_login_key YOUR_API_KEY
```

After the run is complete, you can view the detailed results, including metrics and plots, on your [wandb dashboard](https://wandb.ai/).
