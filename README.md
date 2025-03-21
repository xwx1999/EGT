# EGT: Enhanced Genomic Transformer for Animal Quantitative Trait Prediction

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://badge.fury.io/py/egt.svg)](https://badge.fury.io/py/egt)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

EGT (Enhanced Genomic Transformer) is a deep learning model for predicting animal quantitative traits. The model combines autoencoders with self-attention mechanisms to effectively process SNP data and improve prediction accuracy.

## Key Features

- 🧬 Complete SNP sequence preprocessing pipeline
- 🔄 Autoencoder-based dimensionality reduction
- 🧠 Self-attention mechanism for sequence processing
- 📊 Combined MSE and correlation coefficient loss function
- 🎯 Multi-trait regression prediction support
- 📈 Performance comparison with state-of-the-art methods
- 📊 Rich visualization tools
- 📝 Comprehensive logging system
- 💾 Smart checkpoint management

## Installation

### Via pip

```bash
pip install egt
```

### From source

1. Clone the repository:
```bash
git clone https://github.com/xwx1999/EGT.git
cd EGT
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install the package:
```bash
pip install -e .
```

## Project Structure

```
EGT/
├── data/                   # Data directory
│   ├── raw/               # Raw data
│   └── processed/         # Processed data
├── models/                 # Model implementations
│   ├── egt.py             # EGT model implementation
│   ├── autoencoder.py     # Autoencoder implementation
│   └── attention.py       # Attention mechanism
├── utils/                  # Utility functions
│   ├── preprocessing.py   # Data preprocessing
│   ├── evaluation.py      # Evaluation metrics
│   ├── logger.py          # Logging utilities
│   ├── checkpoint.py      # Checkpoint management
│   └── visualization.py   # Visualization tools
├── configs/               # Configuration files
│   └── default_config.yaml
├── tests/                 # Test cases
├── examples/              # Example code
├── docs/                  # Documentation
├── setup.py              # Installation configuration
├── requirements.txt      # Project dependencies
└── README.md            # Project documentation
```

## Quick Start

1. Prepare data:
```python
from egt.utils.preprocessing import SNPPreprocessor

preprocessor = SNPPreprocessor()
data = preprocessor.process("data/raw/snp_data.csv")
```

2. Train the model:
```python
from egt.models import EGT
from egt.utils.trainer import Trainer

model = EGT(input_dim=1000, hidden_dim=256)
trainer = Trainer(model)
trainer.train(data)
```

3. Evaluate the model:
```python
from egt.utils.evaluation import Evaluator

evaluator = Evaluator(model)
metrics = evaluator.evaluate(test_data)
```

4. Visualize results:
```python
from egt.utils.visualization import Visualizer

visualizer = Visualizer()
visualizer.plot_training_history(trainer.history)
visualizer.plot_correlation_matrix(true_values, predicted_values)
```

## Configuration

The configuration file is located at `configs/default_config.yaml` and includes the following sections:

- Data configuration: data paths, batch sizes, etc.
- Model configuration: model architecture parameters
- Training configuration: learning rate, optimizer, etc.
- Evaluation configuration: evaluation metrics and visualization options
- Logging configuration: log level and format

## Contributing

1. Fork this repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Citation

If you use this code in your research, please cite our paper:

```
@article{wang2024egt,
  title={Enhanced Genomic Transformer for Animal Quantitative Trait Prediction},
  author={Wang, Xiwang},
  journal={arXiv preprint arXiv:2403.xxxxx},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

- Author: Xiwang Wang
- Email: xwx1999@gmail.com
- Project Link: [https://github.com/xwx1999/EGT](https://github.com/xwx1999/EGT) 