import os
import yaml
import torch
from egt.models import EGT
from egt.utils.preprocessing import SNPPreprocessor
from egt.utils.trainer import Trainer
from egt.utils.evaluation import Evaluator
from egt.utils.visualization import Visualizer
from egt.utils.logger import get_logger

def load_config(config_path):
    """Load configuration file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main():
    # Setup logging
    logger = get_logger('egt_example')
    logger.info("Starting EGT example...")
    
    # Load configuration
    config = load_config('configs/default_config.yaml')
    logger.info("Configuration loaded")
    
    # Data preprocessing
    preprocessor = SNPPreprocessor()
    data = preprocessor.process(
        input_path=config['data']['input_dir'],
        output_path=config['data']['output_dir']
    )
    logger.info("Data preprocessing completed")
    
    # Create model
    model = EGT(
        input_dim=config['model']['input_dim'],
        hidden_dim=config['model']['hidden_dim'],
        num_layers=config['model']['num_layers'],
        num_heads=config['model']['num_heads'],
        dropout=config['model']['dropout']
    )
    logger.info("Model created")
    
    # Train model
    trainer = Trainer(
        model=model,
        config=config['training']
    )
    trainer.train(data)
    logger.info("Model training completed")
    
    # Evaluate model
    evaluator = Evaluator(model)
    metrics = evaluator.evaluate(data['test'])
    logger.info(f"Model evaluation results: {metrics}")
    
    # Visualize results
    visualizer = Visualizer()
    visualizer.plot_training_history(trainer.history)
    visualizer.plot_correlation_matrix(
        data['test']['true_values'],
        data['test']['predicted_values']
    )
    visualizer.plot_scatter_plots(
        data['test']['true_values'],
        data['test']['predicted_values']
    )
    logger.info("Visualization results saved")

if __name__ == '__main__':
    main() 