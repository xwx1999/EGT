import os
import yaml
import torch
import numpy as np
from egt.models import EGT
from egt.utils.preprocessing import SNPPreprocessor
from egt.utils.trainer import Trainer
from egt.utils.evaluation import Evaluator
from egt.utils.visualization import Visualizer
from egt.utils.logger import get_logger
from egt.utils.checkpoint import CheckpointManager

class AdvancedEGT:
    def __init__(self, config_path):
        self.logger = get_logger('advanced_egt')
        self.config = self._load_config(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device: {self.device}")
        
        # Initialize components
        self.preprocessor = SNPPreprocessor()
        self.model = self._create_model()
        self.trainer = self._create_trainer()
        self.evaluator = Evaluator(self.model)
        self.visualizer = Visualizer()
        self.checkpoint_manager = CheckpointManager(self.config['training']['checkpoint_dir'])
    
    def _load_config(self, config_path):
        """Load configuration file"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _create_model(self):
        """Create model instance"""
        model = EGT(
            input_dim=self.config['model']['input_dim'],
            hidden_dim=self.config['model']['hidden_dim'],
            num_layers=self.config['model']['num_layers'],
            num_heads=self.config['model']['num_heads'],
            dropout=self.config['model']['dropout']
        ).to(self.device)
        return model
    
    def _create_trainer(self):
        """Create trainer instance"""
        return Trainer(
            model=self.model,
            config=self.config['training'],
            device=self.device
        )
    
    def train_with_validation(self, data):
        """Training process with validation"""
        self.logger.info("Starting training process...")
        best_metric = float('-inf')
        patience_counter = 0
        
        for epoch in range(self.config['training']['epochs']):
            # Train one epoch
            train_metrics = self.trainer.train_epoch(data['train'])
            
            # Validate
            val_metrics = self.evaluator.evaluate(data['val'])
            
            # Record metrics
            self.trainer.history['train_loss'].append(train_metrics['loss'])
            self.trainer.history['val_loss'].append(val_metrics['loss'])
            
            # Early stopping check
            if val_metrics['r2'] > best_metric:
                best_metric = val_metrics['r2']
                patience_counter = 0
                self.checkpoint_manager.save_checkpoint(
                    self.model,
                    self.trainer.optimizer,
                    epoch,
                    val_metrics,
                    is_best=True
                )
            else:
                patience_counter += 1
                if patience_counter >= self.config['training']['early_stopping_patience']:
                    self.logger.info("Early stopping triggered, stopping training")
                    break
            
            # Periodic checkpoint saving
            if (epoch + 1) % 5 == 0:
                self.checkpoint_manager.save_checkpoint(
                    self.model,
                    self.trainer.optimizer,
                    epoch,
                    val_metrics
                )
    
    def evaluate_with_confidence(self, data):
        """Evaluation with confidence intervals"""
        self.logger.info("Starting evaluation process...")
        
        # Multiple predictions for confidence interval calculation
        n_samples = 100
        predictions = []
        
        for _ in range(n_samples):
            with torch.no_grad():
                pred = self.model(data['test']['features'])
                predictions.append(pred.cpu().numpy())
        
        predictions = np.array(predictions)
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        # Calculate 95% confidence intervals
        ci_lower = mean_pred - 1.96 * std_pred
        ci_upper = mean_pred + 1.96 * std_pred
        
        # Evaluation metrics
        metrics = self.evaluator.evaluate(data['test'])
        
        return {
            'metrics': metrics,
            'mean_predictions': mean_pred,
            'confidence_intervals': {
                'lower': ci_lower,
                'upper': ci_upper
            }
        }
    
    def visualize_results(self, data, evaluation_results):
        """Visualize results"""
        self.logger.info("Starting visualization process...")
        
        # Plot training history
        self.visualizer.plot_training_history(self.trainer.history)
        
        # Plot correlation matrix
        self.visualizer.plot_correlation_matrix(
            data['test']['true_values'],
            evaluation_results['mean_predictions']
        )
        
        # Plot scatter plots with confidence intervals
        self.visualizer.plot_scatter_plots(
            data['test']['true_values'],
            evaluation_results['mean_predictions'],
            confidence_intervals=evaluation_results['confidence_intervals']
        )
    
    def run(self, data_path):
        """Run complete training and evaluation pipeline"""
        try:
            # Data preprocessing
            data = self.preprocessor.process(data_path)
            self.logger.info("Data preprocessing completed")
            
            # Model training
            self.train_with_validation(data)
            self.logger.info("Model training completed")
            
            # Model evaluation
            evaluation_results = self.evaluate_with_confidence(data)
            self.logger.info(f"Model evaluation results: {evaluation_results['metrics']}")
            
            # Result visualization
            self.visualize_results(data, evaluation_results)
            self.logger.info("Visualization results saved")
            
        except Exception as e:
            self.logger.error(f"Error occurred during execution: {str(e)}")
            raise

def main():
    # Create AdvancedEGT instance
    egt = AdvancedEGT('configs/default_config.yaml')
    
    # Run complete pipeline
    egt.run('data/raw/snp_data.csv')

if __name__ == '__main__':
    main() 