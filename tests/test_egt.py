import unittest
import torch
import numpy as np
from egt.models import EGT
from egt.utils.preprocessing import SNPPreprocessor
from egt.utils.evaluation import Evaluator
from egt.utils.visualization import Visualizer

class TestEGT(unittest.TestCase):
    def setUp(self):
        """Setup for tests"""
        # Set random seeds for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Create test data
        self.batch_size = 32
        self.input_dim = 1000
        self.output_dim = 5
        
        self.test_features = torch.randn(self.batch_size, self.input_dim)
        self.test_labels = torch.randn(self.batch_size, self.output_dim)
        
        # Create model
        self.model = EGT(
            input_dim=self.input_dim,
            hidden_dim=256,
            num_layers=4,
            num_heads=8,
            dropout=0.1
        )
    
    def test_model_architecture(self):
        """Test model architecture"""
        # Test input dimension
        self.assertEqual(self.model.input_dim, self.input_dim)
        
        # Test output dimension
        output = self.model(self.test_features)
        self.assertEqual(output.shape, (self.batch_size, self.output_dim))
        
        # Test model parameters
        self.assertTrue(hasattr(self.model, 'encoder'))
        self.assertTrue(hasattr(self.model, 'decoder'))
        self.assertTrue(hasattr(self.model, 'attention'))
    
    def test_model_forward(self):
        """Test model forward pass"""
        # Test forward pass
        output = self.model(self.test_features)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.output_dim))
        
        # Check for NaN values
        self.assertFalse(torch.isnan(output).any())
        
        # Check for infinite values
        self.assertFalse(torch.isinf(output).any())
    
    def test_preprocessing(self):
        """Test data preprocessing"""
        preprocessor = SNPPreprocessor()
        
        # Create test data
        test_data = np.random.rand(100, self.input_dim)
        
        # Test preprocessing
        processed_data = preprocessor.process(test_data)
        
        # Check processed data
        self.assertIsInstance(processed_data, dict)
        self.assertTrue('features' in processed_data)
        self.assertTrue('labels' in processed_data)
        
        # Check data shape
        self.assertEqual(processed_data['features'].shape[1], self.input_dim)
    
    def test_evaluation(self):
        """Test evaluation metrics"""
        evaluator = Evaluator(self.model)
        
        # Create test data
        test_data = {
            'features': self.test_features,
            'labels': self.test_labels
        }
        
        # Test evaluation
        metrics = evaluator.evaluate(test_data)
        
        # Check evaluation metrics
        self.assertIsInstance(metrics, dict)
        self.assertTrue('mse' in metrics)
        self.assertTrue('r2' in metrics)
        self.assertTrue('pearson' in metrics)
        
        # Check metric values are reasonable
        self.assertGreaterEqual(metrics['r2'], -1)
        self.assertLessEqual(metrics['r2'], 1)
        self.assertGreaterEqual(metrics['pearson'], -1)
        self.assertLessEqual(metrics['pearson'], 1)
    
    def test_visualization(self):
        """Test visualization functions"""
        visualizer = Visualizer()
        
        # Create test data
        true_values = np.random.rand(100, self.output_dim)
        predicted_values = np.random.rand(100, self.output_dim)
        
        # Test plotting functions
        try:
            visualizer.plot_correlation_matrix(true_values, predicted_values)
            visualizer.plot_scatter_plots(true_values, predicted_values)
        except Exception as e:
            self.fail(f"Visualization test failed: {str(e)}")
    
    def test_model_save_load(self):
        """Test model save and load"""
        # Save model
        torch.save(self.model.state_dict(), 'test_model.pt')
        
        # Create new model
        new_model = EGT(
            input_dim=self.input_dim,
            hidden_dim=256,
            num_layers=4,
            num_heads=8,
            dropout=0.1
        )
        
        # Load model
        new_model.load_state_dict(torch.load('test_model.pt'))
        
        # Test loaded model
        output1 = self.model(self.test_features)
        output2 = new_model(self.test_features)
        
        # Check outputs are identical
        self.assertTrue(torch.allclose(output1, output2))

if __name__ == '__main__':
    unittest.main() 