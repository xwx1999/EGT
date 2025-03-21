import numpy as np
from sklearn.linear_model import LinearRegression
import time

class GBLUP:
    def __init__(self):
        self.model = None
        self.kinship_matrix = None
        
    def compute_kinship_matrix(self, X):
        """Compute genomic relationship matrix (GRM)"""
        # Standardize SNP data
        X_std = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        # Compute GRM
        n = X_std.shape[1]
        self.kinship_matrix = np.dot(X_std, X_std.T) / n
        return self.kinship_matrix
    
    def fit(self, X, y):
        """Fit GBLUP model"""
        start_time = time.time()
        
        # Compute kinship matrix
        self.compute_kinship_matrix(X)
        
        # Fit model using mixed model equations
        n = X.shape[0]
        lambda_value = 0.1  # Regularization parameter
        
        # Construct mixed model equations
        Z = np.eye(n)
        K = self.kinship_matrix
        I = np.eye(n)
        
        # Solve mixed model equations
        beta = np.linalg.inv(Z.T @ Z + lambda_value * np.linalg.inv(K)) @ Z.T @ y
        
        self.model = beta
        training_time = time.time() - start_time
        return training_time
    
    def predict(self, X):
        """Make predictions using fitted model"""
        # Compute kinship matrix for test data
        X_std = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        K_test = np.dot(X_std, X_std.T) / X_std.shape[1]
        
        # Make predictions
        predictions = K_test @ self.model
        return predictions
    
    def get_params(self):
        """Get model parameters"""
        return {
            'kinship_matrix_shape': self.kinship_matrix.shape if self.kinship_matrix is not None else None,
            'model_shape': self.model.shape if self.model is not None else None
        } 