import numpy as np
from sklearn.linear_model import Lasso
import time

class LASSO:
    def __init__(self, alpha=0.01):
        self.model = Lasso(alpha=alpha, max_iter=10000)
        self.feature_importance = None
        
    def fit(self, X, y):
        """Fit LASSO model"""
        start_time = time.time()
        
        # Fit model
        self.model.fit(X, y)
        
        # Get feature importance
        self.feature_importance = np.abs(self.model.coef_)
        
        training_time = time.time() - start_time
        return training_time
    
    def predict(self, X):
        """Make predictions using fitted model"""
        return self.model.predict(X)
    
    def get_params(self):
        """Get model parameters"""
        return {
            'alpha': self.model.alpha,
            'n_features': len(self.model.coef_),
            'n_nonzero_features': np.sum(self.model.coef_ != 0)
        }
    
    def get_feature_importance(self):
        """Get feature importance scores"""
        return self.feature_importance 