import numpy as np
import lightgbm as lgb
import time

class LGBM:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=6):
        self.model = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            objective='regression',
            metric='mse',
            n_jobs=-1
        )
        self.feature_importance = None
        
    def fit(self, X, y):
        """Fit LGBM model"""
        start_time = time.time()
        
        # Fit model
        self.model.fit(X, y)
        
        # Get feature importance
        self.feature_importance = self.model.feature_importances_
        
        training_time = time.time() - start_time
        return training_time
    
    def predict(self, X):
        """Make predictions using fitted model"""
        return self.model.predict(X)
    
    def get_params(self):
        """Get model parameters"""
        return {
            'n_estimators': self.model.n_estimators,
            'learning_rate': self.model.learning_rate,
            'max_depth': self.model.max_depth
        }
    
    def get_feature_importance(self):
        """Get feature importance scores"""
        return self.feature_importance 