import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import spearmanr
import torch
import time

def calculate_mse(y_true, y_pred):
    """Calculate Mean Squared Error"""
    return mean_squared_error(y_true, y_pred)

def calculate_r2(y_true, y_pred):
    """Calculate R-squared score"""
    return r2_score(y_true, y_pred)

def calculate_spearman(y_true, y_pred):
    """Calculate Spearman rank correlation coefficient"""
    return spearmanr(y_true, y_pred)[0]

def calculate_combined_loss(y_true, y_pred, alpha=0.5):
    """
    Calculate combined loss function with MSE and correlation coefficient
    
    Args:
        y_true: true values
        y_pred: predicted values
        alpha: scaling factor between MSE and correlation coefficient
    
    Returns:
        combined loss value
    """
    mse = calculate_mse(y_true, y_pred)
    r2 = calculate_r2(y_true, y_pred)
    return alpha * mse + (1 - alpha) * (1 - r2)

def evaluate_model(model, test_loader, device):
    """
    Evaluate model performance on test set
    
    Args:
        model: PyTorch model
        test_loader: PyTorch DataLoader for test set
        device: device to run evaluation on
    
    Returns:
        dict containing evaluation metrics
    """
    model.eval()
    all_preds = []
    all_targets = []
    total_time = 0
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            
            # Measure inference time
            start_time = time.time()
            output = model(data)
            end_time = time.time()
            total_time += end_time - start_time
            
            all_preds.extend(output.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # Calculate metrics
    mse = calculate_mse(all_targets, all_preds)
    r2 = calculate_r2(all_targets, all_preds)
    spearman = calculate_spearman(all_targets, all_preds)
    avg_inference_time = total_time / len(test_loader)
    
    return {
        'mse': mse,
        'r2': r2,
        'spearman': spearman,
        'inference_time': avg_inference_time
    }

def cross_validate(model_class, train_loader, val_loader, device, num_folds=5):
    """
    Perform k-fold cross-validation
    
    Args:
        model_class: PyTorch model class
        train_loader: PyTorch DataLoader for training set
        val_loader: PyTorch DataLoader for validation set
        device: device to run training on
        num_folds: number of folds for cross-validation
    
    Returns:
        dict containing cross-validation results
    """
    cv_results = {
        'mse': [],
        'r2': [],
        'spearman': []
    }
    
    for fold in range(num_folds):
        # Initialize model for this fold
        model = model_class().to(device)
        
        # Train model
        model.train()
        for epoch in range(300):  # 300 epochs as mentioned in the paper
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                model.optimizer.zero_grad()
                output = model(data)
                loss = model.criterion(output, target)
                loss.backward()
                model.optimizer.step()
        
        # Evaluate on validation set
        metrics = evaluate_model(model, val_loader, device)
        cv_results['mse'].append(metrics['mse'])
        cv_results['r2'].append(metrics['r2'])
        cv_results['spearman'].append(metrics['spearman'])
    
    # Calculate mean and std of metrics
    for metric in cv_results:
        cv_results[f'{metric}_mean'] = np.mean(cv_results[metric])
        cv_results[f'{metric}_std'] = np.std(cv_results[metric])
    
    return cv_results 