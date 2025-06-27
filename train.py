import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import yaml
import os
import argparse
import wandb
from models.egt import EGT
from utils.preprocessing import preprocess_snp_data, create_data_loaders
from utils.evaluation import evaluate_model, cross_validate
import logging

def setup_logging(log_dir='logs'):
    """Setup logging configuration"""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'train.log')),
            logging.StreamHandler()
        ]
    )

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def train_model(model, train_loader, val_loader, criterion, optimizer, device, config):
    """Train the model"""
    best_val_mse = float('inf')
    patience = config['training']['patience']
    patience_counter = 0
    
    for epoch in range(config['training']['epochs']):
        # Training phase
        model.train()
        train_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            if batch_idx % config['training']['log_interval'] == 0:
                logging.info(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                           f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
        
        avg_train_loss = train_loss / len(train_loader)

        # Validation phase
        model.eval()
        val_metrics = evaluate_model(model, val_loader, device)
        val_mse = val_metrics['mse']
        
        logging.info(f'Validation Epoch: {epoch}\tMSE: {val_mse:.6f}\tR2: {val_metrics["r2"]:.6f}')
        
        # Log metrics to wandb
        wandb.log({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_mse": val_mse,
            "val_r2": val_metrics['r2']
        })

        # Early stopping
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            patience_counter = 0
            # Save best model
            model_path = os.path.join(config['model']['save_dir'], config['model']['model_filename'])
            torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info(f'Early stopping triggered after {epoch + 1} epochs')
                break

def main():
    parser = argparse.ArgumentParser(description="Train EGT model.")
    parser.add_argument('--config', type=str, default='configs/train_config.yaml',
                        help="Path to the training configuration file.")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    
    # Setup logging
    setup_logging(config['logging']['log_dir'])
    
    # Initialize wandb
    wandb.init(
        project="EGT",
        config=config,
        name=f"train-{os.path.basename(args.config).replace('.yaml', '')}"
    )

    # Set random seed for reproducibility
    torch.manual_seed(config['training']['seed'])
    np.random.seed(config['training']['seed'])
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')
    
    # Load and preprocess data
    snp_data = np.load(config['data']['snp_data_path'], allow_pickle=True)
    trait_data = np.load(config['data']['trait_data_path'], allow_pickle=True)
    
    train_snp, test_snp, train_trait, test_trait, _ = preprocess_snp_data(
        snp_data, trait_data,
        train_ratio=config['data']['train_ratio'],
        random_state=config['training']['seed']
    )
    
    train_loader, test_loader = create_data_loaders(
        train_snp, test_snp, train_trait, test_trait,
        batch_size=config['training']['batch_size']
    )
    
    # Initialize model
    model = EGT(
        input_dim=config['model']['input_dim'],
        encoding_dim=config['model']['encoding_dim'],
        num_heads=config['model']['num_heads'],
        num_layers=config['model']['num_layers'],
        dropout=config['model']['dropout']
    ).to(device)
    
    # Create save directory if it doesn't exist
    os.makedirs(config['model']['save_dir'], exist_ok=True)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Train model
    train_model(model, train_loader, test_loader, criterion, optimizer, device, config)
    
    # Perform cross-validation
    if config['training'].get('cross_validation', False):
        cv_results = cross_validate(
            lambda: EGT(
                input_dim=config['model']['input_dim'],
                encoding_dim=config['model']['encoding_dim'],
                num_heads=config['model']['num_heads'],
                num_layers=config['model']['num_layers'],
                dropout=config['model']['dropout']
            ),
            train_loader, test_loader, device,
            num_folds=config['training']['num_folds']
        )
    
        # Log cross-validation results
        logging.info('Cross-validation results:')
        cv_log_data = {}
        for metric in ['mse', 'r2', 'spearman']:
            mean_val = cv_results[f"{metric}_mean"]
            std_val = cv_results[f"{metric}_std"]
            logging.info(f'{metric}: {mean_val:.6f} ± {std_val:.6f}')
            cv_log_data[f"cv_{metric}_mean"] = mean_val
            cv_log_data[f"cv_{metric}_std"] = std_val
        wandb.log(cv_log_data)
    
    wandb.finish()

if __name__ == '__main__':
    main() 