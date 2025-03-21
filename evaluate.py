import torch
import numpy as np
import yaml
import os
from models.egt import EGT
from utils.preprocessing import preprocess_snp_data, create_data_loaders
from utils.evaluation import evaluate_model
import logging
import matplotlib.pyplot as plt
import seaborn as sns

def setup_logging(log_dir='logs'):
    """Setup logging configuration"""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'evaluate.log')),
            logging.StreamHandler()
        ]
    )

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def plot_predictions(y_true, y_pred, save_path):
    """Plot predicted vs true values"""
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('Predicted vs True Values')
    plt.savefig(save_path)
    plt.close()

def plot_attention_maps(model, test_loader, device, save_dir):
    """Plot attention maps for visualization"""
    model.eval()
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(test_loader):
            if batch_idx == 0:  # Only plot first batch
                data = data.to(device)
                attention_maps = model.get_attention_maps(data)
                
                for layer_idx, attn_map in enumerate(attention_maps):
                    plt.figure(figsize=(10, 8))
                    sns.heatmap(attn_map[0].cpu().numpy(), cmap='viridis')
                    plt.title(f'Attention Map - Layer {layer_idx + 1}')
                    plt.savefig(os.path.join(save_dir, f'attention_map_layer_{layer_idx + 1}.png'))
                    plt.close()
                break

def main():
    # Load configuration
    config = load_config('configs/train_config.yaml')
    
    # Setup logging
    setup_logging(config['logging']['log_dir'])
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')
    
    # Load and preprocess data
    snp_data = np.load(config['data']['snp_data_path'])
    trait_data = np.load(config['data']['trait_data_path'])
    
    _, test_snp, _, test_trait = preprocess_snp_data(
        snp_data, trait_data,
        train_ratio=config['data']['train_ratio'],
        random_state=config['training']['seed']
    )
    
    _, test_loader = create_data_loaders(
        test_snp, test_snp, test_trait, test_trait,
        batch_size=config['training']['batch_size']
    )
    
    # Load model
    model = EGT(
        input_dim=config['model']['input_dim'],
        encoding_dim=config['model']['encoding_dim'],
        num_heads=config['model']['num_heads'],
        num_layers=config['model']['num_layers'],
        dropout=config['model']['dropout']
    ).to(device)
    
    model.load_state_dict(torch.load(os.path.join(config['model']['save_dir'], 'best_model.pt')))
    
    # Evaluate model
    metrics = evaluate_model(model, test_loader, device)
    
    # Log results
    logging.info('Test Results:')
    for metric, value in metrics.items():
        logging.info(f'{metric}: {value:.6f}')
    
    # Create visualization directory
    vis_dir = os.path.join(config['logging']['log_dir'], 'visualizations')
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    
    # Generate predictions for plotting
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            output = model(data)
            all_preds.extend(output.cpu().numpy())
            all_targets.extend(target.numpy())
    
    # Plot predictions
    plot_predictions(
        np.array(all_targets),
        np.array(all_preds),
        os.path.join(vis_dir, 'predictions.png')
    )
    
    # Plot attention maps
    plot_attention_maps(model, test_loader, device, vis_dir)

if __name__ == '__main__':
    main() 