import os
import subprocess
import yaml
import argparse
import numpy as np

def run_preprocessing(dataset_name, raw_data_dir='data/raw', processed_data_dir='data/processed'):
    """Runs the data preprocessing script."""
    print(f"--- Running preprocessing for {dataset_name} ---")
    script_path = os.path.join('scripts', 'preprocess_data.py')
    cmd = [
        'python', script_path,
        '--dataset', dataset_name,
        '--raw_data_dir', raw_data_dir,
        '--processed_data_dir', processed_data_dir
    ]
    subprocess.run(cmd, check=True)
    print(f"--- Finished preprocessing for {dataset_name} ---")

def create_dataset_config(dataset_name, base_config_path='configs/train_config.yaml', processed_data_dir='data/processed'):
    """Creates a dataset-specific config file."""
    print(f"--- Creating config for {dataset_name} ---")
    with open(base_config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Update data paths
    processed_dataset_dir = os.path.join(processed_data_dir, dataset_name)
    genotypes_path = os.path.join(processed_dataset_dir, 'genotypes.npy')
    ebvs_path = os.path.join(processed_dataset_dir, 'ebvs.npy')
    config['data']['snp_data_path'] = genotypes_path
    config['data']['trait_data_path'] = ebvs_path

    # Update input_dim from shape of preprocessed data
    genotypes = np.load(genotypes_path)
    config['model']['input_dim'] = genotypes.shape[1]

    # Add a dataset-specific model filename
    config['model']['model_filename'] = f'best_model_{dataset_name}.pt'

    # Create a directory for temporary configs if it doesn't exist
    temp_config_dir = 'configs/temp'
    os.makedirs(temp_config_dir, exist_ok=True)
    
    # Save the new config
    new_config_path = os.path.join(temp_config_dir, f'{dataset_name}_config.yaml')
    with open(new_config_path, 'w') as f:
        yaml.dump(config, f)
    
    print(f"--- Saved config to {new_config_path} ---")
    return new_config_path

def run_training(config_path):
    """Runs the training script."""
    print(f"--- Running training with config: {config_path} ---")
    cmd = ['python', 'train.py', '--config', config_path]
    subprocess.run(cmd, check=True)
    print(f"--- Finished training ---")

def run_evaluation(config_path):
    """Runs the evaluation script."""
    print(f"--- Running evaluation with config: {config_path} ---")
    cmd = ['python', 'evaluate.py', '--config', config_path]
    subprocess.run(cmd, check=True)
    print(f"--- Finished evaluation ---")

def main():
    parser = argparse.ArgumentParser(description="Run a full EGT experiment.")
    parser.add_argument('--dataset', type=str, required=True, choices=['AQT_I', 'AQT_II', 'AQT_III'],
                        help="Name of the dataset to run the experiment on.")
    parser.add_argument('--wandb_login_key', type=str, default=None,
                        help="Your wandb login key. If not provided, it's assumed you are logged in.")

    args = parser.parse_args()

    if args.wandb_login_key:
        print("--- Logging in to wandb ---")
        subprocess.run(['wandb', 'login', args.wandb_login_key], check=True)

    # 1. Preprocess data
    run_preprocessing(args.dataset)

    # 2. Create dataset-specific config
    config_path = create_dataset_config(args.dataset)

    # 3. Train model
    run_training(config_path)

    # 4. Evaluate model
    run_evaluation(config_path)

    print(f"\n--- Experiment for dataset {args.dataset} completed successfully! ---")
    print(f"--- Check your results at https://wandb.ai/ ---")

if __name__ == '__main__':
    main() 