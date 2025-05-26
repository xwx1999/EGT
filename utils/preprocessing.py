import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader

class SNPDataset(Dataset):
    def __init__(self, snp_data, trait_data):
        self.snp_data = torch.FloatTensor(snp_data)
        self.trait_data = torch.FloatTensor(trait_data)
        
    def __len__(self):
        return len(self.snp_data)
    
    def __getitem__(self, idx):
        return self.snp_data[idx], self.trait_data[idx]

def preprocess_snp_data(snp_data, trait_data, train_ratio=0.7, random_state=42):
    """
    Preprocess SNP data and split into train/test sets
    
    Args:
        snp_data: numpy array of SNP data
        trait_data: numpy array of trait data
        train_ratio: ratio of training data
        random_state: random seed for reproducibility
    
    Returns:
        train_snp, test_snp, train_trait, test_trait
    """
    # Split data
    train_snp, test_snp, train_trait, test_trait = train_test_split(
        snp_data, trait_data, train_size=train_ratio, random_state=random_state
    )
    
    # Scale SNP data
    scaler = StandardScaler()
    train_snp = scaler.fit_transform(train_snp)
    test_snp = scaler.transform(test_snp)
    
    return train_snp, test_snp, train_trait, test_trait

def create_data_loaders(train_snp, test_snp, train_trait, test_trait, batch_size=32):
    """
    Create PyTorch DataLoaders for training and testing
    
    Args:
        train_snp: training SNP data
        test_snp: test SNP data
        train_trait: training trait data
        test_trait: test trait data
        batch_size: batch size for training
    
    Returns:
        train_loader, test_loader
    """
    train_dataset = SNPDataset(train_snp, train_trait)
    test_dataset = SNPDataset(test_snp, test_trait)
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )
    
    return train_loader, test_loader

def impute_missing_values(snp_data, method='mean'):
    """
    Impute missing values in SNP data
    
    Args:
        snp_data: numpy array of SNP data
        method: imputation method ('mean' or 'median')
    
    Returns:
        imputed SNP data
    """
    if method == 'mean':
        imputer = np.nanmean(snp_data, axis=0)
    else:
        imputer = np.nanmedian(snp_data, axis=0)
        
    # Replace NaN values with imputed values
    snp_data = np.where(np.isnan(snp_data), imputer, snp_data)
    return snp_data

def remove_samples_with_missing_data(snp_data, trait_data, threshold=0.2):
    """
    Remove samples with too many missing values
    
    Args:
        snp_data: numpy array of SNP data
        trait_data: numpy array of trait data
        threshold: maximum allowed proportion of missing values
    
    Returns:
        filtered SNP and trait data
    """
    # Calculate proportion of missing values for each sample
    missing_prop = np.isnan(snp_data).mean(axis=1)
    
    # Keep samples with missing values below threshold
    mask = missing_prop < threshold
    snp_data = snp_data[mask]
    trait_data = trait_data[mask]
    
    return snp_data, trait_data 

class SNPPreprocessor:
    """
    A class for preprocessing SNP data
    
    This class provides functionality to load, clean, and process SNP data
    for use in the Enhanced Genomic Transformer (EGT) model.
    """
    
    def __init__(self, missing_threshold=0.2, imputation_method='mean', train_ratio=0.7, random_state=42):
        """
        Initialize the SNPPreprocessor
        
        Args:
            missing_threshold: maximum allowed proportion of missing values (default: 0.2)
            imputation_method: method for imputing missing values ('mean' or 'median', default: 'mean')
            train_ratio: ratio of training data (default: 0.7)
            random_state: random seed for reproducibility (default: 42)
        """
        self.missing_threshold = missing_threshold
        self.imputation_method = imputation_method
        self.train_ratio = train_ratio
        self.random_state = random_state
        self.scaler = StandardScaler()
        
    def process(self, snp_file_path, trait_file_path=None, return_type='numpy'):
        """
        Process SNP data from a CSV file
        
        Args:
            snp_file_path: path to the SNP data CSV file
            trait_file_path: path to the trait data CSV file (if None, assumes trait data is in the last column of SNP data)
            return_type: type of return data ('numpy', 'torch', 'dataloader', or 'split')
            
        Returns:
            Processed SNP data in the specified format
        """
        # Load SNP data from CSV
        snp_data = pd.read_csv(snp_file_path)
        
        # Extract trait data if not provided separately
        if trait_file_path is None:
            # Assume the last column is the trait data
            trait_data = snp_data.iloc[:, -1].values.reshape(-1, 1)
            snp_data = snp_data.iloc[:, :-1].values
        else:
            # Load trait data from separate file
            trait_df = pd.read_csv(trait_file_path)
            trait_data = trait_df.values
            snp_data = snp_data.values
            
        # Remove samples with too many missing values
        snp_data, trait_data = remove_samples_with_missing_data(
            snp_data, trait_data, threshold=self.missing_threshold
        )
        
        # Impute remaining missing values
        snp_data = impute_missing_values(snp_data, method=self.imputation_method)
        
        if return_type == 'numpy':
            return snp_data, trait_data
        
        elif return_type == 'split':
            # Split data into train/test sets and preprocess
            train_snp, test_snp, train_trait, test_trait = preprocess_snp_data(
                snp_data, trait_data, train_ratio=self.train_ratio, random_state=self.random_state
            )
            return train_snp, test_snp, train_trait, test_trait
        
        elif return_type == 'torch':
            # Convert to PyTorch tensors
            snp_tensor = torch.FloatTensor(self.scaler.fit_transform(snp_data))
            trait_tensor = torch.FloatTensor(trait_data)
            return snp_tensor, trait_tensor
            
        elif return_type == 'dataloader':
            # Split and create DataLoader objects
            train_snp, test_snp, train_trait, test_trait = preprocess_snp_data(
                snp_data, trait_data, train_ratio=self.train_ratio, random_state=self.random_state
            )
            train_loader, test_loader = create_data_loaders(
                train_snp, test_snp, train_trait, test_trait
            )
            return train_loader, test_loader
        
        else:
            raise ValueError(f"Unknown return_type: {return_type}")
