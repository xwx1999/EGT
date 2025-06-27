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

def preprocess_snp_data(snp_data, trait_data, train_ratio=0.8, random_state=42):
    """
    Split SNP and trait data into training and testing sets.
    Returns the split data and the indices of the test set.
    """
    indices = np.arange(snp_data.shape[0])
    
    train_snp, test_snp, train_trait, test_trait, train_indices, test_indices = train_test_split(
        snp_data, trait_data, indices, train_size=train_ratio, random_state=random_state
    )
    
    return train_snp, test_snp, train_trait, test_trait, test_indices

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
    # Convert to float to ensure consistency
    train_snp = train_snp.astype(np.float32)
    test_snp = test_snp.astype(np.float32)
    train_trait = train_trait.astype(np.float32)
    test_trait = test_trait.astype(np.float32)

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