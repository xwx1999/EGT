import argparse
import os
import numpy as np
import pandas as pd

def load_aqt_i(raw_data_path):
    """
    Loads and preprocesses the AQT_I dataset.
    """
    print("Loading AQT_I dataset...")
    genotypes_path = os.path.join(raw_data_path, 'AQT_I', 'genotypes.txt')
    ebvs_path = os.path.join(raw_data_path, 'AQT_I', 'ebvs.txt')

    # Using pandas to load genotypes as it is more robust with headers
    genotypes_df = pd.read_csv(genotypes_path)
    # The first column is the ID, so we skip it.
    genotypes = genotypes_df.iloc[:, 1:].values
    
    ebvs_df = pd.read_csv(ebvs_path)
    ebvs = ebvs_df['ebv1'].values

    return genotypes, ebvs

def load_aqt_ii(raw_data_path):
    """
    Loads and preprocesses the AQT_II dataset.
    """
    print("Loading AQT_II dataset...")
    data_path = os.path.join(raw_data_path, 'AQT_II', 'QTLMAS2010ny012.csv')
    
    df = pd.read_csv(data_path, header=None)
    ebvs = df.iloc[:, 0].values
    genotypes = df.iloc[:, 1:].values
    
    return genotypes, ebvs

def load_aqt_iii(raw_data_path):
    """
    Loads and preprocesses the AQT_III dataset.
    """
    print("Loading AQT_III dataset...")
    genotypes_path = os.path.join(raw_data_path, 'AQT_III', 'genotype.txt')
    ebvs_path = os.path.join(raw_data_path, 'AQT_III', 'mortality_EBV.txt')

    genotypes = np.loadtxt(genotypes_path, skiprows=1) # Assuming space-separated, skip header

    # The EBV file has a peculiar format with 3 blocks of data side-by-side
    # We'll read it as a fixed-width formatted file and take the first block of EBVs.
    # It seems to be space-separated, and we only need the second column of the first block.
    all_ebvs_data = pd.read_csv(ebvs_path, delim_whitespace=True, header=None, skiprows=1)
    
    # This is fragile, depends on the exact spacing.
    # A quick look suggested the columns are 0,1 (block1), 2,3 (block2), 4,5 (block3)
    ebvs = all_ebvs_data[1].values

    # Match the number of samples
    num_genotypes = genotypes.shape[0]
    ebvs = ebvs[:num_genotypes]

    return genotypes, ebvs


def main():
    parser = argparse.ArgumentParser(description="Preprocess genomic data for EGT model.")
    parser.add_argument('--dataset', type=str, required=True, choices=['AQT_I', 'AQT_II', 'AQT_III'],
                        help="Name of the dataset to preprocess.")
    parser.add_argument('--raw_data_dir', type=str, default='data/raw',
                        help="Directory where raw data is stored.")
    parser.add_argument('--processed_data_dir', type=str, default='data/processed',
                        help="Directory where processed data will be saved.")
    
    args = parser.parse_args()

    # Select the appropriate loader function
    if args.dataset == 'AQT_I':
        loader_fn = load_aqt_i
    elif args.dataset == 'AQT_II':
        loader_fn = load_aqt_ii
    elif args.dataset == 'AQT_III':
        loader_fn = load_aqt_iii
    else:
        # This case is already handled by argparse choices, but as a safeguard:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # Load data
    genotypes, ebvs = loader_fn(args.raw_data_dir)

    # Create output directory
    output_dir = os.path.join(args.processed_data_dir, args.dataset)
    os.makedirs(output_dir, exist_ok=True)

    # Save processed data
    genotypes_output_path = os.path.join(output_dir, 'genotypes.npy')
    ebvs_output_path = os.path.join(output_dir, 'ebvs.npy')

    np.save(genotypes_output_path, genotypes)
    np.save(ebvs_output_path, ebvs)

    print(f"Successfully preprocessed dataset {args.dataset}.")
    print(f"Genotypes shape: {genotypes.shape}")
    print(f"EBVs shape: {ebvs.shape}")
    print(f"Processed data saved in {output_dir}")


if __name__ == '__main__':
    main() 