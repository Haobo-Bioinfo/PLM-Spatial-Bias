"""
Script: 01_data_preprocessing.py
Description: Preprocesses raw protein sequences, applies quality control filters, 
             and performs spatially stratified dataset partitioning.
"""

import pandas as pd
import numpy as np
import argparse
import logging
from sklearn.model_selection import train_test_split
import re

# Configure professional logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def clean_and_filter_sequences(df: pd.DataFrame, min_len: int, max_len: int) -> pd.DataFrame:
    """
    Applies quality control to protein sequences by removing ambiguous amino acids
    and filtering by sequence length.
    """
    initial_count = len(df)
    
    # 1. Drop rows with missing sequences or annotations
    df = df.dropna(subset=['Sequence', 'Subcellular_Localization'])
    
    # 2. Filter out sequences with non-standard/ambiguous amino acids (B, Z, X, J, O, U)
    ambiguous_pattern = re.compile(r'[BZXJOU]')
    df = df[~df['Sequence'].str.contains(ambiguous_pattern, na=False)]
    
    # 3. Apply length constraints
    seq_lengths = df['Sequence'].str.len()
    df = df[(seq_lengths >= min_len) & (seq_lengths <= max_len)]
    
    # 4. Remove duplicate entries based on Sequence
    df = df.drop_duplicates(subset=['Sequence'])
    
    final_count = len(df)
    logging.info(f"Quality Control complete. Retained {final_count} out of {initial_count} sequences.")
    return df

def stratified_partitioning(df: pd.DataFrame, test_size: float = 0.1, val_size: float = 0.1, random_state: int = 42):
    """
    Performs stratified splitting into Train, Validation, and Test sets based on subcellular localization.
    """
    logging.info("Initiating stratified dataset partitioning...")
    
    # First split: Train vs (Validation + Test)
    train_df, temp_df = train_test_split(
        df, 
        test_size=(test_size + val_size), 
        stratify=df['Subcellular_Localization'], 
        random_state=random_state
    )
    
    # Second split: Validation vs Test
    relative_test_size = test_size / (test_size + val_size)
    val_df, test_df = train_test_split(
        temp_df, 
        test_size=relative_test_size, 
        stratify=temp_df['Subcellular_Localization'], 
        random_state=random_state
    )
    
    logging.info(f"Partitioning successful. Splits -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    return train_df, val_df, test_df

def main():
    parser = argparse.ArgumentParser(description="Protein Sequence Preprocessing Pipeline")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the raw CD-HIT reduced CSV dataset.")
    parser.add_argument("--output_dir", type=str, default="./data", help="Directory to save the partitioned datasets.")
    parser.add_argument("--min_len", type=int, default=50, help="Minimum sequence length.")
    parser.add_argument("--max_len", type=int, default=1022, help="Maximum sequence length.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    
    args = parser.parse_args()
    
    # Load dataset
    logging.info(f"Loading raw dataset from {args.input_path}...")
    try:
        raw_df = pd.read_csv(args.input_path)
    except FileNotFoundError:
        logging.error(f"File not found: {args.input_path}")
        return

    # Process data
    clean_df = clean_and_filter_sequences(raw_df, args.min_len, args.max_len)
    
    # Partition data
    train_df, val_df, test_df = stratified_partitioning(clean_df, random_state=args.seed)
    
    # Save partitioned datasets
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    
    train_df.to_csv(f"{args.output_dir}/train.csv", index=False)
    val_df.to_csv(f"{args.output_dir}/val.csv", index=False)
    test_df.to_csv(f"{args.output_dir}/test.csv", index=False)
    logging.info(f"Datasets successfully saved to {args.output_dir}/")

if __name__ == "__main__":
    main()