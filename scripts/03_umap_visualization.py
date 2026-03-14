"""
Script: 03_umap_visualization.py
Description: Extracts [CLS] token embeddings from fine-tuned PLMs and performs 
             UMAP dimensionality reduction to visualize spatial feature entanglement.
"""

import argparse
import logging
import torch
import pandas as pd
import numpy as np
import umap.umap_ as umap
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_embeddings(df: pd.DataFrame, model, tokenizer, device, max_len=1024):
    """Passes sequences through the PLM to extract final layer [CLS] embeddings."""
    model.eval()
    embeddings = []
    
    logging.info(f"Extracting embeddings for {len(df)} sequences...")
    with torch.no_grad():
        for seq in df['Sequence']:
            inputs = tokenizer(seq, return_tensors="pt", truncation=True, max_length=max_len, padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs, output_hidden_states=True)
            
            # Extract [CLS] token embedding from the last hidden state
            cls_embedding = outputs.hidden_states[-1][:, 0, :].cpu().numpy().flatten()
            embeddings.append(cls_embedding)
            
    return np.array(embeddings)

def main():
    parser = argparse.ArgumentParser(description="UMAP Visualization of PLM Latent Space")
    parser.add_argument("--model_path", type=str, required=True, help="Path to fine-tuned model.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to test set CSV.")
    parser.add_argument("--output_file", type=str, default="./results/umap_projection.png", help="Output plot path.")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path).to(device)
    
    df = pd.read_csv(args.data_path)
    embeddings = extract_embeddings(df, model, tokenizer, device)
    
    logging.info("Computing UMAP projection (n_neighbors=15, min_dist=0.1)...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    embedding_2d = reducer.fit_transform(embeddings)
    
    df['UMAP_1'] = embedding_2d[:, 0]
    df['UMAP_2'] = embedding_2d[:, 1]
    
    # Plotting
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=df, x='UMAP_1', y='UMAP_2', 
        hue='Subcellular_Localization', palette='Set2', s=50, alpha=0.8
    )
    plt.title("UMAP Projection of PLM Latent Space", fontsize=14)
    plt.tight_layout()
    plt.savefig(args.output_file, dpi=300)
    logging.info(f"UMAP visualization saved to {args.output_file}")

if __name__ == "__main__":
    main()