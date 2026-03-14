"""
Script: 04_attention_extraction.py
Description: Extracts and visualizes attention weights from the final transformer 
             layer, focusing on basic amino acids (R/K) in Nuclear Localization Signals.
"""

import argparse
import logging
import torch
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_attention(sequence: str, model, tokenizer, device):
    """Extracts averaged attention weights across all heads in the final layer."""
    model.eval()
    inputs = tokenizer(sequence, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        # Request attention outputs
        outputs = model(**inputs, output_attentions=True)
    
    # outputs.attentions is a tuple of all layers. Get the last layer.
    # Shape: (batch_size, num_heads, seq_len, seq_len)
    last_layer_attention = outputs.attentions[-1]
    
    # Average across all attention heads for the [CLS] token's attention to other tokens
    # [CLS] is at index 0
    cls_attention = last_layer_attention[0, :, 0, :].mean(dim=0).cpu().numpy()
    
    # Get tokens to map attention weights to amino acids
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    return tokens, cls_attention

def main():
    parser = argparse.ArgumentParser(description="Attention Mechanism Extraction")
    parser.add_argument("--model_path", type=str, required=True, help="Path to fine-tuned model.")
    parser.add_argument("--sequence", type=str, required=True, help="Target protein sequence to analyze.")
    parser.add_argument("--output_file", type=str, default="./results/attention_map.png")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logging.info("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path).to(device)
    
    logging.info("Extracting attention weights...")
    tokens, attention_weights = extract_attention(args.sequence, model, tokenizer, device)
    
    # Filter out special tokens like <cls>, <eos>, <pad>
    valid_indices = [i for i, t in enumerate(tokens) if t not in ['<cls>', '<eos>', '<pad>']]
    amino_acids = [tokens[i] for i in valid_indices]
    weights = [attention_weights[i] for i in valid_indices]
    
    # Identify R and K residues for NLS tracking
    colors = ['firebrick' if aa in ['R', 'K'] else 'cadetblue' for aa in amino_acids]
    
    logging.info("Generating attention plot...")
    plt.figure(figsize=(15, 4))
    plt.bar(range(len(amino_acids)), weights, color=colors)
    plt.xticks(range(len(amino_acids)), amino_acids, fontsize=8)
    plt.xlabel("Amino Acid Sequence")
    plt.ylabel("Attention Weight (Impact on Decision)")
    plt.title("Transformer Attention Map: Highlighting NLS (R/K) Exploitation")
    plt.tight_layout()
    
    plt.savefig(args.output_file, dpi=300)
    logging.info(f"Attention map saved to {args.output_file}")

if __name__ == "__main__":
    main()