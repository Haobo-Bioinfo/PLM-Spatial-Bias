"""
Script: 02_model_finetuning.py
Description: Fine-tunes Protein Language Models (ESM-2, ProtBERT) for RNA-binding 
             protein (RBP) prediction using the HuggingFace Trainer API.
"""

import os
import argparse
import logging
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_score, recall_score, average_precision_score, accuracy_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    set_seed
)
from datasets import Dataset

# Configure professional logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def format_sequence(sequence: str, model_name: str) -> str:
    """
    Formats the protein sequence based on the model's tokenizer requirements.
    ProtBERT requires space-separated amino acids, while ESM-2 does not.
    """
    if "prot_bert" in model_name.lower():
        return " ".join(list(sequence))
    return sequence

def prepare_hf_dataset(csv_path: str, tokenizer, model_name: str, max_length: int = 1024) -> Dataset:
    """
    Loads a CSV dataset, formats sequences, and tokenizes them into a HuggingFace Dataset.
    """
    logging.info(f"Loading and tokenizing dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Map RBP targets (Assuming 'Target' column exists where 1=RBP, 0=Non-RBP)
    # Adjust column names based on your actual CSV structure
    if 'Target' not in df.columns:
        df['Target'] = df['Subcellular_Localization'].apply(lambda x: 1 if x == 'RBP' else 0)
        
    df['formatted_seq'] = df['Sequence'].apply(lambda seq: format_sequence(seq, model_name))
    
    hf_dataset = Dataset.from_pandas(df[['formatted_seq', 'Target']])
    hf_dataset = hf_dataset.rename_column("Target", "labels")
    
    def tokenize_function(examples):
        return tokenizer(
            examples["formatted_seq"],
            padding="max_length",
            truncation=True,
            max_length=max_length
        )
        
    tokenized_dataset = hf_dataset.map(tokenize_function, batched=True, remove_columns=["formatted_seq"])
    return tokenized_dataset

def compute_metrics(eval_pred):
    """
    Custom metrics computation prioritizing Average Precision (AP) for imbalanced datasets,
    as mathematically proven to be more informative (Saito & Rehmsmeier, 2015).
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    probabilities = torch.nn.functional.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
    
    acc = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, zero_division=0)
    recall = recall_score(labels, predictions, zero_division=0)
    ap = average_precision_score(labels, probabilities)
    
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "average_precision": ap
    }

def main():
    parser = argparse.ArgumentParser(description="PLM Fine-tuning for RBP Prediction")
    parser.add_argument("--model_name_or_path", type=str, default="facebook/esm2_t12_35M_UR50D", 
                        help="HuggingFace model repository or local path.")
    parser.add_argument("--train_data", type=str, default="./data/train.csv", help="Path to training data.")
    parser.add_argument("--val_data", type=str, default="./data/val.csv", help="Path to validation data.")
    parser.add_argument("--output_dir", type=str, default="./saved_models/esm2_35m_finetuned", help="Output directory.")
    parser.add_argument("--max_len", type=int, default=1024, help="Maximum token length.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size per device.")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for AdamW.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    
    args = parser.parse_args()
    set_seed(args.seed)
    
    logging.info(f"Initializing Tokenizer and Model: {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path, 
        num_labels=2,
        problem_type="single_label_classification"
    )
    
    train_dataset = prepare_hf_dataset(args.train_data, tokenizer, args.model_name_or_path, args.max_len)
    val_dataset = prepare_hf_dataset(args.val_data, tokenizer, args.model_name_or_path, args.max_len)
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        load_best_model_at_end=True,
        metric_for_best_model="average_precision",
        greater_is_better=True,
        logging_dir="./logs",
        logging_steps=10,
        seed=args.seed
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)]
    )
    
    logging.info("Starting model fine-tuning...")
    trainer.train()
    
    logging.info(f"Training complete. Saving best model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    logging.info("Pipeline execution finished successfully.")

if __name__ == "__main__":
    main()