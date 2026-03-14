# PLM-Spatial-Bias: Unmasking Spatial Shortcuts in Protein Language Models
Official codebase for benchmarking spatial bias in Protein Language Models (ESM-2, ProtBERT). We reveal that PLMs achieve high RBP prediction accuracy by exploiting subcellular spatial shortcuts (pI and NLS) rather than authentic functional interactions. Includes tools for stratified dataset partitioning, model fine-tuning, UMAP visualization, and attention map extraction.

## Dependencies & Installation
To run the scripts and notebooks, please ensure you have the required libraries installed. Note that umap-learn is specifically required for latent space projection, and HuggingFace datasets is utilized for optimized data handling:
```bash
pip install torch transformers datasets umap-learn pandas numpy scikit-learn matplotlib seaborn
```
## Repository Structure
* data/: Contains the homology-reduced (CD-HIT) and spatially stratified datasets (Train/Val/Test).
* notebooks/: Jupyter/Colab notebooks containing the interactive workflow and figure generation.
* scripts/: Python execution scripts for preprocessing, fine-tuning, UMAP projection, and attention extraction.
* results/: Raw output metrics, precision-recall curve data, and generated figures.

## Quick Start
For an interactive walkthrough of our findings (including feature entanglement and NLS attention extraction), please refer to the ESM2_Spatial_Bias_Analysis.ipynb located in the notebooks/ directory.

To run the pipeline via command line:
```bash
# 1. Preprocess data
python scripts/01_data_preprocessing.py --input_path ./data/raw_data.csv
# 2. Fine-tune ESM-2 model
python scripts/02_model_finetuning.py --epochs 3 --batch_size 16
```
