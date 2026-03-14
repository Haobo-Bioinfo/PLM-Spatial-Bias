# PLM-Spatial-Bias: Unmasking Spatial Shortcuts in Protein Language Models

Official codebase for benchmarking spatial bias in Protein Language Models (ESM-2, ProtBERT). We reveal that PLMs achieve high RBP prediction accuracy by exploiting subcellular spatial shortcuts (pI and NLS) rather than authentic functional interactions. Includes tools for stratified dataset partitioning, model fine-tuning, UMAP visualization, and attention map extraction.

## 📁 Repository Structure

* `data/`: Contains the homology-reduced (CD-HIT) and spatially stratified datasets (Train/Val/Test).
* `notebooks/`: Jupyter/Colab notebooks for ESM-2 and ProtBERT fine-tuning.
* `scripts/`: Python scripts for UMAP latent space visualization and attention weight extraction.
* `results/`: Raw output metrics, precision-recall curve data, and generated figures.

## 🚀 Quick Start

*(Code and detailed inference instructions will be uploaded shortly.)*

## 📝 Citation
If you find this benchmark useful, please consider citing our manuscript:
> [Your Name], et al. "Localization is Not Function: Unmasking Spatial Shortcuts and Feature Entanglement in Protein Language Models for RBP Prediction." (Under Review at Briefings in Bioinformatics, 2026).
