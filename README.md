# 🧬GPCRnet: Interpretable Deep Learning Model to Classify Functional States and to Identify Functional Motifs of different GPCR Classes

This repository provides datasets and codes for GPCR functional state classification using Graph Neural Networks (GNNs). The proposed framework integrates residue-level structural information to predict functional states (Active vs Inactive). By incorporating Graph Grad-CAM interpretability adapted for GNNs, enabling residue-level attribution and offering clear insights into the structural and functional motifs and single residues that are important for model's decision. To ensure robustness and realistic generalization, model performance is systematically evaluated using both random split and hard split strategies.
---
## Random Split : 
Graphs are randomly partitioned into training, validation, and test sets to assess overall predictive performance, with potential overlap of structurally similar GPCRs across splits.
## Hard Split : 
Graphs are separated based on receptor identity or structural similarity, ensuring related GPCRs are confined to a single split, thus providing a stringent evaluation of generalization to unseen receptors.

---

## Installation Instructions

The following setup was used to develop and run both ResiNet-GNN and SiteSpec-GNN models.

 pip install numpy pandas scikit-learn matplotlib seaborn tqdm
 
 pip install torch torchvision torchaudio
 
 pip install torch-geometric

---

## How to load Datasets and run Model codes

## 1️⃣ Datasets/
Three datasets of GPCRs were incorporated : Class A, Class A+B and Class A+B+C.

### Loading the Random Split Datasets
* Entering /ResiNet-GNN
* load `PRN_Dataset.pkl` as input file

### reproducing the result for ResiNet-GNN
* Entering /ResiNet-GNN
* run `Model.ipynb` to reproduce ResiNet-GNN result

### Loading the LRN Dataset
* Entering /SiteSpec-GNN
* load `LRN_Dataset.pkl` as input file

### reproducing the result for SiteSpec-GNN
* Entering /SiteSpec-GNN
* run `Model.ipynb` to reproduce SiteSpec-GNN result
  
---

## 2️⃣ Models/

This directory contains the pre-trained models and grad cam programs.

Subfolders:

Hard Split/ — Trained models based on the Hard Split datasets

Random Split/ — Trained models based on Random Split datasets

Grad-CAM_GPCRnet/ — Grad-CAM analysis applied to GPCRnet model, that provides excels that contain score for each node in the test data.
