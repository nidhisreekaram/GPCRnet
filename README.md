# 🧬GPCRnet: Interpretable Deep Learning Model to 
Classify Functional States and to Identify Functional Motifs of different GPCR Classes

This repository provides datasets and codes for GPCR functional state classification using Graph Neural Networks (GNNs). The proposed framework integrates residue-level structural information to predict functional states (Active vs Inactive). 
By incorporating Graph Grad-CAM interpretability adapted for GNNs, enabling residue-level attribution and offering clear insights into the structural and functional motifs and single residues that are important for model's decision. 
To ensure robustness and realistic generalization, model performance is systematically evaluated using both random split and hard split strategies.

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

### Random Split Datasets
* Entering /Datasets/Random Split
* Load `GPCR_A.pkl` as input file for Class A
* Load `GPCR_A+B.pkl` as input file for Class A+B
* Load `GPCR_A+B+C.pkl` as input file for Class A+B+C

### Hard Split Datasets
* Entering /Datasets/Hard Split/A
* Load `Train_A.pkl` and `Test_A.pkl` as input file for Class A

* Entering /Datasets/Hard Split/A+B
* Load `Train_A+B.pkl` and `Test_A+B.pkl` as input file for Class A+B

* Entering /Datasets/Hard Split/A+B+C
* Load `Train_A+B+C.pkl` and `Test_A+B+C.pkl` as input file for Class A+B+C
 
---

## 2️⃣ Models/

This directory contains the pre-trained models and grad cam programs.

### reproducing the result for Random Split
* Entering /Models/Random Split/
* run `A.ipynb` to reproduce Class A result
* run `A+B.ipynb` to reproduce Class A+B result
* run `A+B+C.ipynb` to reproduce Class A+B+C result

### reproducing the result for Hard Split
* Entering /Models/Hard Split/
* run `A.ipynb` to reproduce Class A result
* run `A+B.ipynb` to reproduce Class A+B result
* run `A+B+C.ipynb` to reproduce Class A+B+C result

Grad-CAM-GPCRnet/ — Grad-CAM analysis applied to GPCRnet model, that provides excels that contain score for each node in the test data.

The Class A+B model trained using a random split demonstrated the highest overall performance and was therefore designated as GPCRnet.

### reproducing the result for Grad-CAM results
* Entering /Models/Grad-CAM- GPCRnet
* Load `GPCR_A+B.pkl` as input file for Class A+B
* run `Graph Grad Cam results.ipynb` to reproduce Grad-CAM results
