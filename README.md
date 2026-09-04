# 🧬GPCRnet: An Interpretable Structure-Based Deep Learning Framework for Functional State Prediction and Motif Discovery Across GPCR Classes

This repository provides datasets and codes for GPCR functional state classification using Graph Isomorphism Networks (GINs). The residue-level structural information was integrated to predict functional states (Active vs Inactive). By incorporating Graph Grad-CAM interpretability adapted for GNNs, enabling residue-level attribution and offering clear insights into the structural and functional motifs/residues that are important for model's decision.

---

## Installation Instructions

The following setup was used to develop and run models.

 pip install numpy pandas scikit-learn matplotlib seaborn tqdm
 
 pip install torch torchvision torchaudio
 
 pip install torch-geometric

---

## How to load Datasets and run Model codes

## Datasets/
This folder contains Three datasets of GPCRs were incorporated : Class A, Class A+B and Class A+B+C.

* Enter/Datasets/
* The PDB-IDs list was provided in the excel `PDB_List.xlsx`
* Load `Class A.pkl` as input file for Class A
* Load `Class A+B.pkl` as input file for Class A+B
* Load `Class A+B+C.pkl` as input file for Class A+B+C
 
---

## Models/

This folder contains the models for classification and Graph Grad-CAM for identification of functional motifs.

* Enter/Models/
* run `Model 1.py` to get Class A result
* run `Model 2.py` to get Class A+B result
* run `Model 3.py` to get Class A+B+C result

### Stratified five-Fold Cross Validation (CV)

CV  was performed on all three datasets for further evaluation of model performance. A reference code was provided.

* Enter/Datasets/
* Load `Class A.pkl`, `Class A+B.pkl` and `Class A+B+C.pkl` as input for CV
* Enter/Models/
* run `Cross Validation.py` on three different datasets to get the CV result

Model 2 with Class A+B dataset was demonstrated the highest overall performance and designated as GPCRnet.

### Grad-CAM Analysis

Graph Grad-CAM was applied to GPCRnet model, that provides excels that contain score for each node in the test data.

* Enter/Datasets/
* Load `Class A+B+C.pkl` as input file for Class A+B
* Enter/Models/
* run `Graph GradCam Results.py` for Graph Grad-CAM results
