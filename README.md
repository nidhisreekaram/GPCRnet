# 🧬GPCRnet: Interpretable Deep Learning Model to Classify Functional States and to Identify Functional Motifs of different GPCR Classes

This repository provides datasets and codes for GPCR functional state classification using Graph Neural Networks (GNNs). The proposed framework integrates residue-level structural information to predict functional states (Active vs Inactive). By incorporating Graph Grad-CAM interpretability adapted for GNNs, enabling residue-level attribution and offering clear insights into the structural and functional motifs and single residues that are important for model's decision. To ensure robustness and realistic generalization, model performance is systematically evaluated using both random split and hard split strategies.
## 1️⃣ Datasets/

This directory contains all dataset files used for model training and validation.
Each dataset is stored as serialized .pkl (pickle) objects for efficient loading and experimentation.

Subfolders:

Hard Split/ — Datasets divided ensuring zero overlap between training and testing sets.

Random Split/ — Datasets divided randomly for baseline model benchmarking.

## 2️⃣ Models/

This directory contains the pre-trained models and grad cam programs.

Subfolders:

Hard Split/ — Trained models based on the Hard Split datasets

Random Split/ — Trained models based on Random Split datasets

Grad-CAM_GPCRnet/ — Grad-CAM analysis applied to GPCRnet model, that provides excels that contain score for each node in the test data.
