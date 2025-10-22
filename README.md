🧬GPCRnet Framework 

1️⃣ Datasets/

This directory contains all dataset files used for model training and validation.
Each dataset is stored as serialized .pkl (pickle) objects for efficient loading and experimentation.

Subfolders:

Hard Split/ — Datasets divided ensuring zero overlap between training and testing sets.

Random Split/ — Datasets divided randomly for baseline model benchmarking.

2️⃣ Models/

This directory contains the pre-trained models and grad cam programs.

Subfolders:

Hard Split/ — Trained models based on the Hard Split datasets

Random Split/ — Trained models based on Random Split datasets

Grad-CAM_GPCRnet/ — Grad-CAM analysis applied to GPCRnet model, that provides excels that contain score for each node in the test data.
