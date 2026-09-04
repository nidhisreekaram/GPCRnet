#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import pickle
import pandas as pd
import random
import os


# In[ ]:


seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed_all(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_path = "Model.pth"
train_pickle_path = "Train.pkl"
test_pickle_path = "Test.pkl"

# === Output directory ===
quad_folder = r"GradCAM"
os.makedirs(quad_folder, exist_ok=True)

def load_pytorch_dataset(file_path):
    with open(file_path, 'rb') as f:
        dataset = pickle.load(f)
    print(f"Dataset successfully loaded from {file_path} ({len(dataset)} graphs)")
    return dataset


# In[ ]:


def get_pdb_id(data, fallback_idx):
    if hasattr(data, 'pdb_id') and data.pdb_id is not None:
        return data.pdb_id
    return fallback_idx

train_dataset = load_pytorch_dataset(train_pickle_path)
test_dataset = load_pytorch_dataset(test_pickle_path)

train_dataset = load_pytorch_dataset(train_pickle_path)
test_dataset = load_pytorch_dataset(test_pickle_path)

for i, data in enumerate(train_dataset, start=1):
    data.graph_id = get_pdb_id(data, i)
for i, data in enumerate(test_dataset, start=1):
    data.graph_id = get_pdb_id(data, i)

train_pdb_ids = [data.graph_id for data in train_dataset]
test_pdb_ids = [data.graph_id for data in test_dataset]

train_indices_path = os.path.join(quad_folder, "train_pdb_ids.txt")
test_indices_path = os.path.join(quad_folder, "test_pdb_ids.txt")

with open(train_indices_path, 'w') as f:
    for pid in train_pdb_ids:
        f.write(f"{pid}\n")

with open(test_indices_path, 'w') as f:
    for pid in test_pdb_ids:
        f.write(f"{pid}\n")


# In[ ]:


class SimpleGINModelWithGraphGradCAM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate):
        super().__init__()

        self.gin1 = GINConv(nn.Linear(input_dim, hidden_dim))
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        self.gin2 = GINConv(nn.Linear(hidden_dim, hidden_dim))
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        self.gin3 = GINConv(nn.Linear(hidden_dim, hidden_dim))
        self.bn3 = nn.BatchNorm1d(hidden_dim)

        self.gin4 = GINConv(nn.Linear(hidden_dim, hidden_dim))
        self.bn4 = nn.BatchNorm1d(hidden_dim)

        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim, output_dim)

        self.activations = None
        self.gradients = None

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.bn1(self.gin1(x, edge_index)))
        x = F.relu(self.bn2(self.gin2(x, edge_index)))
        x = F.relu(self.bn3(self.gin3(x, edge_index)))
        x = F.relu(self.bn4(self.gin4(x, edge_index)))

        self.activations = x
        if x.requires_grad:
            x.register_hook(self._save_gradients)

        x = self.dropout(x)
        x = global_mean_pool(x, batch)
        out = self.fc(x)
        return out

    def _save_gradients(self, grad):
        self.gradients = grad

def graph_grad_cam_with_edges_all(model, data, target_class=None):
    model.eval()
    batched = Batch.from_data_list([data]).to(device)

    model.zero_grad()
    model.activations = None
    model.gradients = None

    output = model(batched)

    predicted_class = output.argmax(dim=1).item()
    if target_class is None:
        target_class = predicted_class

    target = output[:, target_class]
    target.squeeze().backward() 

    activations = model.activations.detach()
    gradients = model.gradients.detach()

    weights = gradients.mean(dim=0, keepdim=True)
    graph_grad_cam_values = F.relu((activations * weights).sum(dim=1))

    node_positions = torch.arange(graph_grad_cam_values.size(0)).cpu().numpy()
    residue_types = batched.x.cpu().numpy()
    node_scores = graph_grad_cam_values.cpu().numpy()
    edge_index = batched.edge_index.cpu().numpy()
    edge_scores = [
        (graph_grad_cam_values[src] + graph_grad_cam_values[dst]) / 2
        for src, dst in edge_index.T
    ]
    edge_scores = torch.stack(edge_scores).cpu().numpy()

    return node_positions, residue_types, node_scores, edge_index.T, edge_scores, predicted_class


# In[ ]:


def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct = 0, 0
    all_preds, all_labels, all_pdb_ids = [], [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            data.y = data.y.long()
            output = model(data)
            loss = criterion(output, data.y)
            preds = output.argmax(dim=1)
            total_loss += loss.item() * data.num_graphs
            correct += (preds == data.y).sum().item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())
            all_pdb_ids.extend([getattr(d, 'graph_id', 'unknown') for d in data.to_data_list()])
    return total_loss / len(loader.dataset), correct / len(loader.dataset), all_preds, all_labels, all_pdb_ids

test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# In[ ]:


model = SimpleGINModelWithGraphGradCAM(20, 64, 2, 0.5).to(device)
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
criterion = nn.CrossEntropyLoss()

test_loss, test_acc, preds, labels, pdb_ids = evaluate(model, test_loader, criterion)
print(f"\n📌 Loaded model test accuracy: {test_acc:.4f} (loss: {test_loss:.4f})")

cm = confusion_matrix(labels, preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
plt.figure(figsize=(5, 5))
disp.plot()
plt.title("Confusion Matrix (Loaded Model)")
plt.savefig(confusion_path)
plt.close()

tp, fp, tn, fn = [], [], [], []
for pred, label, pid in zip(preds, labels, pdb_ids):
    if pred == 1 and label == 1:
        tp.append(pid)
    elif pred == 1 and label == 0:
        fp.append(pid)
    elif pred == 0 and label == 0:
        tn.append(pid)
    elif pred == 0 and label == 1:
        fn.append(pid)

def save_indices(name, indices):
    with open(path, 'w') as f:
        for pid in indices:
            f.write(f"{pid}\n")

save_indices("True_Positive", tp)
save_indices("False_Positive", fp)
save_indices("True_Negative", tn)
save_indices("False_Negative", fn)

def categorize(pred, label):
    if pred == 1 and label == 1:
        return "True_Positive"
    if pred == 1 and label == 0:
        return "False_Positive"
    if pred == 0 and label == 0:
        return "True_Negative"
    return "False_Negative"

report_df = pd.DataFrame({
    "PDB_ID": pdb_ids,
    "True_Label": labels,
    "Predicted_Label": preds,
    "Prediction_Correct": [pred == label for pred, label in zip(preds, labels)],
    "Category": [categorize(pred, label) for pred, label in zip(preds, labels)],
})

report_path = os.path.join(quad_folder, "Prediction_Report.xlsx")
with pd.ExcelWriter(report_path, engine='xlsxwriter') as writer:
    report_df.to_excel(writer, sheet_name="Predictions", index=False)
    summary_df = report_df["Category"].value_counts().rename_axis("Category").reset_index(name="Count")
    summary_df.to_excel(writer, sheet_name="Summary", index=False)


# In[ ]:


for data in test_dataset:
    pdb_id = data.graph_id

    node_positions, residue_types, node_scores, edge_indices, edge_scores, predicted_class = (
        graph_grad_cam_with_edges_all(model, data, target_class=None)
    )

    node_df = pd.DataFrame({
        "Node_Index": node_positions,
        "Residue_Type": [res.tolist() for res in residue_types],
        "Graph_Grad_CAM_Score": node_scores
    })

    edge_df = pd.DataFrame({
        "Edge_Start": edge_indices[:, 0],
        "Edge_End": edge_indices[:, 1],
        "Graph_Grad_CAM_Edge_Score": edge_scores
    })

    output_file = os.path.join(quad_folder, f"Graph_{pdb_id}_Grad_CAM.xlsx")
    with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
        node_df.to_excel(writer, sheet_name="Nodes", index=False)
        edge_df.to_excel(writer, sheet_name="Edges", index=False)

