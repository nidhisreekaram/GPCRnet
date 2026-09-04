#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pickle
import random
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_mean_pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_curve, auc, precision_score, recall_score, f1_score, confusion_matrix)


# In[2]:


seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

def load_pytorch_dataset(file_path):
    import pickle
    with open(file_path, 'rb') as f:
        return pickle.load(f)

DATA_DIR = "Datasets"
FILE_NAME = "Class A+B.pkl"

pickle_file_path = os.path.join(DATA_DIR, FILE_NAME)
pytorch_dataset = load_pytorch_dataset(pickle_file_path)

RESULTS_DIR = "Validation Results"
os.makedirs(RESULTS_DIR, exist_ok=True)


# In[3]:


pytorch_dataset = load_pytorch_dataset(pickle_file_path)
labels = []
for data in pytorch_dataset:
    if isinstance(data.y, torch.Tensor):
        labels.append(int(data.y.item()))
    else:
        labels.append(int(data.y))
labels = np.array(labels)

class SimpleGINModel(nn.Module):
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

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        batch = data.batch

        x = F.relu(self.bn1(self.gin1(x, edge_index)))
        x = F.relu(self.bn2(self.gin2(x, edge_index)))
        x = F.relu(self.bn3(self.gin3(x, edge_index)))
        x = F.relu(self.bn4(self.gin4(x, edge_index)))
        x = self.dropout(x)
        x = global_mean_pool(x, batch)
        return self.fc(x)


# In[4]:


def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        data.y = data.y.long()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += (loss.item() * data.num_graphs)
        correct += (out.argmax(1) == data.y).sum().item()

    return (total_loss / len(loader.dataset), correct / len(loader.dataset))

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    labels = []
    preds = []
    probs = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            data.y = data.y.long()
            out = model(data)
            loss = criterion(out, data.y)
            total_loss += (loss.item() * data.num_graphs)
            pred = out.argmax(1)
            correct += (pred == data.y).sum().item()
            labels.extend(data.y.cpu().numpy())
            preds.extend(pred.cpu().numpy())
            probs.extend(F.softmax(out, dim=1)[:, 1].cpu().numpy())

    return (total_loss / len(loader.dataset), correct / len(loader.dataset),
            np.array(labels), np.array(preds), np.array(probs))


# In[5]:


def calculate_metrics(y_true, y_pred, y_prob):
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = (tn / (tn + fp) if (tn + fp) > 0 else 0)
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    return (accuracy, precision, recall, f1, roc_auc, specificity, tn, fp, fn, tp)

def save_pdb_ids(dataset, excel_path):
    pdb_ids = []
    for data in dataset:
        if hasattr(data, "pdb_id"): pdb_ids.append(data.pdb_id)
        else: pdb_ids.append("UNKNOWN")
    pd.DataFrame({"PDB_ID": pdb_ids}).to_excel(excel_path, index=False)

def save_dataset(dataset, file_path):
    with open(file_path, "wb") as f:
        pickle.dump(dataset, f)


# In[6]:


device = "cuda" if torch.cuda.is_available() else "cpu"

num_epochs = 350
num_folds = 5

skf = StratifiedKFold(
    n_splits=num_folds,
    shuffle=True,
    random_state=seed
)

fold_results = []

for fold, (train_idx, test_idx) in enumerate(
        skf.split(np.zeros(len(labels)), labels),
        start=1):

    print("\n")
    print("=" * 80)
    print(f"FOLD {fold}")
    print("=" * 80)

    fold_folder = os.path.join(RESULTS_DIR, f"Fold_{fold}")
    os.makedirs(fold_folder, exist_ok=True)

    train_data = [pytorch_dataset[i] for i in train_idx]
    test_data = [pytorch_dataset[i] for i in test_idx]

    save_dataset(train_data, os.path.join(fold_folder, "train_dataset.pkl"))
    save_dataset(test_data, os.path.join(fold_folder, "test_dataset.pkl"))

    save_pdb_ids(train_data, os.path.join(fold_folder, "Train_PDB_IDs.xlsx"))
    save_pdb_ids(test_data, os.path.join(fold_folder, "Test_PDB_IDs.xlsx"))

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

    model = SimpleGINModel(
        input_dim=20,
        hidden_dim=64,
        output_dim=2,
        dropout_rate=0.5
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()
    epoch_history = []

    for epoch in range(1, num_epochs + 1):

        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc, y_true, y_pred, y_prob = evaluate(model, test_loader, criterion, device)

        print(
            f"Epoch {epoch:3d}/{num_epochs} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Train Acc : {train_acc:.4f} | "
            f"Test Loss : {test_loss:.6f} | "
            f"Test Acc  : {test_acc:.4f}"
        )

        epoch_history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "y_true": y_true,
            "y_pred": y_pred,
            "y_prob": y_prob,
            "model_state": copy.deepcopy(model.state_dict())
        })

    best_epoch_data = sorted(
        epoch_history,
        key=lambda x: (x["test_loss"], -x["test_acc"])
    )[0]

    best_epoch = best_epoch_data["epoch"]
    best_acc = best_epoch_data["test_acc"]
    best_loss = best_epoch_data["test_loss"]

    y_true = best_epoch_data["y_true"]
    y_pred = best_epoch_data["y_pred"]
    y_prob = best_epoch_data["y_prob"]

    # Load and save best weights for this fold
    model.load_state_dict(best_epoch_data["model_state"])
    best_model_path = os.path.join(fold_folder, "best_model.pth")
    torch.save(model.state_dict(), best_model_path)

    print("\n")
    print("=" * 80)
    print("BEST MODEL SELECTED AFTER ALL EPOCHS")
    print("=" * 80)
    print(f"Best Epoch    : {best_epoch}")
    print(f"Best Accuracy : {best_acc:.4f}")
    print(f"Best Loss     : {best_loss:.6f}")
    print("=" * 80)

    # --- FIX: metrics + per-fold Excel files now computed & saved INSIDE the loop ---
    acc, precision, recall, f1, roc_auc, specificity, tn, fp, fn, tp = calculate_metrics(
        y_true, y_pred, y_prob
    )

    fold_metrics = pd.DataFrame({
        "Fold": [fold],
        "Best Epoch": [best_epoch],
        "Lowest Test Loss": [best_loss],
        "Highest Test Accuracy": [best_acc],
        "Accuracy": [acc],
        "Precision": [precision],
        "Recall": [recall],
        "F1 Score": [f1],
        "AUC-ROC": [roc_auc],
        "Specificity": [specificity]
    })
    fold_metrics.to_excel(os.path.join(fold_folder, "Fold_Metrics.xlsx"), index=False)

    confusion_df = pd.DataFrame({
        "Metric": ["True Positive", "True Negative", "False Positive", "False Negative"],
        "Value": [tp, tn, fp, fn]
    })
    confusion_df.to_excel(os.path.join(fold_folder, "Confusion_Matrix.xlsx"), index=False)
    fold_results.append({
        "Fold": fold,
        "Best Epoch": best_epoch,
        "Lowest Test Loss": best_loss,
        "Highest Test Accuracy": best_acc,
        "Accuracy": acc,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "AUC-ROC": roc_auc,
        "Specificity": specificity,
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn
    })

    print("\n")
    print("=" * 80)
    print(f"Fold {fold} Completed")
    print("=" * 80)
    print(f"Best Epoch           : {best_epoch}")
    print(f"Lowest Test Loss     : {best_loss:.6f}")
    print(f"Highest Test Accuracy: {best_acc:.6f}")
    print(f"Accuracy             : {acc:.6f}")
    print(f"Precision            : {precision:.6f}")
    print(f"Recall               : {recall:.6f}")
    print(f"F1 Score             : {f1:.6f}")
    print(f"AUC-ROC              : {roc_auc:.6f}")
    print(f"Specificity          : {specificity:.6f}")


# In[7]:


results = pd.DataFrame(fold_results)
results.to_excel(os.path.join(RESULTS_DIR, "Final_Results.xlsx"), index=False)

print("\n")
print("=" * 100)
print("FINAL 5-FOLD RESULTS")
print("=" * 100)
print(results.to_string(index=False))
print("\n")
print("=" * 100)
print("MEAN \u00b1 STD")
print("=" * 100)

summary = []

for column in [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score",
    "AUC-ROC",
    "Specificity"
]:
    mean = results[column].mean()
    std = results[column].std()
    summary.append({
        "Metric": column,
        "Mean": mean,
        "Std": std,
        "Mean \u00b1 Std": f"{mean:.4f} \u00b1 {std:.4f}"
    })
    print(f"{column:<15} : {mean:.4f} \u00b1 {std:.4f}")

summary_df = pd.DataFrame(summary)
summary_df.to_excel(os.path.join(RESULTS_DIR, "Summary_Mean_STD.xlsx"), index=False)

print("\n")
print("=" * 100)
print("ALL FILES SAVED SUCCESSFULLY")
print("=" * 100)
print("\nSaved Files:\n")
for f in range(1, 6):
    print(f"Fold_{f}/")
    print("    best_model.pth")
    print("    train_dataset.pkl")
    print("    test_dataset.pkl")
    print("    Train_PDB_IDs.xlsx")
    print("    Test_PDB_IDs.xlsx")
    print("    Fold_Metrics.xlsx")
    print("    Confusion_Matrix.xlsx")
    print("")
print("Final_Results.xlsx")
print("Summary_Mean_STD.xlsx")
print("\nDone.")


# In[ ]:




