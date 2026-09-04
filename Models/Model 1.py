#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GINConv, global_mean_pool
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_curve, auc, precision_score, recall_score, f1_score)
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import numpy as np
import random
import os


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
FILE_NAME = "Class A.pkl"

pickle_file_path = os.path.join(DATA_DIR, FILE_NAME)
pytorch_dataset = load_pytorch_dataset(pickle_file_path)


# In[3]:


class SimpleGINModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate):
        super(SimpleGINModel, self).__init__()
        self.gin1 = GINConv(nn.Linear(input_dim, hidden_dim))
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.gin2 = GINConv(nn.Linear(hidden_dim, hidden_dim))
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.gin3 = GINConv(nn.Linear(hidden_dim, hidden_dim))
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.gin4 = GINConv(nn.Linear(hidden_dim, hidden_dim))
        self.bn4 = nn.BatchNorm1d(hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.bn1(self.gin1(x, edge_index)))
        x = F.relu(self.bn2(self.gin2(x, edge_index)))
        x = F.relu(self.bn3(self.gin3(x, edge_index)))
        x = F.relu(self.bn4(self.gin4(x, edge_index)))
        x = self.dropout(x)
        x = global_mean_pool(x, batch)
        return self.fc(x)


# In[4]:


# === Training ===
def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct = 0, 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        data.y = data.y.long()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
        correct += (out.argmax(dim=1) == data.y).sum().item()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)

# === Evaluation ===
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct = 0, 0
    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            data.y = data.y.long()
            out = model(data)
            loss = criterion(out, data.y)
            total_loss += loss.item() * data.num_graphs
            correct += (out.argmax(dim=1) == data.y).sum().item()
            probs = F.softmax(out, dim=1)[:, 1].cpu().numpy()
            all_labels.extend(data.y.cpu().numpy())
            all_preds.extend(out.argmax(dim=1).cpu().numpy())
            all_probs.extend(probs)
    return total_loss / len(loader.dataset), correct / len(loader.dataset), \
           np.array(all_labels), np.array(all_preds), np.array(all_probs)


# In[5]:


# === Prepare Data ===
train_data, test_data = train_test_split(pytorch_dataset, test_size=0.2, random_state=seed)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# === Initialize Model ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleGINModel(20, 64, 2, 0.5).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()

# === Training Loop ===
num_epochs = 350
train_losses, train_accs, test_losses, test_accs = [], [], [], []
history = []

for epoch in range(1, num_epochs + 1):
    tr_loss, tr_acc = train(model, train_loader, optimizer, criterion, device)
    te_loss, te_acc, y_true, y_pred, y_probs = evaluate(model, test_loader, criterion, device)

    train_losses.append(tr_loss)
    train_accs.append(tr_acc)
    test_losses.append(te_loss)
    test_accs.append(te_acc)
    history.append((y_true, y_pred, y_probs))

    print(f"Epoch {epoch:03d}: Train Loss={tr_loss:.4f}, Train Acc={tr_acc:.4f}, Test Loss={te_loss:.4f}, Test Acc={te_acc:.4f}")


# In[ ]:




