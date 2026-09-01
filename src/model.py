# src/model.py
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from config import MODEL

class GCNScorer(torch.nn.Module):
    def __init__(self, num_node_features):
        super(GCNScorer, self).__init__()
        self.conv1 = GCNConv(num_node_features, MODEL['hidden_dim'])
        self.conv2 = GCNConv(MODEL['hidden_dim'], 16)
        self.classifier = torch.nn.Linear(16, 1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=MODEL['dropout'], training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        return torch.sigmoid(self.classifier(x))
