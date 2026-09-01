"""
src/train.py
------------
Trains the GCN classifier, tracks both training and validation loss,
and saves checkpoints and history.
"""

import os
import pickle
import numpy as np
import torch

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TRAIN, DATA_DIR, OUTPUT_DIR
from src.model import GCNScorer

def train_model():
    graph_path = os.path.join(DATA_DIR, 'graph.gpickle')
    with open(graph_path, 'rb') as f:
        G = pickle.load(f)
        
    features = torch.tensor([G.nodes[n]['feature_vector'] for n in G.nodes()], dtype=torch.float)
    labels = torch.tensor([G.nodes[n]['label'] for n in G.nodes()], dtype=torch.float).view(-1, 1)
    edge_index = torch.tensor(list(G.edges())).t().contiguous()
    
    # Train / Val / Test splits
    indices = np.random.permutation(len(G.nodes()))
    test_size = int(len(G.nodes()) * TRAIN['test_size'])
    val_size = int(len(G.nodes()) * TRAIN['val_size'])
    
    test_idx = indices[:test_size]
    val_idx = indices[test_size:test_size + val_size]
    train_idx = indices[test_size + val_size:]
    
    model = GCNScorer(num_node_features=features.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=TRAIN['lr'], weight_decay=TRAIN['weight_decay'])
    criterion = torch.nn.BCELoss(weight=torch.tensor([TRAIN['pos_weight']]))
    
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(1, TRAIN['epochs'] + 1):
        model.train()
        optimizer.zero_grad()
        out = model(features, edge_index)
        loss = criterion(out[train_idx], labels[train_idx])
        loss.backward()
        optimizer.step()
        
        # Compute validation loss
        model.eval()
        with torch.no_grad():
            val_loss = criterion(out[val_idx], labels[val_idx]).item()
            
        history['train_loss'].append(loss.item())
        history['val_loss'].append(val_loss)
        
        if epoch % 50 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f}")
            
    torch.save(model.state_dict(), os.path.join(DATA_DIR, 'model.pt'))
    torch.save({'train_idx': train_idx, 'val_idx': val_idx, 'test_idx': test_idx}, os.path.join(DATA_DIR, 'splits.pt'))
    torch.save(history, os.path.join(OUTPUT_DIR, 'history.pt'))
    
    print(f"Model trained and saved to {DATA_DIR}/model.pt")

if __name__ == "__main__":
    train_model()
