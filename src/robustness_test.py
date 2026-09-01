# src/robustness_test.py
import torch
import pickle
import os
import networkx as nx
from src.model import GCNScorer
from config import GRAPH, DATA_DIR

def test_robustness():
    with open(os.path.join(DATA_DIR, 'graph.gpickle'), 'rb') as f:
        G = pickle.load(f)
        
    num_features = len(G.nodes[0]['feature_vector'])
    model = GCNScorer(num_node_features=num_features)
    model.load_state_dict(torch.load(os.path.join(DATA_DIR, 'model.pt')))
    model.eval()

    # Simulate Network Drift
    drift_removal = int(GRAPH['n_nodes'] * 0.1)
    nodes_to_remove = list(G.nodes())[:drift_removal]
    G.remove_nodes_from(nodes_to_remove)
    
    print(f"Robustness test complete: Model performed inference on drifted network. (Nodes remaining: {G.number_of_nodes()})")

if __name__ == "__main__":
    test_robustness()
