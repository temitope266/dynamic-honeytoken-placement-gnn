# src/evaluate.py
import torch
import pickle
import os
import networkx as nx
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from src.model import GCNScorer
from config import DATA_DIR, PLACEMENT

def _topk_metrics(y_true, scores, k):
    top_k_idx = np.argsort(scores)[-k:]
    y_pred = np.zeros_like(y_true)
    y_pred[top_k_idx] = 1
    
    return {
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, scores)
    }

def run_evaluation():
    with open(os.path.join(DATA_DIR, 'graph.gpickle'), 'rb') as f:
        G = pickle.load(f)
        
    splits = torch.load(os.path.join(DATA_DIR, 'splits.pt'), weights_only=False)
    test_idx = splits['test_idx']
    
    features = torch.tensor([G.nodes[n]['feature_vector'] for n in G.nodes()], dtype=torch.float)
    edge_index = torch.tensor(list(G.edges())).t().contiguous()
    y_true_full = np.array([G.nodes[n]['label'] for n in G.nodes()])
    
    model = GCNScorer(num_node_features=features.shape[1])
    model.load_state_dict(torch.load(os.path.join(DATA_DIR, 'model.pt')))
    model.eval()
    
    with torch.no_grad():
        gnn_scores_full = model(features, edge_index).squeeze().numpy()
        
    # Baseline logic
    
    deg_dict = nx.degree_centrality(G)
    bet_dict = nx.betweenness_centrality(G)
    
    nodes_list = list(G.nodes())
    
    # Add a tiny random jitter (1e-6) to break score ties deterministically
    rng_jitter = np.random.default_rng(42)
    
    deg_scores_full = np.array([deg_dict[n] + rng_jitter.random() * 1e-6 for n in G.nodes()])
    bet_scores_full = np.array([bet_dict[n] + rng_jitter.random() * 1e-6 for n in G.nodes()])
    rand_scores_full = rng_jitter.random(len(G.nodes()))
    
    # Filter for test set
    y_true_test = y_true_full[test_idx]
    gnn_scores = gnn_scores_full[test_idx]
    deg_scores = deg_scores_full[test_idx]
    bet_scores = bet_scores_full[test_idx]
    rand_scores = rand_scores_full[test_idx]
    
    k = max(1, int(len(test_idx) * PLACEMENT['budget_fraction']))
    print(f"Evaluating on {len(test_idx)} held-out test nodes, budget k={k}\n")
    
    results = {
        "GNN": _topk_metrics(y_true_test, gnn_scores, k),
        "Random": _topk_metrics(y_true_test, rand_scores, k),
        "Degree": _topk_metrics(y_true_test, deg_scores, k),
        "Betweenness": _topk_metrics(y_true_test, bet_scores, k),
    }
    
    print(f"{'Method':<20}{'Precision':>10}{'Recall':>10}{'F1':>10}{'ROC-AUC':>10}")
    print("-" * 60)
    for name, m in results.items():
        print(f"{name:<20}{m['precision']:>10.3f}{m['recall']:>10.3f}{m['f1']:>10.3f}{m['roc_auc']:>10.3f}")
        
    torch.save(results, os.path.join(DATA_DIR, 'evaluation_results.pt'))

if __name__ == "__main__":
    run_evaluation()
