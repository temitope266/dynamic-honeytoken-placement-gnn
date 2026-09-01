# src/labeller.py
import networkx as nx
import numpy as np
import pickle
import os
from config import LABELLING, DATA_DIR

def simulate_lateral_movement(G):
    print("Simulating attacker lateral movement via Biased Random Walks...")
    traversal_counts = {node: 0 for node in G.nodes()}
    
    entry_nodes = np.random.choice(G.nodes(), size=LABELLING['n_entry_nodes'], replace=False)
    
    for start_node in entry_nodes:
        for _ in range(LABELLING['n_walks_per_entry']):
            current_node = start_node
            for _ in range(LABELLING['walk_length']):
                neighbors = list(G.neighbors(current_node))
                if not neighbors:
                    break
                    
                weights = [G.edges[current_node, n]['weight'] for n in neighbors]
                total_weight = sum(weights)
                probs = [w / total_weight for w in weights]
                
                next_node = np.random.choice(neighbors, p=probs)
                traversal_counts[next_node] += 1
                current_node = next_node

    counts = list(traversal_counts.values())
    threshold = np.percentile(counts, LABELLING['top_percentile'] * 100)
    
    positive_count = 0
    for node in G.nodes():
        label = 1 if traversal_counts[node] >= threshold else 0
        G.nodes[node]['label'] = label
        positive_count += label
        
    print(f"Labelled {positive_count} / {G.number_of_nodes()} nodes as honeytoken candidates.")
    
    with open(os.path.join(DATA_DIR, 'graph.gpickle'), 'wb') as f:
        pickle.dump(G, f)

if __name__ == "__main__":
    with open(os.path.join(DATA_DIR, 'graph.gpickle'), 'rb') as f:
        G = pickle.load(f)
    simulate_lateral_movement(G)
