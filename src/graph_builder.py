# src/graph_builder.py
import networkx as nx
import numpy as np
import pickle
import os
import random
from config import GRAPH, DATA_DIR

def build_enterprise_graph():
    print(f"Building Enterprise Graph with {GRAPH['n_vlans']} VLANs...")
    
    nodes_per_vlan = GRAPH['n_nodes'] // GRAPH['n_vlans']
    G = nx.connected_caveman_graph(GRAPH['n_vlans'], nodes_per_vlan)
    
    # Add random cross-VLAN connections
    for _ in range(GRAPH['n_nodes'] // 10):
        u, v = random.sample(list(G.nodes()), 2)
        G.add_edge(u, v)

    # Assign Node Types
    types = np.random.choice(
        GRAPH['node_types'], 
        size=GRAPH['n_nodes'], 
        p=GRAPH['node_type_ratios']
    )
    
    clustering = nx.clustering(G)
    degrees = dict(G.degree())
    max_degree = max(degrees.values())
    
    for node in G.nodes():
        node_type = types[node]
        G.nodes[node]['type'] = node_type
        
        # FEATURE ENGINEERING (No Betweenness Data Leakage!)
        type_encoding = [1.0 if node_type == t else 0.0 for t in GRAPH['node_types']]
        norm_degree = degrees[node] / max_degree
        clust_coeff = clustering[node]
        
        G.nodes[node]['feature_vector'] = type_encoding + [norm_degree, clust_coeff]
        G.nodes[node]['is_high_value'] = 1 if node_type in ['server', 'database'] else 0

    # Assign Edge Attributes 
    for u, v in G.edges():
        protocol = random.choice(GRAPH['protocols'])
        G.edges[u, v]['protocol'] = protocol
        G.edges[u, v]['weight'] = GRAPH['protocol_weights'][protocol]

    with open(os.path.join(DATA_DIR, 'graph.gpickle'), 'wb') as f:
        pickle.dump(G, f)
        
    print(f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"Features per node: {len(G.nodes[0]['feature_vector'])}")

if __name__ == "__main__":
    build_enterprise_graph()
