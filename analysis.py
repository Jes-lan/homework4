import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from networkx.algorithms import community

def main():
    print("Starting Network Analysis...")

    # 1. Load Nodes
    try:
        nodes_df = pd.read_csv('InputFileNodes.csv')
        node_col = 'id' if 'id' in nodes_df.columns else nodes_df.columns[0]
        nodes = nodes_df[node_col].tolist()
        print(f"Loaded {len(nodes)} nodes.")
    except Exception as e:
        print(f"Error loading nodes: {e}")
        return

    # 2. Load Edges
    try:
        edges_df = pd.read_csv('InputFileEdges.csv')
        edges_df.columns = [c.lower() for c in edges_df.columns]
        if 'from' not in edges_df.columns or 'to' not in edges_df.columns:
             edges_df.rename(columns={edges_df.columns[0]: 'from', edges_df.columns[1]: 'to'}, inplace=True)
        
        raw_edges = list(zip(edges_df['from'], edges_df['to']))
        print(f"Loaded {len(raw_edges)} raw edges.")
    except Exception as e:
        print(f"Error loading edges: {e}")
        return

    # 3. Merge duplicate edges
    unique_edges = list(set(raw_edges))
    print(f"Unique edges after merging: {len(unique_edges)}")

    # 4. Undirected Network G
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(unique_edges)
    
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=500, font_size=10)
    plt.title("Undirected Network G")
    plt.savefig("network_G.png")
    plt.show() # Interactive
    print("Generated network_G.png")

    # 5. Directed Network G_directed
    G_directed = nx.DiGraph()
    G_directed.add_nodes_from(nodes)
    G_directed.add_edges_from(unique_edges)

    plt.figure(figsize=(10, 8))
    nx.draw(G_directed, pos, with_labels=True, node_color='lightgreen', node_size=500, font_size=10, arrows=True)
    plt.title("Directed Network G_directed")
    plt.savefig("network_G_directed.png")
    plt.show() # Interactive
    print("Generated network_G_directed.png")

    # --- Metrics for G (Undirected) ---

    # 6. List degrees
    degrees = dict(G.degree())
    print("\n--- 6. Node Degrees ---")
    # for node, deg in degrees.items(): print(f"Node {node}: {deg}") # Reduced output for cleanliness

    # 7. Average degree
    avg_degree = sum(degrees.values()) / len(G)
    print(f"\n--- 7. Average Degree: {avg_degree:.4f} ---")

    # 8. Histogram
    degree_values = list(degrees.values())
    plt.figure()
    plt.hist(degree_values, bins=range(min(degree_values), max(degree_values) + 2), align='left', rwidth=0.8)
    plt.title("Degree Histogram")
    plt.xlabel("Degree")
    plt.ylabel("Frequency")
    plt.savefig("histogram.png")
    plt.show() # Interactive
    print("Generated histogram.png")

    # 9. Degree Centrality
    deg_centrality = nx.degree_centrality(G)
    
    # 10. Closeness Centrality
    closeness_centrality = nx.closeness_centrality(G)
    
    # 11. Betweenness Centrality
    betweenness_centrality = nx.betweenness_centrality(G)
    
    # 12. Eigenvector Centrality
    try:
        eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
    except:
        eigenvector_centrality = {}

    # 13. Katz Centrality
    try:
        katz_centrality = nx.katz_centrality(G, max_iter=1000)
    except:
        katz_centrality = {}

    # 14. PageRank
    pagerank = nx.pagerank(G)

    # 15. Modularity
    communities = community.greedy_modularity_communities(G)
    modularity_score = community.modularity(G, communities)
    print(f"\n--- 15. Modularity: {modularity_score:.4f} ---")

    # 16. Density
    density = nx.density(G)
    print(f"\n--- 16. Density: {density:.4f} ---")

    # 17. Average Clustering
    avg_clustering = nx.average_clustering(G)
    print(f"\n--- 17. Average Clustering Coefficient: {avg_clustering:.4f} ---")

    # 18. Diameter
    if nx.is_connected(G):
        diameter = nx.diameter(G)
        print(f"\n--- 18. Diameter: {diameter} ---")
    else:
        comps = [G.subgraph(c).copy() for c in nx.connected_components(G)]
        diameter = nx.diameter(max(comps, key=len))
        print(f"\n--- 18. Diameter (largest component): {diameter} ---")

    # 19. Degree Distribution Plot
    deg_counts = pd.Series(degree_values).value_counts().sort_index()
    deg_probs = deg_counts / len(nodes)
    plt.figure()
    plt.plot(deg_probs.index, deg_probs.values, 'bo-')
    plt.title("Degree Distribution (Undirected)")
    plt.xlabel("Degree (k)")
    plt.ylabel("P(k)")
    plt.savefig("degree_distribution.png")
    plt.show() # Interactive
    print("Generated degree_distribution.png")

    # 20. Interpretation (Summary)
    print("\n--- 20. Interpretation Summary ---")
    print("Metrics calculated. See full report for details.")

    # --- NEW STEPS ---

    # 21. Top 5 Bridge Nodes (Betweenness)
    print("\n--- 21. Top 5 'Bridge' Nodes (Betweenness) ---")
    top_bridges = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
    for n, c in top_bridges:
        print(f"Node {n}: {c:.4f}")

    # 22. Top 5 Leader Nodes (Degree)
    print("\n--- 22. Top 5 'Leader' Nodes (Degree) ---")
    top_leaders = sorted(deg_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
    for n, c in top_leaders:
        print(f"Node {n}: {c:.4f}")

    # 23. Directed Degree Distribution
    print("\n--- 23. Directed Degree Distribution ---")
    in_degrees = [d for n, d in G_directed.in_degree()]
    out_degrees = [d for n, d in G_directed.out_degree()]
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    in_counts = pd.Series(in_degrees).value_counts().sort_index()
    in_probs = in_counts / len(nodes)
    plt.plot(in_probs.index, in_probs.values, 'ro-', label='In-Degree')
    plt.title("In-Degree Distribution")
    plt.xlabel("In-Degree (k_in)")
    plt.ylabel("P(k_in)")
    plt.legend()

    plt.subplot(1, 2, 2)
    out_counts = pd.Series(out_degrees).value_counts().sort_index()
    out_probs = out_counts / len(nodes)
    plt.plot(out_probs.index, out_probs.values, 'go-', label='Out-Degree')
    plt.title("Out-Degree Distribution")
    plt.xlabel("Out-Degree (k_out)")
    plt.ylabel("P(k_out)")
    plt.legend()

    plt.tight_layout()
    plt.savefig("directed_degree_distribution.png")
    plt.show() # Interactive
    print("Generated directed_degree_distribution.png")

    # 24. Robustness Analysis (Remove Top 3 Strongest Nodes)
    print("\n--- 24. Robustness Analysis ---")
    # "Strongest" usually means Degree Centrality in this context (or could be betweenness, but degree is standard for 'attacks')
    # We will use the top 3 from Step 22.
    nodes_to_remove = [n for n, c in top_leaders[:3]]
    print(f"Removing top 3 nodes: {nodes_to_remove}")
    
    G_robust = G.copy()
    G_robust.remove_nodes_from(nodes_to_remove)
    
    plt.figure(figsize=(10, 8))
    # Recalculate layout for new graph
    pos_robust = nx.spring_layout(G_robust, seed=42)
    nx.draw(G_robust, pos_robust, with_labels=True, node_color='salmon', node_size=500, font_size=10)
    plt.title(f"Network after removing {nodes_to_remove}")
    plt.savefig("network_robustness.png")
    plt.show() # Interactive
    print("Generated network_robustness.png")
    
    # Interpret changes
    if nx.is_connected(G_robust):
        print("Network remains connected.")
    else:
        num_components = nx.number_connected_components(G_robust)
        print(f"Network is now disconnected into {num_components} components.")
    
    old_density = density
    new_density = nx.density(G_robust)
    print(f"Density change: {old_density:.4f} -> {new_density:.4f}")

if __name__ == "__main__":
    main()
