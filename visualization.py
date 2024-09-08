import matplotlib.pyplot as plt
import networkx as nx
import community as community_louvain
import matplotlib.patches as mpatches

def visualize_graph_with_clustering_and_weights(G, title="Grafo Completo", log_message=None, community_weights=None):
    """
    Visualiza o grafo completo com clusters de comunidades e exibe as comunidades com as conexões mais fortalecidas e enfraquecidas.
    Mostra também o peso total de cada comunidade, excluindo as que têm peso zero.
    """
    # Detecta as comunidades com o algoritmo de Louvain
    partition = community_louvain.best_partition(G)
    unique_communities = sorted(set(partition.values()))  # Comunidades detectadas

    # Layout do grafo
    pos = nx.spring_layout(G, seed=42)

    # Usar um cmap com mais cores (tab20 tem até 20 cores)
    cmap = plt.get_cmap('tab20', len(unique_communities))

    # Criar subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8), gridspec_kw={'width_ratios': [3, 1]})

    # Filtrar as comunidades que têm peso maior que zero
    active_communities = {comm: weight for comm, weight in community_weights.items() if weight > 0}
    
    # Verifique se as comunidades ativas estão presentes em unique_communities
    active_communities_filtered = {comm: weight for comm, weight in active_communities.items() if comm in unique_communities}
    
    active_nodes = [node for node in G.nodes() if partition[node] in active_communities_filtered]

    # Desenhar apenas as comunidades ativas no grafo
    node_colors = [cmap(unique_communities.index(partition[node])) for node in active_nodes]
    nx.draw_networkx_nodes(G, pos, nodelist=active_nodes, node_size=300, node_color=node_colors, ax=ax1)
    nx.draw_networkx_edges(G, pos, edgelist=[(u, v) for u, v in G.edges() if partition[u] in active_communities_filtered and partition[v] in active_communities_filtered], edge_color='black', ax=ax1)
    nx.draw_networkx_labels(G, pos, font_size=10, ax=ax1)

    # Criar patches para a legenda com base nas comunidades ativas
    patches = [mpatches.Patch(color=cmap(unique_communities.index(comm)), label=f'Comunidade {comm}') for comm in active_communities_filtered]
    ax1.legend(handles=patches, loc='best')

    ax1.set_title(title)

    # Exibir a legenda amarela com as informações das comunidades ativas
    ax2.axis('off')
    text_log = log_message + "\n\n" if log_message else ""
    if community_weights:
        weight_text = "\n".join([f"Comunidade {comm}: Peso Total = {weight:.2f}" for comm, weight in sorted(active_communities_filtered.items())])
        text_log += weight_text

    # Exibir o log dos estímulos e pesos das comunidades
    ax2.text(0.5, 0.5, text_log, fontsize=14, ha='center', va='center', wrap=True, 
             bbox=dict(facecolor='yellow', alpha=0.5, edgecolor='black'))

    plt.tight_layout()
    plt.show()
