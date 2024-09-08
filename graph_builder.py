# graph_builder.py
import random
import networkx as nx
from community import community_louvain
import torch
from torch_geometric.data import Data

def initialize_graph():
    # Inicializa um grafo Erdos-Renyi com pesos aleatórios
    G = nx.erdos_renyi_graph(n=100, p=0.1)
    for (u, v) in G.edges():
        G[u][v]['weight'] = random.uniform(0.5, 1.5)  # Define pesos aleatórios nas arestas
    return G

def simulate_plasticity_by_community(G, spike_times, strengthen_prob=0.1, weaken_prob=0.1, stimulus_type="sensorial", stimulated_community=None):
    # Simula plasticidade ao modificar pesos das arestas da comunidade selecionada
    weight_modifiers = {
        "sensorial": 1.2,
        "motor": 1.1,
        "cognitivo": 1.3,
        "emocional": 1.4
    }
    weight_modifier = weight_modifiers.get(stimulus_type, 1.0)
    stdp_factor = {"strengthen": 1.05, "weaken": 0.95}  # Define fatores para fortalecimento e enfraquecimento

    total_stimulus_value = 0
    num_modified_edges = 0

    partition = community_louvain.best_partition(G)

    # Modifica pesos de acordo com a comunidade e probabilidades
    for u, v in list(G.edges()):
        if stimulated_community is None or (partition[u] == stimulated_community and partition[v] == stimulated_community):
            if random.random() < weaken_prob:  # Aplicar enfraquecimento
                if G.degree(u) > 1 and G.degree(v) > 1:
                    G[u][v]['weight'] *= stdp_factor["weaken"]
                    if G[u][v]['weight'] < 0.1:
                        G.remove_edge(u, v)
                    else:
                        total_stimulus_value += stdp_factor["weaken"]
                        num_modified_edges += 1
            elif random.random() < strengthen_prob:  # Aplicar fortalecimento
                G[u][v]['weight'] *= stdp_factor["strengthen"] * weight_modifier
                total_stimulus_value += stdp_factor["strengthen"] * weight_modifier
                num_modified_edges += 1

    if num_modified_edges > 0:
        average_stimulus_value = total_stimulus_value / num_modified_edges  # Calcula valor médio de estímulo
    else:
        average_stimulus_value = 0

    return average_stimulus_value

def calculate_community_weights(G, partition):
    # Calcula o peso total das arestas dentro de cada comunidade
    community_weights = {community: 0 for community in set(partition.values())}

    # Somar pesos das arestas dentro de cada comunidade
    for u, v, data in G.edges(data=True):
        comm_u = partition[u]
        comm_v = partition[v]

        if comm_u == comm_v:
            community_weights[comm_u] += data['weight']

    return community_weights

def graph_to_data(G):
    # Converte o grafo NetworkX para o formato PyTorch Geometric Data
    edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor([G[u][v]['weight'] for u, v in G.edges], dtype=torch.float).view(-1, 1)
    node_features = torch.ones((len(G.nodes), 1), dtype=torch.float)  # Atributos dos nós

    return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)
