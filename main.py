# main.py
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from gcn_model import GCN, train_model
from graph_builder import initialize_graph, graph_to_data, simulate_plasticity_by_community, calculate_community_weights
import graph_builder
from visualization import visualize_graph_with_clustering_and_weights
from community import community_louvain

# Funções de visualização

def plot_community_weight_changes(community_weights_over_time):
    # Plota mudanças nos pesos das comunidades ao longo das épocas
    num_epochs = len(community_weights_over_time)
    community_ids = sorted(list(community_weights_over_time[0].keys()))
    weight_changes = {community: [] for community in community_ids}

    # Calcula mudanças acumuladas de pesos para cada comunidade
    for epoch in range(1, num_epochs):
        for community in community_ids:
            prev_weight = community_weights_over_time[epoch-1].get(community, 0)
            curr_weight = community_weights_over_time[epoch].get(community, 0)
            change = curr_weight - prev_weight
            weight_changes[community].append(change)

    # Plota os dados
    plt.figure(figsize=(12, 6))
    
    for community in community_ids:
        plt.bar(np.arange(1, num_epochs), weight_changes[community], label=f'Comunidade {community}')

    plt.xlabel('Época', fontsize=14)
    plt.ylabel('Mudança acumulada no Peso Total', fontsize=14)
    plt.title('Fortalecimento/Enfraquecimento das Comunidades ao Longo das Épocas', fontsize=16)
    plt.legend(loc='upper right', fontsize=10, title="Comunidades", title_fontsize=12)
    plt.xticks(np.arange(1, num_epochs + 1, step=1))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_loss_heatmap(losses):
    # Plota heatmap das perdas ao longo das épocas
    plt.figure(figsize=(8, 4))
    sns.heatmap(np.array(losses).reshape(1, -1), annot=True, cmap="YlGnBu", cbar=True, xticklabels=np.arange(1, len(losses) + 1))
    plt.title('Heatmap das Perdas ao Longo das Épocas')
    plt.xlabel('Época')
    plt.yticks([])
    plt.show()

# Loop de treinamento

def log_stimulus_application(log, epoch, stimulated_community, stimulus_type, stimulus_value):
    # Registra detalhes sobre estímulo aplicado no log
    log_message = (f"Epoch {epoch + 1}: Comunidade {stimulated_community} recebeu "
                   f"estímulo {stimulus_type} com valor médio de modificação {stimulus_value:.2f}.")
    log.append(log_message)
    return log_message

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GCN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

log = []
G = initialize_graph()  # Inicializa o grafo
partition = community_louvain.best_partition(G)

epochs = 5
losses = []
community_weights_over_time = []

# Executa o treinamento por múltiplas épocas
for epoch in range(epochs):
    print(f'\n=== Epoch {epoch + 1} ===')
    
    # Treina o modelo e aplica estímulo
    loss, stimulated_community, stimulus_type = train_model(
        G, model, graph_to_data(G), optimizer, criterion, log, epoch, partition, graph_builder
    )
    
    # Simula a plasticidade
    stimulus_value = simulate_plasticity_by_community(
        G, spike_times={}, strengthen_prob=0.2, weaken_prob=0.1, stimulus_type=stimulus_type, stimulated_community=stimulated_community
    )
    
    # Registra os estímulos aplicados
    log_message = log_stimulus_application(log, epoch, stimulated_community, stimulus_type, stimulus_value)
    
    print(f'Loss: {loss:.4f}')
    losses.append(loss)

    # Calcula o peso das comunidades
    community_weights = calculate_community_weights(G, partition)
    community_weights_over_time.append(community_weights)

    # Visualiza o grafo com pesos atualizados
    visualize_graph_with_clustering_and_weights(G, title=f"Grafo na Epoch {epoch + 1}", log_message=log_message, community_weights=community_weights)

# Exibe o log completo de estímulos
print("\nLog Completo de Estímulos:")
for entry in log:
    print(entry)
