# gcn_model.py
import random
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool

class GCN(nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        # Define camadas convolucionais e fully-connected
        self.conv1 = GCNConv(1, 16)
        self.conv2 = GCNConv(16, 32)
        self.fc1 = nn.Linear(32, 10)
        self.fc2 = nn.Linear(10, 2)

    def forward(self, data):
        # Executa o forward pass do modelo
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = global_mean_pool(x, torch.zeros(x.size(0), dtype=torch.long))  # Pooling global
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x

def train_model(G, model, data, optimizer, criterion, log, epoch, partition, graph_builder):
    # Treina o modelo e aplica plasticidade à comunidade selecionada
    model.train()
    optimizer.zero_grad()
    output = model(data)
    target = torch.tensor([0], dtype=torch.long).to(data.x.device)  # Target fixo
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

    # Escolhe uma comunidade aleatória para aplicar o estímulo
    stimulated_community = random.choice(list(set(partition.values())))
    stimulus_type = random.choice(["sensorial", "motor", "emocional", "cognitivo"])

    # Aplica plasticidade na comunidade estimulada
    G = graph_builder.simulate_plasticity_by_community(G, spike_times={}, strengthen_prob=0.2, weaken_prob=0.1, stimulus_type=stimulus_type, stimulated_community=stimulated_community)

    return loss.item(), stimulated_community, stimulus_type
