import torch
from torch import nn
from torch.nn import init, functional as F


class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super(GNN, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim1)
        self.gcn_layer1 = GCNLayer(hidden_dim1, hidden_dim2)
        self.gcn_layer2 = GCNLayer(hidden_dim2, hidden_dim2)
        self.linear2 = nn.Linear(hidden_dim2, output_dim)

    def forward(self, adjacency, features):
        h = F.sigmoid(self.linear1(features))
        h = F.relu(self.gcn_layer1(adjacency, h))
        h = F.relu(self.gcn_layer2(adjacency, h))
        x = self.linear2(h)
        return x


class GCNLayer(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=True):
        super(GCNLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim), requires_grad=True)
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_normal_(self.weight)
        if self.use_bias:
            init.zeros_(self.bias)

    def forward(self, adjacency, features):
        h = torch.mm(features, self.weight)
        h = torch.sparse.mm(adjacency, h)
        if self.use_bias:
            h = h.clone() + self.bias
        return h