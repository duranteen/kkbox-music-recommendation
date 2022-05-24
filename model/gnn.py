import torch
from torch import nn
from torch.nn import init, functional as F


class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim,
                 num_layers=3, dropout=0., activation=F.relu):
        """
        use 3 layers and residual GCNs
        :param input_dim:
        :param hidden_dim1:
        :param hidden_dim2:
        :param output_dim:
        :param num_layers: 3 layers
        :param dropout:
        """
        super(GNN, self).__init__()
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.gcn1 = GCNLayer(hidden_dim, embedding_dim)
        self.gcn2 = GCNLayer(embedding_dim, embedding_dim)
        # self.gcn3 = GCNLayer(embedding_dim, embedding_dim)

    def forward(self, adjacency, features):
        """
        first, fully-connected layer transform features (project);
        then, into 3 layers of GCNs
        :param adjacency:
        :param features:
        :return:
        """
        projected_x = self.linear(features)
        x = F.sigmoid(projected_x)
        # GCN layer
        gcn1_x = self.gcn1(adjacency, x)
        x = self.activation(gcn1_x)
        gcn2_x = self.gcn2(adjacency, x)
        x = self.activation(gcn2_x)
        gcn3_x = self.gcn2(adjacency, x)
        return torch.cat([projected_x, gcn1_x, gcn2_x, gcn3_x], 1)


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