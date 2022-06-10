import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init


class MessagePassing(nn.Module):
    def __init__(self, input_dim, output_dim,
                 use_bias=True):
        """

        :param input_dim:
        :param output_dim:
        :param use_bias:
        :param activation:
        """
        super(MessagePassing, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim), requires_grad=True)
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(self.output_dim), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.weight)
        if self.use_bias:
            init.zeros_(self.bias)

    def forward(self, neighbor_feature):
        """

        :param neighbor_feature:
        :return:
        """
        neighbor_message = neighbor_feature.mean(dim=1)
        neighbor_hidden = torch.matmul(neighbor_message, self.weight)
        if self.use_bias:
            neighbor_hidden += self.bias
        return neighbor_hidden


class GCNLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim,
                 activation=F.leaky_relu):
        """

        :param input_dim:
        :param hidden_dim:
        :param activation:
        """
        super(GCNLayer, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.aggregator = MessagePassing(input_dim, hidden_dim)
        self.weight = nn.Parameter(torch.Tensor(input_dim, hidden_dim), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.weight)

    def forward(self, src_features, neighbor_features):
        neighbor_hidden = self.aggregator(neighbor_features)
        self_hidden = torch.matmul(src_features, self.weight)
        hidden = self_hidden + neighbor_hidden
        if self.activation:
            hidden = self.activation(hidden)
        return hidden


class SaGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=[256, 256], num_neighbor_list=[10, 10]):
        """

        :param input_dim:
        :param hidden_dim:
        :param num_neighbor_list:
        """
        super(SaGNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_neighbor_list = num_neighbor_list
        self.num_layers = len(num_neighbor_list)
        self.gcn = nn.ModuleList()
        self.gcn.append(GCNLayer(input_dim, hidden_dim[0]))
        for index in range(0, len(hidden_dim) - 2):
            self.gcn.append(GCNLayer(hidden_dim[index], hidden_dim[index + 1]))
        self.gcn.append(GCNLayer(hidden_dim[-2], hidden_dim[-1], activation=None))

    def forward(self, node_features_list):
        """

        :param node_features_list:
        :return:
        """
        # print("GCN message passing...")
        hidden = node_features_list
        for layer in range(self.num_layers):
            next_hidden = []
            gcn = self.gcn[layer]
            for hop in range(self.num_layers - layer):
                src_feat = hidden[hop]
                src_feat_len = len(src_feat)
                neighbor_node_feat = hidden[hop+1].view(
                    (src_feat_len, self.num_neighbor_list[hop], -1))
                h = gcn(src_feat, neighbor_node_feat)
                next_hidden.append(h)
            hidden = next_hidden

        return hidden[0]