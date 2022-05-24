import torch
from torch import nn
from torch.nn import functional as F
from gnn import GNN


class Net(nn.Module):

    def __init__(self, user_input_dim, item_input_dim, hidden_dim, embedding_dim, edge_dim, dropout=0., activation=F.sigmoid, use_edge_feature=True):
        super(Net, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.use_edge_feature = use_edge_feature
        self.gnn_user = GNN(user_input_dim, hidden_dim, embedding_dim)
        self.gnn_item = GNN(item_input_dim, hidden_dim, embedding_dim)
        if use_edge_feature:
            self.linear = nn.Linear(hidden_dim+embedding_dim*3+edge_dim, 1)
        else:
            self.linear = nn.Linear(hidden_dim+embedding_dim*3, 1)

    def forward(self, adjacency, user_feat, item_feat, edge_feature):
        embd_user = self.gnn_user(adjacency, user_feat)
        embd_item = self.gnn_item(adjacency, item_feat)

        x = torch.mul(embd_user, embd_item)
        if self.use_edge_feature:
            x = torch.cat((x, edge_feature), 1)

        x = self.linear(x)

        return self.activation(x)