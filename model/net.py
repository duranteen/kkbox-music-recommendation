import torch
from torch import nn
from torch.nn import functional as F
from model.gnn import GNN


class Net(nn.Module):
    def __init__(self, user_input_dim, item_input_dim, hidden_dim,
                 embedding_dim, edge_dim, dropout=0.,
                 activation=F.sigmoid, use_edge_feature=True):
        super(Net, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.use_edge_feature = use_edge_feature
        self.proj_user = nn.Linear(user_input_dim, hidden_dim)
        self.proj_item = nn.Linear(item_input_dim, hidden_dim)
        self.gnn_user = GNN(hidden_dim, embedding_dim)
        self.gnn_item = GNN(hidden_dim, embedding_dim)
        if use_edge_feature:
            self.linear = nn.Linear(hidden_dim+embedding_dim*3+edge_dim, 1)
        else:
            self.linear = nn.Linear(hidden_dim+embedding_dim*3, 1)

    def forward(self, adjacency, user_feat, item_feat, edge_feature):
        user_proj_feat = self.activation(self.proj_user(user_feat))
        item_proj_feat = self.activation(self.proj_item(item_feat))

        feat = torch.cat((user_proj_feat, item_proj_feat), 0)
        if feat.shape[0] < adjacency.shape[0]:
            feat = torch.cat((feat, torch.zeros((adjacency.shape[0]-feat.shape[0], feat.shape[1]), dtype=torch.float)), 1)

        embd_user = self.gnn_user(adjacency, feat)
        embd_item = self.gnn_item(adjacency, feat)

        x = torch.mul(embd_user, embd_item)
        if self.use_edge_feature:
            x = torch.cat((x, edge_feature), 1)

        x = self.linear(x)

        return self.activation(x)