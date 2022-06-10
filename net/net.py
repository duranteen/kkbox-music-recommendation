import torch
from torch import nn
import numpy as np
from torch.nn import functional as F
# from GNN import GNN
from net.SamplingGNN import SaGNN as GNN
from sampling import multi_hop_sampling


class Net(nn.Module):
    def __init__(self, user_input_dim, item_input_dim, hidden_dim,
                 edge_dim, dropout=0., num_neighbor_list=[100, 100],
                 activation=F.leaky_relu, use_edge_feature=True):


        super(Net, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # self.sampling = sampling
        self.activation = activation
        self.use_edge_feature = use_edge_feature
        # self.proj_user = nn.Linear(user_input_dim, proj_dim)
        # self.proj_item = nn.Linear(item_input_dim, proj_dim)
        self.gnn_user = GNN(item_input_dim, hidden_dim, num_neighbor_list)
        self.gnn_item = GNN(item_input_dim, hidden_dim, num_neighbor_list)
        self.num_neighbor_list = num_neighbor_list
        self.linear = nn.Linear(hidden_dim[-1], 2)
        # self.proj_dim = proj_dim
        # if use_edge_feature:
        #     self.linear = nn.Linear(hidden_dim[0] * len(hidden_dim) + edge_dim, 2)
        # else:
        #     self.linear = nn.Linear(hidden_dim[0] * len(hidden_dim), 2)
        # self.sample = Sample(hidden_dim+embedding_dim*3, edge_dim)

    def forward(self, sampling_user_feat, sampling_item_feat):

        # 特征映射
        # user_feat = self.proj_user(sampling_user_feat)
        # item_feat = self.proj_item(sampling_item_feat)

        user_hidden = self.activation(self.gnn_user(sampling_user_feat))
        item_hidden = self.activation(self.gnn_item(sampling_item_feat))
        user_item_pred = self.linear(torch.mul(user_hidden, item_hidden))

        return torch.sigmoid(user_item_pred)

        # x = torch.mul(embd_user, embd_item)
        # if self.use_edge_feature:
        #     x = torch.cat((x, edge_feature), 1)
        # x = self.linear(x)
        # return self.activation(x)





"""
class Sample(nn.Module):
    def __init__(self, embd_dim, edge_dim, sample_ratio=0.1, use_edge_feature=False):
        super(Sample, self).__init__()
        self.sample_ratio = sample_ratio
        self.use_edge_feature = use_edge_feature
        if use_edge_feature:
            self.linear = nn.Linear(embd_dim+edge_dim, 1)
        else:
            self.linear = nn.Linear(embd_dim, 1)


    def forward(self, adjacency, embd_user, embd_item, edge_feat):
        """"""

        :param embd_user: 用户嵌入特征
        :param embd_item: 商品嵌入特征
        :param edge_feat: 边特征
        :return: 采样的节点对边的预测，采样序号
        """"""
        adj = adjacency.tocoo()
        num__sampling = self.sample_ratio * len(adj.data)
"""

