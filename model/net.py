import torch
from torch import nn
import numpy as np
from torch.nn import functional as F
# from GNN import GNN
from model.SamplingGNN import SaGNN as GNN
from sampling import multi_hop_sampling


class Net(nn.Module):
    def __init__(self, user_input_dim, item_input_dim, proj_dim, hidden_dim,
                 edge_dim, dropout=0., num_neighbor_list=[100, 100],
                 activation=torch.sigmoid, use_edge_feature=True):
        """

        :param user_input_dim:
        :param item_input_dim:
        :param hidden_dim: 各隐藏层的维度
        :param edge_dim:
        :param sampling:
        :param dropout:
        :param activation:
        :param use_edge_feature:
        """

        super(Net, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # self.sampling = sampling
        self.activation = activation
        self.use_edge_feature = use_edge_feature
        self.proj_user = nn.Linear(user_input_dim, proj_dim)
        self.proj_item = nn.Linear(item_input_dim, proj_dim)
        self.gnn_user = GNN(proj_dim, hidden_dim, num_neighbor_list)
        self.gnn_item = GNN(proj_dim, hidden_dim, num_neighbor_list)
        self.num_neighbor_list = num_neighbor_list
        self.linear = nn.Linear(hidden_dim[-1], 1)
        # self.proj_dim = proj_dim
        # if use_edge_feature:
        #     self.linear = nn.Linear(hidden_dim[0] * len(hidden_dim) + edge_dim, 2)
        # else:
        #     self.linear = nn.Linear(hidden_dim[0] * len(hidden_dim), 2)
        # self.sample = Sample(hidden_dim+embedding_dim*3, edge_dim)

    def forward(self, user_id, item_id,
                user_feat, item_feat,
                edge_feature, num_sampling,
                user2item, item2user):

        # 先对特征进行映射
        print("特征映射...")
        user_proj_feat = self.activation(self.proj_user(user_feat[user_id]))
        item_proj_feat = self.activation(self.proj_item(item_feat[item_id]))


        print("采样邻居特征...")
        sampling_user_feat, sampling_item_feat = \
            self.sampling_neighbor_feature(user_id, item_id,
                                           user_proj_feat, item_proj_feat,
                                           user2item, item2user)
        user_hidden = self.gnn_user(sampling_user_feat)
        item_hidden = self.gnn_item(sampling_item_feat)
        print("预测边...")
        user_item_pred = torch.mul(user_hidden, item_hidden)

        return self.linear(user_item_pred), num_sampling

        # x = torch.mul(embd_user, embd_item)
        # if self.use_edge_feature:
        #     x = torch.cat((x, edge_feature), 1)
        # x = self.linear(x)
        # return self.activation(x)


    def sampling_neighbor_feature(self, user_id,
                                  item_id,
                                  user_feat,
                                  item_feat,
                                  user2item,
                                  item2user):
        """

        :param user_feat: pos_indices
        :param item_feat:
        :param adj:
        :return:
        """

        sampling_src_id = multi_hop_sampling(user_id, self.num_neighbor_list, user2item, item2user)
        sampling_dst_id = multi_hop_sampling(item_id, self.num_neighbor_list, item2user, user2item)
        sampling_src_x = []
        sampling_dst_x = []
        for i, nodes_id in enumerate(sampling_src_id):
            if i % 2 == 0:
                sampling_src_x.append(torch.from_numpy(user_feat[nodes_id]).float())
                sampling_dst_x.append(torch.from_numpy(item_feat[sampling_dst_id[i]]).float())
            else:
                sampling_src_x.append(torch.from_numpy(item_feat[nodes_id]).float())
                sampling_dst_x.append(torch.from_numpy(user_feat[sampling_dst_id[i]]).float())

        return sampling_src_x, sampling_dst_x


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

