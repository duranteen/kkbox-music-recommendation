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

    def forward(self, adj,  neg_adj, L, user_feat, item_feat, edge_feature, sampling=0.1):

        # 先对特征进行映射
        user_proj_feat = self.activation(self.proj_user(user_feat))
        item_proj_feat = self.activation(self.proj_item(item_feat))

        # 采样大小
        num_sampling = sampling * len(adj.tocoo().data)

        # 选取正样本
        pos_indices = np.array([L.row, L.col]).astype("int64").T
        pos_index = np.random.choice(range(len(pos_indices)), size=(num_sampling // 2,))
        pos_src_dst_index = pos_indices[pos_index]
        # 选取负样本
        neg_indices = np.array([neg_adj.row, neg_adj.col]).astype("int64").T
        neg_index = np.random.choice(range(len(neg_indices)), size=(num_sampling // 2,))
        neg_src_dst_index = neg_indices[[neg_index]]

        # sampled_user_feat = user_proj_feat[list(pos_src_dst_index[:, 0])]
        # sampled_item_feat = item_proj_feat[list(pos_src_dst_index[:, 1])]

        sampling_user_feat, sampling_item_feat = self.sampling_neighbor_feature(pos_src_dst_index,
                                                                                neg_src_dst_index,
                                                                                user_proj_feat,
                                                                                item_proj_feat,
                                                                                pos_indices)
        user_hidden = self.gnn_user(sampling_user_feat)
        item_hidden = self.gnn_item(sampling_item_feat)

        user_item_pred = torch.mul(user_hidden, item_hidden)

        return self.linear(user_item_pred), num_sampling

        # x = torch.mul(embd_user, embd_item)
        # if self.use_edge_feature:
        #     x = torch.cat((x, edge_feature), 1)
        # x = self.linear(x)
        # return self.activation(x)

    def sampling_neighbor_feature(self, pos_src_dst_index, neg_src_dst_index, user_feat, item_feat, adj):
        """

        :param user_feat: pos_indices
        :param item_feat:
        :param adj:
        :return:
        """
        src_dst_index = np.concatenate([pos_src_dst_index, neg_src_dst_index], axis=0)

        # 构建邻接表 dict
        user2item = {}
        item2user = {}
        for i in range(len(adj)):
            uid, iid = adj[i][0], adj[i][1]
            if uid in user2item:
                user2item[uid].append(iid)
            else:
                user2item[uid] = [iid]
            if iid in item2user:
                item2user[iid].append(uid)
            else:
                item2user[iid] = [uid]

        sampling_src_id = multi_hop_sampling(src_dst_index[:, 0],
                                                self.num_neighbor_list, user2item)
        sampling_dst_id = multi_hop_sampling(src_dst_index[:, 1],
                                                self.num_neighbor_list, item2user)
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

