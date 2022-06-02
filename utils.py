# build adjacency and normalize adjacency
import itertools
import os.path as op

import pandas as pd
from scipy import sparse as sp
import numpy as np
import time
import datetime


class NetworkData(object):
    def __init__(self, data_root='data/', cache='data/cache/', save=False):
        self.data_root = data_root
        self.cache = cache
        self.save = save
        self.train_data = pd.read_csv(op.join(self.data_root, 'train_data.csv'))
        self.test_data = pd.read_csv(op.join(self.data_root, 'test_data.csv'))
        self.mem_info = pd.read_csv(op.join(self.data_root, 'mem_info.csv'))
        self.song_info = pd.read_csv(op.join(self.data_root, 'song_info.csv'))
        # self.train_data = self.train_data[self.train_data.target == 1]
        self.n_users = len(set(self.train_data['msno'].tolist())
                           .union(set(self.test_data['msno'].tolist()))) + 1
        self.n_items = len(set(self.train_data['song_id'].tolist())
                           .union(set(self.test_data['song_id'].tolist()))) + 1
        print(self.n_users, self.n_items)
        # print(self.song_info['song_id'].max())
        self.test_data.drop('id', axis=1, inplace=True)
        self.y_train = self.train_data['target']
        self.train_data.drop('target', axis=1, inplace=True)

        # adjacency and L_thelta
        self.adj, self.L = self.build_adjacency()

        # feature
        self.x_user, self.x_item = \
            self.build_node_feature(self.mem_info, 'mem'), self.build_node_feature(self.song_info, 'song')
        self.x_train_edge, self.x_test_edge = \
            self.build_edge_feature(self.train_data), self.build_edge_feature(self.test_data)

    def get_data(self):
        print("get data: ", end='\t')
        if self.cache is None:
            if self.save:
                path = 'data/cache/'
                sp.save_npz(path + 'sp_adj.npz', self.adj)
                # sp.save_npz(path + 'sp_normal_L.npz', self.L)
                sp.save_npz(path + 'sp_x_user.npz', self.x_user)
                sp.save_npz(path + 'sp_x_item.npz', self.x_item)
                sp.save_npz(path + 'sp_x_train_edge.npz', self.x_train_edge)
                sp.save_npz(path + 'sp_x_test_edge.npz', self.x_test_edge)
                print("cache saved to %s" % path[:-1])
            return self.adj, self.L, self.x_user, self.x_item, self.x_train_edge, self.x_test_edge, self.y_train
        else:
            print("using cache %s ..." % self.cache)
            adj = sp.load_npz(self.cache + 'sp_adj.npz')
            L = sp.load_npz(self.cache + 'sp_normal_L.npz')
            x_user = sp.load_npz(self.cache + 'sp_x_user.npz')
            x_item = sp.load_npz(self.cache + 'sp_x_item.npz')
            x_train_edge = sp.load_npz(self.cache + 'sp_x_train_edge.npz')
            x_test_edge = sp.load_npz(self.cache + 'sp_x_test_edge.npz')
            return adj, L, x_user, x_item, x_train_edge, x_test_edge

    def id_mapping(self):
        # print(self.mem_info['id'].max())
        pass

    def build_node_feature(self, data, who='mem'):
        # print("data sorted by node_id...")
        print("building node features ...")
        features = None
        if who == 'song':
            # id_col = data.columns.to_list()[0]
            features = data.values[:, 1:]
            # feat_dict = {}
            # for i, id in enumerate(ids):
            #     feat_dict[id] = features[i]
        if who == 'mem':
            src_dates = data['registration_init_time']
            dst_dates = data['expiration_date']
            diffs = []
            for i in range(len(src_dates)):
                src = time.strptime(src_dates[i], "%Y/%m/%d")
                dst = time.strptime(dst_dates[i], "%Y/%m/%d")
                diff = datetime.date(src[0], src[1], src[2]) - datetime.date(dst[0], dst[1], dst[2])
                diffs.append(diff)
            data['date_diff'] = diffs
            data.drop(columns=['registration_init_time', 'expiration_date'], inplace=True)
            features = data.values[:, 1:]
        return features

    def build_edge_feature(self, data):
        print("building edge features ...")
        user_id, item_id = list(data['msno']), list(data['song_id'])
        if 'target' in data.columns:
            data.drop('target', axis=1, inplace=True)
        features = data.values[:, 2:]
        feat_dict = {}
        for i in range(len(user_id)):
            feat_dict['%d-%d' % (user_id[i], item_id[i])] = features[i]
        return feat_dict

    def build_adjacency(self, train=True):
        print("building adjacency ...")
        self.R = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)
        if train:
            data = self.train_data
        else:
            data = self.test_data
        for i in range(len(self.train_data)):
            uid, iid = self.train_data['msno'][i], self.train_data['song_id'][i]
            self.R[uid, iid] = 1
            # self.R[iid, uid] = 1
        adj = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items),
                            dtype=np.float32)
        adj = adj.tolil()
        R = self.R.tolil()

        adj[: self.n_users, self.n_users:] = R
        adj[self.n_users:, :self.n_users] = R.T
        adj = adj.todok()
        return adj, self.normalize_adjacency(adj)

    def normalize_adjacency(self, adj):
        """
        L = D^-0.5 * (A + I) * D^-0.5
        :param adj:
        :return:
        """
        print("normalize adjacency ...")
        adj += sp.eye(adj.shape[0])
        row_sum = np.array(adj.sum(1))
        d_sqrt = np.power(row_sum, -0.5).flatten()
        d_sqrt[np.isinf(d_sqrt)] = 0.
        d_sqrt_mat = sp.diags(d_sqrt)

        L = d_sqrt_mat.dot(adj).dot(d_sqrt_mat)
        return L.tocoo()

    def describe(self):
        print("number of users: %d" % self.n_users)
        print("number of songs: %d" % self.n_items)
        print("adjacency:", self.adj.shape)
        print("Lap:", self.L.shape)
        print("dim of user feature: %d" % len(self.x_user[0]))


if __name__ == '__main__':
    network_data = NetworkData(cache=None, save=True)
    out = network_data.get_data()
    network_data.describe()