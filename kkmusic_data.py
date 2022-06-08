# build adjacency and normalize adjacency
import itertools
import os.path as op

import pandas as pd
from scipy import sparse as sp
import numpy as np
import time
import datetime

from torch.utils.data import dataset
from torch.utils.data import dataloader


class KKMuicData(dataset.Dataset):
    def __init__(self, data_root='data/', train=True, cache='data/cache/', save=False):
        self.data_root = data_root
        self.train = train
        self.cache = cache
        self.save = save

        if self.train:
            self.data = pd.read_csv(op.join(self.data_root, 'train_data.csv'))
        else:
            self.data = pd.read_csv(op.join(self.data_root, 'test_data.csv'))
        self.train_data = pd.read_csv(op.join(self.data_root, 'train_data.csv'))
        self.test_data = pd.read_csv(op.join(self.data_root, 'test_data.csv'))
        self.mem_info = pd.read_csv(op.join(self.data_root, 'mem_info.csv'))
        self.song_info = pd.read_csv(op.join(self.data_root, 'song_info.csv'))

        self.n_users = len(set(self.train_data['msno'].tolist()).union(set(self.test_data['msno'].tolist()))) + 1
        self.n_items = len(set(self.train_data['song_id'].tolist()).union(set(self.test_data['song_id'].tolist()))) + 1

        self.train_data = self.train_data[self.train_data.target == 1]
        # print(self.n_users, self.n_items)
        # print(self.song_info['song_id'].max())
        self.test_data.drop('id', axis=1, inplace=True)
        self.y_train = self.data['target']
        self.train_data.drop('target', axis=1, inplace=True)

        # adjacency and L_thelta
        # self.adj, self.L = self.build_adjacency()
        self.adj_table = self.build_adjacency()

        self.x_user, self.x_item = \
            self.build_node_feature(self.mem_info, 'mem'), self.build_node_feature(self.song_info, 'song')
        self.x_train_edge, self.x_test_edge = \
            self.build_edge_feature(self.train_data), self.build_edge_feature(self.test_data)


    def __getitem__(self, item):
        # feature
        return self.data['msno'].tolist()[item], self.data['song_id'].tolist()[item], self.y_train[item]


    def __len__(self):
        return len(self.data)


    def get_data(self):
        print("get data: ", end='\t')
        return self.adj_table, self.x_user, self.x_item, self.x_train_edge, self.x_test_edge



    def build_node_feature(self, data, who='mem'):
        print("building node features ...")
        features = []
        if who == 'song':
            index_data = data.set_index('song_id')
            num_col = len(index_data.columns)
            for i in range(self.n_items):
                if i in index_data.index:
                    features.append(np.array(index_data.loc[i]))
                else:
                    features.append(np.zeros((num_col,)))

        if who == 'mem':
            src_dates = data['registration_init_time']
            dst_dates = data['expiration_date']
            diffs = []
            for i in range(len(src_dates)):
                src = time.strptime(src_dates[i], "%Y/%m/%d")
                dst = time.strptime(dst_dates[i], "%Y/%m/%d")
                diff = datetime.date(src[0], src[1], src[2]) - datetime.date(dst[0], dst[1], dst[2])
                diffs.append(diff.days)
            data['date_diff'] = diffs
            data.drop(columns=['registration_init_time', 'expiration_date'], inplace=True)
            index_data = data.set_index('msno')
            num_col = len(index_data.columns)
            for i in range(self.n_users):
                if i in index_data.index:
                    features.append(np.array(index_data.loc[i]))
                else:
                    features.append(np.zeros((num_col,)))
        return np.array(features)

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
        # self.R = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)
        # # if train:
        # #     data = self.train_data
        # # else:
        # #     data = self.test_data
        # for i in range(len(self.train_data)):
        #     uid, iid = self.train_data['msno'][i], self.train_data['song_id'][i]
        #     self.R[uid, iid] = 1
        #     # self.R[iid, uid] = 1
        # adj = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items),
        #                     dtype=np.float32)
        # adj = adj.tolil()
        # R = self.R.tolil()
        #
        # adj[: self.n_users, self.n_users:] = R
        # adj[self.n_users:, :self.n_users] = R.T
        # adj = adj.todok()
        # return adj, self.normalize_adjacency(adj)
        user2item = {}
        item2user = {}
        uids, iids = self.train_data['msno'].tolist(), self.train_data['song_id'].tolist()
        for i in range(len(self.train_data)):
            uid, iid = uids[i], iids[i]
            if uid in user2item:
                user2item[uid].append(iid)
            else:
                user2item[uid] = [iid]
            if iid in item2user:
                item2user[iid].append(uid)
            else:
                item2user[iid] = [uid]
        return user2item, item2user

    # def normalize_adjacency(self, adj):
    #     """
    #     L = D^-0.5 * (A + I) * D^-0.5
    #     :param adj:
    #     :return:
    #     """
    #     print("normalize adjacency ...")
    #     adj += sp.eye(adj.shape[0])
    #     row_sum = np.array(adj.sum(1))
    #     d_sqrt = np.power(row_sum, -0.5).flatten()
    #     d_sqrt[np.isinf(d_sqrt)] = 0.
    #     d_sqrt_mat = sp.diags(d_sqrt)
    #
    #     L = d_sqrt_mat.dot(adj).dot(d_sqrt_mat)
    #     return L.tocoo()




if __name__ == '__main__':
    kk = KKMuicData()

    (user2item, item2user), x_user, x_item, _, _ = kk.get_data()

    loader = dataloader.DataLoader(dataset=kk, batch_size=32, shuffle=True)

    print(x_user.shape, x_item.shape)
    print(kk.n_users, kk.n_items)

    # for uid, iid, y in loader:
    #     print(uid, iid, y)
