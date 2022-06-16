# build adjacency and normalize adjacency
import json
import os.path as op
import os
import random

import pandas as pd
import torch
from scipy import sparse as sp
import numpy as np
import time
import datetime

from tqdm import tqdm

from torch.utils.data import dataset


class NetworkData(dataset.Dataset):
    def __init__(self, data_root='data/', cache='data/cache2/'):
        print("loading data ... ")
        self.data_root = data_root
        self.cache = cache
        self.train_data = pd.read_csv(op.join(self.data_root, 'train_data.csv'))
        self.data = self.train_data
        self.test_data = pd.read_csv(op.join(self.data_root, 'test_data.csv'))
        self.mem_info = pd.read_csv(op.join(self.data_root, 'mem_info.csv'))
        self.song_info = pd.read_csv(op.join(self.data_root, 'song_info.csv'))
        self.n_users = max(set(self.train_data['msno'].tolist()).union(set(self.test_data['msno'].tolist())))+1
        self.n_items = max(set(self.train_data['song_id'].tolist()).union(set(self.test_data['song_id'].tolist())))+1
        self.train_data = self.train_data[self.train_data.target == 1]
        print(self.n_users, self.n_items)
        self.user2item = {}
        self.neg_user2item = {}
        self.test_data.drop('id', axis=1, inplace=True)
        self.y_train = self.data['target']
        if 'target' in self.train_data.columns:
            self.train_data = self.train_data.drop('target', axis=1)

        try:
            fp = open(self.cache + 'user2item.json', 'r')
            self.user2item = json.load(fp)
            fp.close()
            fp = open(self.cache + 'neg_user2item.json', 'r')
            self.neg_user2item = json.load(fp)
            fp.close()
        except:
            src, dst = self.train_data['msno'].tolist(), self.train_data['song_id'].tolist()
            for i in tqdm(range(len(self.train_data)), desc="build user-item table=> "):
                uid, iid = src[i], dst[i]
                if uid in self.user2item:
                    self.user2item[uid].append(iid)
                else:
                    self.user2item[uid] = [iid]
            self.existing_users = self.user2item.keys()
            for u in tqdm(self.user2item, desc="build neg user-item table=> "):
                neg_items = list(set(range(self.n_items)) - set(self.user2item[u]))
                neg_items = list(np.random.choice(neg_items, 100))
                self.neg_user2item[u] = neg_items
            info = json.dumps(self.user2item, sort_keys=False, indent=4, separators=(',', ':'))
            fp = open(self.cache + 'user2item.json', 'w')
            fp.write(info)
            fp.close()
            info = json.dumps(self.neg_user2item, sort_keys=False, indent=4, separators=(',', ':'))
            fp = open(self.cache + 'neg_user2item.json', 'w')
            fp.write(info)
            fp.close()


        try:
            self.adj = sp.load_npz(self.cache + 'sp_adj.npz')
            self.L = sp.load_npz(self.cache + 'sp_normal_L.npz')
            self.x_user = np.load(self.cache + 'x_user.npy')
            self.x_item = np.load(self.cache + 'x_item.npy')
            self.x_train_edge = np.load(self.cache + 'x_train_edge.npy')
            self.x_test_edge = np.load(self.cache + 'x_test_edge.npy')
            print("using cache: %s" % self.cache)

        except FileNotFoundError:
            if not op.exists(self.cache):
                os.mkdir(self.cache)
                print("cache path not existing, mkdir %s" % self.cache)
            # adjacency
            self.adj, self.L = self.build_adjacency()
            # feature
            self.x_user, self.x_item = \
                self.build_node_feature(self.mem_info, 'mem'), self.build_node_feature(self.song_info, 'song')
            self.x_train_edge, self.x_test_edge = \
                self.build_edge_feature(self.train_data), self.build_edge_feature(self.test_data)
            sp.save_npz(self.cache + 'sp_adj.npz', self.adj)
            sp.save_npz(self.cache + 'sp_normal_L.npz', self.L)
            np.save(self.cache + 'x_user.npy', self.x_user)
            np.save(self.cache + 'x_item.npy', self.x_item)
            np.save(self.cache + 'x_train_edge.npy', self.x_train_edge)
            np.save(self.cache + 'x_test_edge.npy', self.x_test_edge)
            print("saved to %s" % self.cache)

    def __getitem__(self, item):
        return self.data['msno'][item], self.data['song_id'][item], self.y_train[item]

    def __len__(self):
        return len(self.data)

    def get_data(self):
        return self.adj, self.L, self.x_user, self.x_item, self.x_train_edge, self.x_test_edge

    def id_mapping(self):
        pass

    def build_node_feature(self, data, who='mem'):
        print("building node features ...")
        features = []
        data = data.fillna(0)
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
        # features = data.values[:, 2:]
        # edge_feat = np.concatenate([np.array(user_id), np.arange(item_id), features], axis=1)
        return data.values

    def build_adjacency(self):
        print("building adjacency ...")
        self.R = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)

        src, dst = self.train_data['msno'].tolist(), self.train_data['song_id'].tolist()
        for i in tqdm(range(len(self.train_data)), desc="build positive links=> "):
            uid, iid = src[i], dst[i]
            self.R[uid, iid] = 1

            # if uid in self.user2item:
            #     self.user2item[uid].append(iid)
            # else:
            #     self.user2item[uid] = [iid]

        adj = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items),
                            dtype=np.float32)
        adj = adj.tolil()
        R = self.R.tolil()
        adj[: self.n_users, self.n_users:] = R
        adj[self.n_users:, :self.n_users] = R.T
        adj = adj.todok()

        return adj.tocoo(), self.normalize_adjacency(adj)

    def normalize_adjacency(self, adj):
        """
        L = D^-0.5 * (A + I) * D^-0.5
        :param adj:
        :return:
        """
        print("normalize adjacency ...")
        # adj += sp.eye(adj.shape[0])
        row_sum = np.array(adj.sum(1))
        d_sqrt = np.power(row_sum, -0.5).flatten()
        d_sqrt[np.isinf(d_sqrt)] = 0.
        d_sqrt_mat = sp.diags(d_sqrt)

        L = d_sqrt_mat.dot(adj).dot(d_sqrt_mat)
        return L.tocoo()

    def sampling(self, users):
        def sample_pos(u, num_sampling):
            pos_items = self.user2item[u]
            n_pos_items = len(pos_items)
            sampling_result = []
            while len(sampling_result) < num_sampling:
                pos_item_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_item = pos_items[pos_item_id]
                if pos_item not in sampling_result:
                    sampling_result.append(pos_item)
            return sampling_result

        def sample_neg(u, num_sampling):
            sampling_result = []
            while len(sampling_result) < num_sampling:
                neg_item_id = np.random.randint(low=0, high=self.n_items, size=1)[0]
                if neg_item_id not in self.user2item[u] and neg_item_id not in sampling_result:
                    sampling_result.append(neg_item_id)
            return sampling_result

        pos_items, neg_items = [], []
        if isinstance(users, torch.Tensor):
            users = users.cpu().numpy()
        sampling_users = []
        for u in users:
            if u in self.user2item:
                sampling_users.append(u)
                pos_items += sample_pos(u, 1)
                neg_items += sample_neg(u, 1)

        return sampling_users, pos_items, neg_items

    def convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.from_numpy(np.asarray([coo.row, coo.col]).astype("int64")).long()
        v = torch.from_numpy(coo.data.astype(np.float32))
        return torch.sparse.FloatTensor(i, v, coo.shape)


    def describe(self):
        print("number of users: %d" % self.n_users)
        print("number of songs: %d" % self.n_items)
        print("adjacency:", self.adj.shape)
        print("Lap:", self.L.shape)
        print("dim of user feature: %d" % len(self.x_user[0]))


if __name__ == '__main__':
    network_data = NetworkData(cache=None, save=True)
    out = network_data.get_data()
    # network_data.describe()
