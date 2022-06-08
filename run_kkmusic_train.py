#!/usr/bin/env python
# coding: utf-8

# In[29]:


import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader
import numpy as np
import pandas as pd
import random
import os.path as op
import scipy
import sklearn

from net.net import Net
# from utils import NetworkData
from kkmusic_data import KKMuicData
from sampling import multi_hop_sampling

from tqdm import tqdm



learning_rate = 0.02
weight_decay = 5e-4
num_epochs = 5
batch_size = 2
validation_split=.2



data = KKMuicData()
(user2item, item2user), x_user, x_item, x_train_edge, x_test_edge = data.get_data()

train_size = int(validation_split * len(data))
test_size = len(data) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(data, [train_size, test_size])
train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


# norm
x_user = x_user / x_user.sum(1, keepdims=True)
x_item = x_item / x_item.sum(1, keepdims=True)


# to tensor
x_user = torch.from_numpy(np.float32(x_user))
x_item = torch.from_numpy(np.float32(x_item))

num_user_nodes, user_input_dim = x_user.shape
num_item_nodes, item_input_dim = x_item.shape
hidden_dim = [200, 200]
proj_dim = 500
edge_dim = x_train_edge.shape[1] - 2
num_nodes = num_item_nodes + num_user_nodes
unified_feat_dim = max(user_input_dim, item_input_dim)

if isinstance(x_user, np.ndarray):
    x_user = np.concatenate([x_user,
                         np.zeros((x_user.shape[0],
                                   unified_feat_dim-x_user.shape[1]))],
                        axis=1)
if isinstance(x_user, torch.Tensor):
    x_user = torch.cat([x_user, torch.zeros((x_user.shape[0], unified_feat_dim-x_user.shape[1]))], 1)

model = Net(user_input_dim=unified_feat_dim,
            item_input_dim=unified_feat_dim,
            proj_dim=proj_dim,
            hidden_dim=hidden_dim,
            edge_dim=edge_dim,
            num_neighbor_list=[5, 5],
            use_edge_feature=False)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
criterion = nn.CrossEntropyLoss


def sampling_neighbor_feature(user_id, item_id,
                              x_user, x_item,
                              user2item, item2user,
                              num_neighbor_list):
    """

    :param user_id:
    :param item_id:
    :param x_user:
    :param x_item:
    :param user2item:
    :param item2user:
    :param num_neighbor_list:
    :return:
    """

    sampling_src_id = multi_hop_sampling(user_id, num_neighbor_list, user2item, item2user)
    sampling_dst_id = multi_hop_sampling(item_id, num_neighbor_list, item2user, user2item)
    sampling_src_x = []
    sampling_dst_x = []
    for i, nodes_id in enumerate(sampling_src_id):
        # if not isinstance(nodes_id, np.ndarray):
        nodes_id = np.array(nodes_id, dtype=int)
        if len(nodes_id) == 0:
            sampling_src_x.append(np.zeros((unified_feat_dim, )))
            continue
        if i % 2 == 0:
            print(x_user[nodes_id].shape)
            sampling_src_x.append(x_user[nodes_id])
        else:
            print(x_item[nodes_id].shape)
            sampling_src_x.append(x_item[nodes_id])
    for i, nodes_id in enumerate(sampling_dst_id):
        # if not isinstance(nodes_id, np.ndarray):
        nodes_id = np.array(nodes_id, dtype=int)
        if len(nodes_id) == 0:
            sampling_dst_x.append(np.zeros((unified_feat_dim, )))
            continue
        if i % 2 == 0:
            print(x_item[nodes_id].shape)
            sampling_dst_x.append(x_item[nodes_id])
        else:
            print(x_user[nodes_id].shape)
            sampling_dst_x.append(x_user[nodes_id])
    return sampling_src_x, sampling_dst_x


# train
def train():
    loss_history = []
    acc_history = []
    model.train()
    for epoch in range(num_epochs):
        for uid, iid, y in tqdm(train_dataloader):
            sampling_user_x, sampling_item_x = sampling_neighbor_feature(uid, iid, x_user, x_item,
                                                                         user2item, item2user, model.num_neighbor_list)
            # print(sampling_item_x, sampling_item_x)
            # for x in sampling_item_x:
            #     print(x)
            # for x in sampling_user_x:
            #     print(x)
            # sampling_user_x, sampling_item_x = \
            #     torch.from_numpy(np.array(sampling_user_x)).float(), torch.from_numpy(np.array(sampling_item_x)).float()
            logits = model(sampling_user_x, sampling_item_x)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_history.append(loss.item())
            acc = accuracy(logits, y)
            acc_history.append(acc.item())
            print("Epoch {:03d}: Loss {:.4f}, TrainAcc {:.4f}".format(
                epoch, loss.item(), acc.item()
            ))

    return loss_history, acc_history


# test
def accuracy(logits, y):
    ones = torch.ones_like(logits)
    zeros = torch.zeros_like(logits)
    y_hat = torch.where(logits > 0.5, ones, logits)
    y_hat = torch.where(logits <= 0.5, zeros, logits)
    return (y_hat == y).sum() / len(y_hat)

# plot
from matplotlib import pyplot as plt
def plot_loss_and_acc(loss_history, acc_history):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(range(len(loss_history)), loss_history,
             c=np.array([255, 71, 90]) / 255.)
    plt.ylabel('Loss')

    ax2 = fig.add_subplot(111, sharex=ax1, frameon=False)
    ax2.plot(range(len(acc_history)), acc_history,
             c=np.array([79, 179, 255]) / 255.)
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    plt.ylabel('TrainAcc')

    plt.xlabel('Epoch')
    plt.title('Training Loss & Training Accuracy')
    plt.show()



loss, acc = train()
plot_loss_and_acc(loss, acc)

