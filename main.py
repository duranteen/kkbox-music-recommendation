import torch
from torch import nn
import numpy as np
import pandas as pd
import random
import os.path as op
import scipy
import sklearn

from model.net import Net
from utils import NetworkData


learning_rate = 0.02
weight_decay = 5e-4
num_epochs = 10
# val_ratio = 0.2

data = NetworkData(cache=None, save=False)
adj, L, x_user, x_item, x_train_edge, x_test_edge, y_train = data.get_data()
data.describe()
# norm
x_user = x_user / x_user.sum(1, keepdims=True)
x_item = x_item / x_item.sum(1, keepdims=True)

# to tensor
x_user = torch.from_numpy(x_user)
x_item = torch.from_numpy(x_item)
y_train = torch.from_numpy(np.array(y_train))

num_user_nodes, user_input_dim = x_user.shape
num_item_nodes, item_input_dim = x_item.shape
hidden_dim = 500
embedding_dim = 500
edge_dim = len(x_train_edge['0-0'])
num_nodes = num_item_nodes + num_user_nodes

indices = torch.from_numpy(np.array([L.row,
                                    L.col]).astype("int64")).long()
values = torch.from_numpy(L.data.astype(np.float32))
adjacency = torch.sparse.FloatTensor(indices, values, (num_nodes, num_nodes))

model = Net(user_input_dim, item_input_dim, hidden_dim, embedding_dim, edge_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
criterion = nn.CrossEntropyLoss()

# train
def train():
    loss_history = []
    acc_history = []
    model.train()
    for epoch in range(num_epochs):

        logits = model(adjacency, x_user, x_item)
        loss = criterion(logits, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())
        acc = accuracy(logits, y_train)
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