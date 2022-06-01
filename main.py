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
num_epochs = 5
# val_ratio = 0.2

data = NetworkData()
adj, L, x_user, x_item, x_train_edge, x_test_edge = data.get_data()
# norm
x_user = x_user / x_user.sum(1, keepdims=True)
x_item = x_item / x_item.sum(1, keepdims=True)

user_input_dim = x_user.shape[1]
item_input_dim = x_item.shape[1]
hidden_dim = 500
embedding_dim = 500
edge_dim = len(x_train_edge['0-0'])

model = Net(user_input_dim, item_input_dim, hidden_dim, embedding_dim, edge_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
criterion = nn.CrossEntropyLoss()

# train



# test


# plot
