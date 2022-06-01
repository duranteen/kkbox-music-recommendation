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
num_epochs = 5
# val_ratio = 0.2

data = NetworkData()
adj, L, x_user, x_item, x_train_edge, x_test_edge = data.get_data()

# 将特征与对应节点对应好，用于输入
# 接下来定义各维度特征




# model = Net()
optimizer = torch.optim.Adam()
criterion = nn.CrossEntropyLoss()


# train



# test


# plot
