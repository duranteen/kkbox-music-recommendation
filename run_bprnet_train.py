import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import time

from net.BPRNet import BPRNet
from utils import NetworkData


learning_rate = 0.001
weight_decay = 0.
num_epochs = 100
batch_size = 1024

data = NetworkData()
adj, L, x_user, x_item, x_train_edge, x_test_edge = data.get_data()

L = data.convert_sp_mat_to_sp_tensor(L)

train_loader = DataLoader(dataset=data, batch_size=batch_size, shuffle=True)

_, user_input_dim = x_user.shape
_, item_input_dim = x_item.shape
edge_dim = x_train_edge.shape[1] - 2
unified_feat_dim = max(user_input_dim, item_input_dim)
embedding_dim = 256

# to tensor
x_user = torch.from_numpy(np.float32(x_user))
x_item = torch.from_numpy(np.float32(x_item))
x_user = F.normalize(x_user, dim=1)
x_item = F.normalize(x_item, dim=1)
if isinstance(x_user, torch.Tensor):
    x_user = torch.cat([x_user, torch.zeros((x_user.shape[0], unified_feat_dim-x_user.shape[1]))], 1)

n_users, n_items = data.n_users, data.n_items

model = BPRNet(n_users, n_items, unified_feat_dim, embedding_dim, batch_size)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
# criterion = nn.CrossEntropyLoss()


loss_history = []
for epoch in tqdm(range(num_epochs)):
    loss_epoch = []
    for users_id, items_id, y in tqdm(train_loader):
        sampling_users, pos_items, neg_items = data.sampling(users_id)
        user_embeddings, s_user_embeddings, item_embeddings, pos_item_embeddings, neg_item_embeddings = \
            model(x_user, x_item, L, users_id, sampling_users, items_id, pos_items, neg_items)
        loss, bpr_loss, ep_loss = \
            model.loss(user_embeddings, s_user_embeddings, item_embeddings, pos_item_embeddings, neg_item_embeddings, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_epoch.append(loss.item())

        print("\nEpoch {:02d}: Loss {:.4f}, BPR Loss {:.4f}, Edge Prediction Loss {:.4f}".format(
            epoch, loss.item(), bpr_loss.item(), ep_loss.item()
        ))
    loss_history.append(sum(loss_epoch) / len(loss_epoch))


