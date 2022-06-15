import torch
from torch import nn
from torch.nn import functional as F

from net.GNN import GNN


class BPRNet(nn.Module):
    def __init__(self, n_user, n_item, input_dim, embedding_dim, batch_size, decay=0.1):
        super(BPRNet, self).__init__()
        self.n_user = n_user
        self.n_item = n_item
        self.decay = decay
        self.batch_size = batch_size
        self.project = nn.Linear(input_dim, embedding_dim, bias=True)
        self.gnn = GNN(embedding_dim)
        self.edge_linear = nn.Linear(embedding_dim * (self.gnn.num_layers + 1), 1)

    def forward(self, x_user, x_item, norm_adj, users, s_users, items, pos_items, neg_items):
        embeddings = torch.cat([x_user, x_item], 0)
        embeddings = self.project(embeddings)

        residual_embeddings = self.gnn(norm_adj, embeddings)

        all_user_embeddings = residual_embeddings[:self.n_user, :]
        all_item_embeddings = residual_embeddings[self.n_user:, :]

        user_embeddings = all_user_embeddings[users, :]
        item_embeddings = all_item_embeddings[items, :]
        s_user_embeddings = all_user_embeddings[s_users, :]
        pos_item_embeddings = all_item_embeddings[pos_items, :]
        neg_item_embeddings = all_item_embeddings[neg_items, :]

        return user_embeddings, s_user_embeddings, item_embeddings, pos_item_embeddings, neg_item_embeddings

    def loss(self, user_embeddings, s_user_embeddings, item_embeddings, pos_item_embeddings, neg_item_embeddings, y_edge):

        pos_scores = torch.sum(torch.mul(s_user_embeddings, pos_item_embeddings), axis=1)
        neg_scores = torch.sum(torch.mul(s_user_embeddings, neg_item_embeddings), axis=1)

        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -1 * torch.mean(maxi)

        regularizer = (torch.norm(user_embeddings) ** 2
                       + torch.norm(pos_item_embeddings) ** 2
                       + torch.norm(neg_item_embeddings) ** 2) / 2
        emb_loss = self.decay * regularizer / self.batch_size

        edge_pred = torch.mul(user_embeddings, item_embeddings)
        edge_pred = torch.sigmoid(self.edge_linear(edge_pred))

        edge_pred_loss = torch.mean((edge_pred - y_edge) ** 2)

        return mf_loss + emb_loss + edge_pred_loss, mf_loss + emb_loss, edge_pred_loss