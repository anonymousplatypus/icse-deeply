import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import GraphConvolution


class Encoder(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, dropout):
        super(Encoder, self).__init__()

        self.dropout = dropout

        self.encgc1 = GraphConvolution(input_feat_dim, hidden_dim1, dropout=dropout)
        self.encgc2 = GraphConvolution(hidden_dim1, hidden_dim2, dropout=dropout)

        self.decgc1 = GraphConvolution(hidden_dim2, hidden_dim1, dropout=dropout)
        self.decgc2 = GraphConvolution(hidden_dim1, input_feat_dim, dropout=dropout, act=lambda x: x)

    def encode(self, x, adj):
        hidden1 = self.encgc1(x, adj)
        hidden2 = self.encgc2(hidden1, adj)
        return hidden2

    def decode(self, x, adj):
        return self.decgc2(self.decgc1(x, adj), adj)

    def forward(self, x, adj):
        enc = self.encode(x, adj)
        dec = self.decode(enc, adj)
        return dec, enc
