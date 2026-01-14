import os
import argparse
import pickle
import math
import copy
import itertools
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ======================================================
# Argument Parser (config.py 대체)
# ======================================================

def parse_args():
    p = argparse.ArgumentParser("T-MPHN single-file runner")

    p.add_argument('--data_type', type=str, required=True)
    p.add_argument('--dataset', type=str, required=True)

    p.add_argument('--num_layers', type=int, default=1)
    p.add_argument('--hid_dim', type=int, default=256)
    p.add_argument('--combine', type=str, default='concat')
    p.add_argument('--Mlst', type=list, default=[3])

    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--wd', type=float, default=0.0005)
    p.add_argument('--train_ratio', type=float, default=0.5)
    p.add_argument('--valid_ratio', type=float, default=0.25)

    p.add_argument('--cuda', type=int, default=0)
    p.add_argument('--seed', type=int, default=42)

    return p.parse_args()


# ======================================================
# Data Loader (prepare.py 대체)
# ======================================================

def read_data(base_dir, data_type, dataset):
    path = os.path.join(base_dir, data_type, dataset)

    with open(os.path.join(path, 'hypergraph.pickle'), 'rb') as f:
        H = pickle.load(f)

    with open(os.path.join(path, 'feature.pickle'), 'rb') as f:
        X = pickle.load(f)

    with open(os.path.join(path, 'labels.pickle'), 'rb') as f:
        Y = pickle.load(f)

    return H, X, Y


def split_indices(n, train_ratio, valid_ratio):
    perm = torch.randperm(n)
    t = int(n * train_ratio)
    v = int(n * valid_ratio)
    return perm[:t], perm[t:t+v], perm[t+v:]


# ======================================================
# Neighbor Finder (utils/Neighbors.py 대체)
# ======================================================

class NeighborFinder:
    def __init__(self, H):
        self.H = H

    def neig_for_targets(self, nodes):
        neig_dict = {}
        for i in nodes:
            neig_dict[i] = [
                list(np.where(self.H[:, e] > 0)[0])
                for e in range(self.H.shape[1]) if self.H[i, e] > 0
            ]
        return neig_dict


# ======================================================
# TMessagePassing
# ======================================================

class TMessagePassing(nn.Module):
    def __init__(self, features, structure, m, args):
        super().__init__()
        self.features = features
        self.structure = structure
        self.m = m
        self.args = args

        self.w_att = nn.Linear(1, 1)
        self.att_act = nn.Sigmoid()

        self.device = torch.device(
            f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu'
        )

    def forward(self, nodes):
        return torch.stack(
            [self.aggregate_one(n) for n in nodes], dim=0
        )

    def aggregate_one(self, node):
        edges = self.structure[int(node)]
        self_feat = self.features(
            torch.LongTensor([node]).to(self.device)
        ).squeeze()

        if not edges:
            return self_feat

        out = torch.zeros_like(self_feat)

        for edge in edges:
            feats = self.features(torch.LongTensor(edge).to(self.device))
            edge_var = feats.var(dim=0, unbiased=False).mean().view(1, 1)
            att = self.att_act(self.w_att(edge_var)).squeeze()

            edge_mean = feats.mean(dim=0)

            if len(edge) > self.m:
                msg = edge_mean
            elif len(edge) == self.m:
                msg = torch.prod(feats[:-1], dim=0)
            else:
                msg = edge_mean

            out += att * msg

        return out


# ======================================================
# Encoder
# ======================================================

class Encoder(nn.Module):
    def __init__(self, features, in_dim, out_dim, args, aggregator):
        super().__init__()
        self.features = features
        self.aggregator = aggregator
        self.beta = nn.Parameter(torch.tensor(0.5))

        if args.combine == 'concat':
            self.lin = nn.Linear(in_dim * 2, out_dim)
        else:
            self.lin = nn.Linear(in_dim, out_dim)

        self.skip = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, nodes):
        neigh = self.aggregator(nodes)
        self_feat = self.features(
            torch.LongTensor(nodes).to(neigh.device)
        )

        if self.lin.in_features == self_feat.size(1) * 2:
            h = torch.cat([self_feat, neigh], dim=1)
        else:
            h = self_feat + neigh

        out = F.relu(self.lin(h))
        return (1 - self.beta) * out + self.beta * self.skip(self_feat)


# ======================================================
# TMPHN Model
# ======================================================

class TMPHN(nn.Module):
    def __init__(self, X, neig_dict, args):
        super().__init__()
        self.device = torch.device(
            f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu'
        )

        self.features = nn.Embedding(X.shape[0], X.shape[1])
        self.features.weight.data = torch.FloatTensor(X)
        self.features.weight.requires_grad = False

        m = args.Mlst[0]
        agg = TMessagePassing(self.features, neig_dict, m, args)
        self.encoder = Encoder(self.features, X.shape[1], args.hid_dim, args, agg)
        self.classifier = nn.Linear(args.hid_dim, args.num_classes)

    def forward(self, nodes):
        h = self.encoder(nodes)
        return F.log_softmax(self.classifier(h), dim=1)


# ======================================================
# Evaluation
# ======================================================

@torch.no_grad()
def accuracy(model, Y, idx):
    out = model(idx)
    pred = out.argmax(dim=1)
    return (pred == Y[idx]).float().mean().item()


# ======================================================
# Main
# ======================================================

def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(
        f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu'
    )

    H, X, Y = read_data('./dataset', args.data_type, args.dataset)

    X = torch.FloatTensor(X)
    Y = torch.LongTensor(Y).to(device)
    args.num_classes = len(torch.unique(Y))

    neig_dict = NeighborFinder(H).neig_for_targets(range(X.shape[0]))

    model = TMPHN(X, neig_dict, args).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.wd
    )

    train_idx, val_idx, test_idx = split_indices(
        len(Y), args.train_ratio, args.valid_ratio
    )

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()

        loss = F.nll_loss(model(train_idx), Y[train_idx])
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            tr = accuracy(model, Y, train_idx)
            va = accuracy(model, Y, val_idx)
            te = accuracy(model, Y, test_idx)
            print(f"[epoch={epoch:03d}] loss={loss:.4f} train={tr:.3f} val={va:.3f} test={te:.3f}")


if __name__ == "__main__":
    main()
