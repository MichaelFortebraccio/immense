import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch_geometric.data import Data
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import SAGEConv, GraphConv
from tqdm import tqdm
torch.manual_seed(42)


def create_loader(data, batch_size, num_neighbors):
    return LinkNeighborLoader(data, num_neighbors=num_neighbors, batch_size=batch_size, neg_sampling_ratio=1.0,
                              edge_label_index=data.edge_index)


def create_mappers(features_dict):
    """
    Pytorch requires nodes to have IDs that start from 0. If it's not the case in the dataset, this function creates
    a mapper from the original IDs to progressive IDs suitable with Pytorch. The function also creates an inverse
    mapper, which is the same as the mapper but with keys and values are switched
    :return:
    """
    mapper = {}
    for i, k in enumerate(features_dict):
        mapper[i] = k
    inv_map = {v: k for k, v in mapper.items()}
    return mapper, inv_map


def create_graph(inv_map, weighted, features, edg_dir):
    inv_mapper_list = list(inv_map.keys())
    feats = []
    # Create the graph
    for i in inv_mapper_list:
        feats.append(features[int(i)])
    x = torch.Tensor(np.array(feats))
    edgelist = []
    weights = []
    with open(edg_dir, 'r') as f:
        for l in f.readlines():
            split = l.split("\t")
            e1, e2 = split[0].strip(), split[1].strip()
            if len(split) == 3:
                e3 = split[2].strip()
                weights.append(float(e3))
            edgelist.append((inv_map[int(e1)], inv_map[int(e2.strip())]))
    edges = list(zip(*edgelist))
    edge_index = torch.tensor(np.array(edges), dtype=torch.long)
    if weighted:
        edge_attr = torch.tensor(np.array(weights), dtype=torch.float)
        graph = Data(x=x, edge_index=edge_index, edge_weight=edge_attr)
    else:
        graph = Data(x=x, edge_index=edge_index)
    return graph


class SAGE(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers, weighted, directed):
        super(SAGE, self).__init__()
        self.device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
        self.convs = torch.nn.ModuleList()
        self.num_layers = num_layers
        self.weighted = weighted
        self.directed = directed
        if self.weighted:
            self.convs.append(GraphConv(in_dim, hidden_dim, aggr="mean", normalize=True))
            for _ in range(num_layers-1):
                self.convs.append(GraphConv(hidden_dim, hidden_dim, aggr="mean", normalize=True))
        else:
            self.convs.append(SAGEConv(in_dim, hidden_dim, aggr="mean", normalize=True))
            for _ in range(num_layers-1):
                self.convs.append(SAGEConv(hidden_dim, hidden_dim, aggr="mean", normalize=True))

    def forward(self, batch):
        x = batch.x
        for i in range(len(self.convs)):
            if self.weighted:
                x = self.convs[i](x, batch.edge_index, batch.edge_weight)
            else:
                x = self.convs[i](x, batch.edge_index)
            x = F.relu(x)
        return x

    def train_sage(self, train_loader, optimizer, mapper):
        self.train()
        total_loss = 0
        # Train on batches
        for batch in tqdm(train_loader):
            optimizer.zero_grad()
            batch = batch.to(self.device)
            out = self(batch)
            out_src = out[batch.edge_label_index[0]]
            out_dst = out[batch.edge_label_index[1]]
            link_pred = (out_src * out_dst).sum(-1)
            loss = F.binary_cross_entropy_with_logits(link_pred, batch.edge_label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss/len(train_loader)

    @torch.no_grad()
    def test(self, data):
        data = data.to(self.device)
        self.eval()
        out_val = self(data)
        out_val_src = out_val[data.edge_label_index[0]]
        out_val_dst = out_val[data.edge_label_index[1]]
        link_pred = (out_val_src * out_val_dst).sum(-1)
        return roc_auc_score(data.edge_label.cpu().numpy(), link_pred.cpu().numpy())
