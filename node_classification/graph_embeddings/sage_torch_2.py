import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import NeighborSampler, LinkNeighborLoader
from torch_geometric.nn import SAGEConv
from torchmetrics.classification import BinaryAccuracy
from tqdm import tqdm
from utils import load_from_pickle


class SAGE(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers, edg_dir, features, dropout=0.2):
        super(SAGE, self).__init__()
        self.device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
        self.dropout = dropout
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_dim, hidden_dim, normalize=True))
        self.num_layers = num_layers
        self.edg_dir = edg_dir
        self.features = features
        for _ in range(num_layers-1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim, normalize=True))

    def create_mappers(self):
        """
        Pytorch requires nodes to have IDs that start from 0. If it's not the case in the dataset, this function creates
        a mapper from the original IDs to progressive IDs suitable with Pytorch. The function also creates an inverse
        mapper, which is the same as the mapper but with keys and values are switched
        :return:
        """
        l = []
        with open(self.edg_dir, 'r') as f:
            for line in f.readlines():
                e1, e2 = line.split("\t")
                l.append(int(e1.strip()))
                l.append(int(e2.strip()))
        l = list(set(l))
        mapper = {}
        for i in range(len(l)):
            mapper[i] = l[i]
        inv_map = {v: k for k, v in mapper.items()}
        return mapper, inv_map

    def create_graph(self, inv_map):
        inv_mapper_list = list(inv_map.keys())
        feats = []
        # Create the graph
        for i in inv_mapper_list:
            feats.append(self.features[int(i)])
        x = torch.Tensor(np.array(feats))
        edgelist = []
        with open(self.edg_dir, 'r') as f:
            for l in f.readlines():
                e1, e2 = l.split("\t")
                edgelist.append((inv_map[int(e1)], inv_map[int(e2.strip())]))
        edges = list(zip(*edgelist))
        edge_index = torch.tensor(np.array(edges), dtype=torch.long)
        graph = Data(x=x, edge_index=edge_index)
        self.graph = graph
        return graph

    def create_loader(self, data, batch_size):
        return LinkNeighborLoader(data.edge_index, num_neighbors=[2, 2], batch_size=batch_size,
                                   neg_sampling_ratio=1.0, edge_label_index=self.graph.edge_index)

    def forward(self, batch):
        x = batch.x
        for i in range(len(self.convs)):
            x = self.convs[i](x, batch.edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout)
        return x

    def train_sage(self, train_loader, optimizer):
        self.train()
        total_loss = 0
        # Train on batches
        for batch in train_loader:
            optimizer.zero_grad()
            batch = batch.to(self.device)
            out = self(batch)
            out_src = out[batch.edge_label_index[0]]
            out_dst = out[batch.edge_label_index[1]]
            link_pred = (out_src * out_dst).sum(-1)
            loss = F.binary_cross_entropy_with_logits(link_pred, batch.edge_label)
            loss.backward()
            optimizer.step()
            total_loss += loss

            """out_val = self(batch.x[batch.val_mask], batch.edge_index[batch.val_mask])
            # Validation
            out_val_src = out_val[batch.edge_label_index[batch.val_mask][0]]
            out_val_dst = out_val[batch.edge_label_index[batch.val_mask][1]]
            link_pred_val = (out_val_src * out_val_dst).sum(-1)
            val_loss = F.binary_cross_entropy_with_logits(link_pred_val, batch.edge_label[batch.val_mask])"""
        return total_loss/len(train_loader)    #, val_loss

