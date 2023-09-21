import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import NeighborSampler
from tqdm import tqdm
from utils import load_from_pickle
from torch_geometric.nn import GraphSAGE
from torch_geometric.nn.norm import LayerNorm
class GraphSAGEEmbedder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, edg_dir, features, train_df, norm=False, num_layers=3):
        """
        Class for creating a Pytorch graph and training an inductive graph embedding model with graphsage
        :param in_channels: Dimension of input features
        :param hidden_channels: Dimension of hidden channels
        :param edg_dir: Directory with the edge list
        :param features: Dictionary with the features for each node
        :param train_df: Dataframe used for retrieving each node's label
        :param num_layers:
        """
        super(GraphSAGEEmbedder, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.norm = LayerNorm(in_channels=hidden_channels) if norm else None
        self.model = GraphSAGE(in_channels=in_channels, num_layers=num_layers, hidden_channels=hidden_channels,
                               norm=self.norm).to(self.device)
        self.edg_dir = edg_dir
        self.features = features
        self.train_df = train_df

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
        labs = []
        # Create the graph
        for i in inv_mapper_list:
            feats.append(self.features[int(i)])
            labs.append(self.train_df[self.train_df.id == i]['label'].values[0])
        x = torch.Tensor(np.array(feats))
        mean = x.mean(dim=0)
        std = x.std(dim=0)
        edgelist = []
        with open(self.edg_dir, 'r') as f:
            for l in f.readlines():
                e1, e2 = l.split("\t")
                edgelist.append((inv_map[int(e1)], inv_map[int(e2.strip())]))
        edges = list(zip(*edgelist))
        edge_index = torch.tensor(np.array(edges), dtype=torch.long)
        graph = Data(x=x, edge_index=edge_index, y=torch.tensor(labs))
        self.graph = graph
        return graph

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward_l(self, x, adjs):
        # `train_loader` computes the k-hop neighborhood of a batch of nodes and returns, for each layer, a bipartite
        # graph object, holding the bipartite edges `edge_index`, the index `e_id` of the original edges, and the
        # size/shape `size` of the bipartite graph. Target nodes are also included in the source nodes so that one can
        # easily apply skip-connections or add self-loops.
        for i, (edge_index, _, size) in enumerate(adjs):
            #TODO vedere cosa rappresenta size
            xs = []
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.2, training=self.training)
            xs.append(x)
            if i == 0:
                x_all = torch.cat(xs, dim=0)
                layer_1_embeddings = x_all
            elif i == 1:
                x_all = torch.cat(xs, dim=0)
                layer_2_embeddings = x_all
            elif i == 2:
                x_all = torch.cat(xs, dim=0)
                layer_3_embeddings = x_all
        return layer_1_embeddings, layer_2_embeddings, layer_3_embeddings

    def train_sage(self, train_loader, optimizer, mapper):
        self.model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(self.device)
            optimizer.zero_grad()
            h = self.model(batch.x, batch.edge_index)
            h_src = h[batch.edge_label_index[0]]
            h_dst = h[batch.edge_label_index[1]]
            pred = (h_src * h_dst).sum(dim=-1)
            loss = F.binary_cross_entropy_with_logits(pred, batch.edge_label)
            loss.backward()
            optimizer.step()
            total_loss += float(loss) * pred.size(0)
        return total_loss / self.graph.num_nodes

    def train_sage1(self, train_loader, optimizer, x, y):
        self.train()
        total_loss = 0
        for batch_size, n_id, adjs in tqdm(train_loader):
            adjs = [adj.to(self.device) for adj in adjs]
            optimizer.zero_grad()
            feats = x[n_id]  # Node features for the current minibatch
            l1_emb, l2_emb, l3_emb = self(feats, adjs)
            out = l3_emb.log_softmax(dim=-1)
            loss = F.nll_loss(out, y[n_id[:batch_size]])
            loss.backward()
            optimizer.step()

            total_loss += float(loss)
        loss = total_loss / len(train_loader)
        return loss

    def inference1(self, x_all, subgraph_loader):
        pbar = tqdm(total=x_all.size(0) * self.model.num_layers)
        pbar.set_description('Evaluating')
        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch.
        total_edges = 0
        for i in range(self.model.num_layers):
            xs = []
            for batch_size, n_id, adjs in subgraph_loader:
                for adj in adjs:
                    edge_index, _, size = adj.to(self.device)
                    total_edges += edge_index.size(1)
                    x = x_all[n_id].to(self.device)
                    x_target = x[:size[1]]
                    x = self.convs[i]((x, x_target), edge_index)
                if i != self.num_layers - 1:
                    x = F.relu(x)
                xs.append(x)
                pbar.update(batch_size)
            if i == 0:
                x_all = torch.cat(xs, dim=0)
                layer_1_embeddings = x_all
            elif i == 1:
                x_all = torch.cat(xs, dim=0)
                layer_2_embeddings = x_all
            elif i == 2:
                x_all = torch.cat(xs, dim=0)
                layer_3_embeddings = x_all

        pbar.close()
        return layer_2_embeddings, layer_3_embeddings

    def inference(self, x_all, subgraph_loader):
        pbar = tqdm(total=x_all.size(0) * self.model.num_layers)
        pbar.set_description('Evaluating')
        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch.
        total_edges = 0
        embs = []
        for batch in subgraph_loader:
            embs.append(self.model(batch.x, batch.edge_index))
        pbar.close()
        return embs

#df_dir = "dataset/tweets_labeled.csv"
#map_dir = "node_classification/graph_embeddings/stuff/sn_labeled_nodes_mapping.pkl"
#edg_dir = "node_classification/graph_embeddings/stuff/sn_labeled_nodes.edg"
"""
if __name__ == "__main__":
    df_dir = "../../dataset/tweets_labeled.csv"
    map_dir = "stuff/sn_labeled_nodes_mapping.pkl"
    edg_dir = "stuff/sn_labeled_nodes.edg"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    df = pd.read_csv(df_dir)
    mapper = load_from_pickle(map_dir)
    feats = []
    labs = []
    inv_map = {v: k for k, v in mapper.items()}
    inv_mapper_list = list(inv_map.keys())
    for i in inv_mapper_list:
        feats.append(np.random.rand(1, 5))
        labs.append(df[df.id == i]['label'].values[0])
    x = torch.Tensor(np.array(feats))
    edgelist = []
    with open(edg_dir, 'r') as f:
        for l in f.readlines():
            e1, e2 = l.split("\t")
            edgelist.append((inv_map[int(e1)], inv_map[int(e2.strip())]))
    edges = list(zip(*edgelist))
    edge_index = torch.tensor(np.array(edges), dtype=torch.long)
    graph = Data(x=x, edge_index=edge_index, y=torch.tensor(labs))
    x = graph.x.squeeze().to(device)
    y = graph.y.squeeze().to(device)
    sizes=[2, 2]
    train_loader = NeighborSampler(graph.edge_index, sizes=sizes, batch_size=2, shuffle=True)

    model = GraphSAGEEmbedder(in_channels=graph.num_features, hidden_channels=32, out_channels=2, num_layers=len(sizes),
                              edg_dir=edg_dir, )
    model = model.to(device)
    optimizer = torch.optim.Adam(lr=.003, params=model.parameters())
    for i in range(20):
        print(i)
        loss = train(train_loader, optimizer)
        if i%5 == 0:
            print(loss)
    model.eval()
    e1, e2 = model.inference(x, train_loader)
    e3, e4 = model.inference(x, train_loader)
    torch.save(model, "model.pt")"""

