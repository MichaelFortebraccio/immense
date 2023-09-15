import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import NeighborSampler
from torch_geometric.nn import SAGEConv
from tqdm import tqdm
from utils import load_from_pickle


class GraphSAGEEmbedder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, edg_dir, features, train_df, num_hidden_layers=3):
        """
        Class for creating a Pytorch graph and training an inductive graph embedding model with graphsage
        :param in_channels: Dimension of input features
        :param hidden_channels: Dimension of hidden channels
        :param out_channels: How many output channels
        :param edg_dir: Directory with the edge list
        :param features: Dictionary with the features for each node
        :param train_df: Dataframe used for retrieving each node's label
        :param num_layers:
        """
        super(GraphSAGEEmbedder, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels, normalize=True))
        for _ in range(num_hidden_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, normalize=True))
        self.convs.append(SAGEConv(hidden_channels, out_channels, normalize=True))
        self.edg_dir = edg_dir
        self.features = features
        self.train_df = train_df
        self.num_layers = num_hidden_layers + 1
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

    def create_graph(self, mapper, inv_map):
        inv_mapper_list = list(inv_map.keys())
        feats = []
        labs = []
        # Create the graph
        for i in inv_mapper_list:
            feats.append(self.features[int(i)])
            labs.append(self.train_df[self.train_df.id == i]['label'].values[0])
        x = torch.Tensor(np.array(feats))
        edgelist = []
        with open(self.edg_dir, 'r') as f:
            for l in f.readlines():
                e1, e2 = l.split("\t")
                edgelist.append((inv_map[int(e1)], inv_map[int(e2.strip())]))
        edges = list(zip(*edgelist))
        edge_index = torch.tensor(np.array(edges), dtype=torch.long)
        graph = Data(x=x, edge_index=edge_index, y=torch.tensor(labs))
        return graph

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adjs):
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

    def train_sage(self, train_loader, optimizer, x, y):
        self.train()
        total_loss = 0
        for batch_size, n_id, adjs in tqdm(train_loader):
            # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
            # batch_size = batch size; n_id = list of node ids in the current minibatch. These node ids make a subgraph.
            # adjs is a list containing two edge index elements. Each edge index has an edge_index attribute, which is a
            # tensor having two rows and a number of column that depends on how many edges are in current subgraph. If
            # there is an edge that goes from node A to B, then in the nth column, on the first row there will be the
            # index of node A's ID in the e_id list. On the second row there will be the index of node B. For instance,
            # suppose that n_id = [10, 31, 40, 54] and that there is an edge from 10 to 31, an edge from 40 to 10 and an
            # edge from 31 to 10, then the adj matrix will be:
            # [[0, 2, 1],
            #  [1, 0, 0]].
            # The edge index object also has the e_id attribute, a tensor with the ids of the edges in the subgraph.
            adjs = [adj.to(self.device) for adj in adjs]
            optimizer.zero_grad()
            feats = x[n_id]  # Node features for the current minibatch
            l1_emb, l2_emb, l3_emb = self(feats, adjs)
            out = l3_emb.log_softmax(dim=-1)
            loss = F.nll_loss(out, y[n_id[:batch_size]])
            loss.backward()
            optimizer.step()

            total_loss += float(loss)
            # pbar.update(batch_size)
        loss = total_loss / len(train_loader)
        return loss

    def inference(self, x_all, subgraph_loader):
        pbar = tqdm(total=x_all.size(0) * self.num_layers)
        pbar.set_description('Evaluating')
        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch.
        total_edges = 0
        for i in range(self.num_layers):
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

        pbar.close()
        return layer_2_embeddings

#df_dir = "dataset/tweets_labeled.csv"
#map_dir = "node_classification/graph_embeddings/stuff/sn_labeled_nodes_mapping.pkl"
#edg_dir = "node_classification/graph_embeddings/stuff/sn_labeled_nodes.edg"
"""if __name__ == "__main__":
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

