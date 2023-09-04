import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import NeighborSampler
from torch_geometric.nn import SAGEConv
from utils import load_from_pickle
import pandas as pd
import torch.nn.functional as F
from tqdm import tqdm

#df_dir = "dataset/tweets_labeled.csv"
#map_dir = "node_classification/graph_embeddings/stuff/sn_labeled_nodes_mapping.pkl"
#edg_dir = "node_classification/graph_embeddings/stuff/sn_labeled_nodes.edg"
df_dir = "../../dataset/tweets_labeled.csv"
map_dir = "stuff/sn_labeled_nodes_mapping.pkl"
edg_dir = "stuff/sn_labeled_nodes.edg"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
df = pd.read_csv(df_dir)
mapper = load_from_pickle(map_dir)
feats = []
labs = []
inv_map = {v: k for k, v in mapper.items()}
mapperk = list(inv_map.keys())
for i in mapperk:
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
train_loader = NeighborSampler(graph.edge_index, sizes=[1, 2], batch_size=3, shuffle=True)


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3):
        super(SAGE, self).__init__()

        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adjs):
        # `train_loader` computes the k-hop neighborhood of a batch of nodes,
        # and returns, for each layer, a bipartite graph object, holding the
        # bipartite edges `edge_index`, the index `e_id` of the original edges,
        # and the size/shape `size` of the bipartite graph.
        # Target nodes are also included in the source nodes so that one can
        # easily apply skip-connections or add self-loops.
        for i, (edge_index, _, size) in enumerate(adjs):
            xs = []
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
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
                # return x.log_softmax(dim=-1)
        return layer_1_embeddings, layer_2_embeddings, layer_3_embeddings

    def inference(self, x_all, subgraph_loader):
        pbar = tqdm(total=x_all.size(0) * self.num_layers)
        pbar.set_description('Evaluating')
        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch.
        total_edges = 0
        for i in range(self.num_layers):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                total_edges += edge_index.size(1)
                x = x_all[n_id].to(device)
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
        return layer_1_embeddings, layer_2_embeddings, layer_3_embeddings

x = graph.x.to(device)
y = graph.y.squeeze().to(device)
model = SAGE(in_channels=graph.num_features, hidden_channels=32, out_channels=2, num_layers=3)
model = model.to(device)
optimizer = torch.optim.Adam(lr=.003, params=model.parameters())


def train(epoch):
    model.train()
    total_loss = total_correct = 0
    for batch_size, n_id, adjs in train_loader:
        # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
        adjs = [adj.to(device) for adj in adjs]
        optimizer.zero_grad()
        feats = x[n_id]  # Node features for the current minibatch
        l1_emb, l2_emb, l3_emb = model(feats, adjs)
        # print("Layer 1 embeddings", l1_emb.shape)
        # print("Layer 2 embeddings", l1_emb.shape)
        out = l3_emb.log_softmax(dim=-1)
        loss = F.nll_loss(out, y[n_id[:batch_size]])
        loss.backward()
        optimizer.step()

        total_loss += float(loss)
        total_correct += int(out.argmax(dim=-1).eq(y[n_id[:batch_size]]).sum())
        # pbar.update(batch_size)
    loss = total_loss / len(train_loader)
    return loss


for i in range(10):
    loss = train(i)
    if i%5 == 0:
        print(loss)

print("sss")