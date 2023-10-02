import numpy as np
import os
import pickle
import torch
from modelling.ae import AE
from node_classification.graph_embeddings.node2vec import Node2VecEmbedder
from node_classification.graph_embeddings.sage_torch_1 import GraphSAGEEmbedder
from node_classification.graph_embeddings.sage_torch_2 import SAGE
from sklearn.decomposition import PCA
from torch_geometric.loader import NeighborSampler, LinkNeighborLoader
from torch_geometric.sampler import NegativeSampling
from torch_geometric.nn import GraphSAGE
from utils import is_square, load_from_pickle, save_to_pickle


def dimensionality_reduction(node_emb_technique: str, model_dir, train_df, node_embedding_size, lab, edge_path=None,
                             n_of_walks=None, walk_length=None, p=None, q=None, n2v_epochs=None, adj_matrix_path=None,
                             id2idx_path=None, epochs=None, features_dict=None):
    """
    This function applies one of the node dimensionality reduction techniques in order to generate the feature vectors that will be used for training
    the decision tree.
    Args:
        :param node_emb_technique: Can be either "node2vec", "pca", "autoencoder" or "none" (uses the whole adjacency
            matrix rows as feature vectors)
        :param model_dir: Directory where the models will be saved
        :param train_df: Dataframe with the training data. The IDs will be used
        :param node_embedding_size: Dimension of the embeddings to create
        :param lab: Label, can be either "spat" or "rel"
        :param edge_path: Path to the list of edges used by node2vec. Ignored if node_emb_technique != 'node2vec'
        :param n_of_walks: Number of walks that the n2v model will do. Ignored if node_emb_technique != 'node2vec'
        :param walk_length: Length of the walks that the n2v model will do. Ignored if node_emb_technique != 'node2vec'
        :param p: n2v's hyperparameter p. Ignored if node_emb_technique != 'node2vec'
        :param q: n2v's hyperparameter q. Ignored if node_emb_technique != 'node2vec'
        :param n2v_epochs: for how many epochs the n2v model will be trained. Ignored if node_emb_technique != 'node2vec'
        :param adj_matrix_path: Adjacency matrix. Used only if node_emb_technique in ["pca", "autoencoder", "none"]
        :param id2idx_path: Mapping between the node IDs and the rows in the adj matrix. If you are using a technique
            different from node2vec, and the user IDs are not the index of the position of the users into the adjacency
            matrix, this parameter must be set. Otherwise, it can be left to none
        :param epochs: Epochs for training the autoencoder, if it is chosen as embedding technique. If another technique is
            chosen, this parameter can be ignored
        :param features_dict: dictionary having as keys the IDs of the users and as values the sum of the embeddings of
            their posts. Used only if node_emb_technique == "graphsage"
    Returns:
        train_set: Array containing the node embeddings, which will be used for training the decision tree
        train_set_labels: Labels of the training vectors
    """
    node_emb_technique = node_emb_technique.lower()
    if node_emb_technique == "node2vec":
        model_path = os.path.join(model_dir, "n2v.h5")
        weighted = False
        directed = True
        if lab == "spat":
            weighted = True
            directed = False
        n2v = Node2VecEmbedder(path_to_edges=edge_path, weighted=weighted, directed=directed, n_of_walks=n_of_walks,
                               walk_length=walk_length, embedding_size=node_embedding_size, p=p, q=q,
                               epochs=n2v_epochs, model_path=model_path).learn_n2v_embeddings()
        mod = n2v.wv
        train_set_ids = [i for i in train_df['id'] if str(i) in mod.index_to_key]  # we use this cicle so to keep the order of the users as they appear in the df. The same applies for the next line
        train_set = []
        train_set_labels = []
        for i in train_set_ids:
            train_set.append(mod[str(i)])
            train_set_labels.append(train_df[train_df.id == i]['label'].values[0])
    elif node_emb_technique == "graphsage":
        sizes = [15, 10]
        batch_size = 64
        model_path = os.path.join(model_dir, "graphsage_{}.h5".format(lab))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        map_dir = "stuff/sn_labeled_nodes_mapping.pkl"

        first_key = list(features_dict.keys())[0]
        in_channels = len(
            features_dict[first_key])  # Take a random element in features dict in order to know the features dimension
        # sage = GraphSAGEEmbedder(in_channels=in_channels, hidden_channels=node_embedding_size,
        #                         num_layers=len(sizes), edg_dir=edge_path, train_df=train_df, features=features_dict)
        weighted = False
        directed = True
        if lab == "spat":
            weighted = True
            directed = False
        sage = SAGE(in_dim=in_channels, hidden_dim=node_embedding_size, num_layers=len(sizes), edg_dir=edge_path,
                    features=features_dict, weighted=weighted, directed=directed)
        mapper, inv_map = sage.create_mappers()
        graph = sage.create_graph(inv_map, weighted=weighted)
        sage = sage.to(device)

        train_loader = LinkNeighborLoader(graph, num_neighbors=sizes, batch_size=batch_size,
                                          neg_sampling_ratio=1.0, edge_label_index=graph.edge_index)

        optimizer = torch.optim.Adam(lr=.03, params=sage.parameters(), weight_decay=1e-4)
        #TODO PROVARE ALTRI OPTIMIZER
        for i in range(15):
            loss = sage.train_sage(train_loader, optimizer=optimizer)
            # loss = sage.train_sage(train_loader, optimizer=optimizer, mapper=mapper)
            # loss = sage.train_sage(train_loader, optimizer=optimizer, x=x, y=y)
            # train(train_loader, optimizer=optimizer, model=sage, device=device)
            print(loss)
        # sage.eval()
        test_loader = LinkNeighborLoader(graph, num_neighbors=sizes, batch_size=batch_size,
                                         neg_sampling_ratio=1.0, edge_label_index=graph.edge_index)
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(sage.device)
                e2 = sage(batch)
                e3 = sage(batch)
        torch.save(sage, model_path)
        graph = sage.create_graph(inv_map, weighted=weighted)

    else:
        adj_matrix = np.genfromtxt(adj_matrix_path, delimiter=',')
        if not is_square(adj_matrix):
            raise Exception("The {} adjacency matrix is not square".format(lab))
        id2idx = load_from_pickle(id2idx_path)
        if node_emb_technique == "pca":
            if not os.path.exists("{}/pca_{}.pkl".format(model_dir, lab)):
                print("Learning PCA")
                pca = PCA(n_components=node_embedding_size)
                pca.fit(adj_matrix)
                with open("{}/pca_{}.pkl".format(model_dir, lab), 'wb') as f:
                    pickle.dump(pca, f)
            else:
                with open("{}/pca_{}.pkl".format(model_dir, lab), 'rb') as f:
                    pca = pickle.load(f)
            train_set = pca.transform(adj_matrix)
        elif node_emb_technique == "autoencoder":
            ae = AE(X_train=adj_matrix, name="encoder_{}".format(lab), model_dir=model_dir, epochs=epochs, batch_size=128, lr=0.05).train_autoencoder_node(node_embedding_size)
            train_set = ae.predict(adj_matrix)
        elif node_emb_technique == "none":
            train_set = adj_matrix
        train_set_labels = [train_df[train_df['id'] == k]['label'].values[0] for k in id2idx.keys()]
    return train_set, train_set_labels
