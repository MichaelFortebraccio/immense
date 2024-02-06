import numpy as np
from os.path import join, exists
import pickle
import torch
from tqdm import tqdm
import torch_geometric.transforms as T
from modelling.ae import AE
from node_classification.graph_embeddings.node2vec import Node2VecEmbedder
from node_classification.graph_embeddings.sage import SAGE, create_mappers, create_graph
from sklearn.decomposition import PCA
from torch_geometric.loader import NeighborLoader
from utils import is_square, embeddings_pca, load_from_pickle, save_to_pickle


def reduce_dimension(emb_technique: str, lab, model_dir, ne_dim, train_df, we_dim, adj_matrix_path=None,
                     batch_size=None, edge_path=None, epochs=None, features_dict=None, id2idx_path=None,
                     n_of_walks=10, p=1, q=4, sizes=None, walk_length=10, training_weights=None):
    """
    This function applies one of the node dimensionality reduction techniques and generate the feature vectors for
    training the decision tree.
    Args:
        :param emb_technique: Can be either "node2vec", "graphsage", "pca", "autoencoder" or "none"
        (uses the whole adjacency matrix rows as feature vectors)
        :param lab: Label, can be either "spat" or "rel".
        :param model_dir: Directory where the models will be saved.
        :param ne_dim: Dimension of the embeddings to create.
        :param train_df: Dataframe with the training data. The IDs will be used.
        :param adj_matrix_path: (pca, autoencoder, none) Adjacency matrix.
        :param batch_size: (graphsage) Batch size to use during training.
        :param edge_path: (graphsage, node2vec) Path to the list of edges used by the node embedding technique
        :param epochs: (graphsage, node2vec) Epochs for training the node embedding model.
        :param features_dict: (graphsage) Dictionary having as keys the IDs of the users and as values the sum of the
        embeddings of their posts.
        :param id2idx_path: (pca, autoencoder, none) Mapping between the node IDs and the rows in the adj matrix.
        :param n_of_walks: (node2vec) Number of walks that the n2v model will do.
        :param p: (node2vec) n2v's hyperparameter p.
        :param q: (node2vec) n2v's hyperparameter q.
        :param sizes: (graphsage) Array containing the number of neighbors to sample for each node.
        :param walk_length: (node2vec) Length of the walks that the n2v model will do.
        :param training_weights: tensor of shape (1, num_classes) containing the weights to give to each class while
        training the graphsage model. If None, no weights will be used
    Returns:
        train_set: Array containing the node embeddings, which will be used for training the decision tree.
        train_set_labels: Labels of the training vectors.
    """
    train_set = []
    train_set_labels = []
    emb_technique = emb_technique.lower()
    if emb_technique == "node2vec":
        model_path = join(model_dir, "n2v.h5")
        weighted = False
        directed = True
        if lab == "spat":
            weighted = True
            directed = False
        n2v = Node2VecEmbedder(path_to_edges=edge_path, weighted=weighted, directed=directed, n_of_walks=n_of_walks,
                               walk_length=walk_length, embedding_size=ne_dim, p=p, q=q,
                               epochs=epochs, model_path=model_path).learn_n2v_embeddings()
        embeddings_pca(n2v, "node2vec", dst_dir=model_dir)
        mod = n2v.wv
        train_set_ids = [i for i in train_df['id'] if str(i) in mod.index_to_key]  # we use this cicle so to keep the order of the users as they appear in the df. The same applies for the next line
        for i in train_set_ids:
            train_set.append(mod[str(i)])
            train_set_labels.append(train_df[train_df.id == i]['label'].values[0])
    elif emb_technique == "graphsage":
        weights_path = join(model_dir, "graphsage_{}_{}.h5".format(ne_dim, we_dim))
        model_path = join(model_dir, "graphsage_{}_{}.pkl".format(ne_dim, we_dim))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        first_key = list(features_dict.keys())[0]
        in_channels = len(features_dict[first_key])
        weighted = False
        directed = True
        if lab == "spat":
            weighted = True
            directed = False

        mapper_train, inv_map_train = create_mappers(features_dict)
        graph = create_graph(inv_map=inv_map_train, weighted=weighted, features=features_dict, edg_dir=edge_path, df=train_df)
        split = T.RandomLinkSplit(num_val=0.1, num_test=0.0, is_undirected=not directed,
                                  add_negative_train_samples=False, neg_sampling_ratio=1.0,)
        train_data, valid_data, _ = split(graph)
        sage = SAGE(in_dim=in_channels, hidden_dim=ne_dim, num_layers=len(sizes), weighted=weighted,
                    directed=directed)
        sage = sage.to(device)
        train_loader = NeighborLoader(train_data, num_neighbors=sizes, batch_size=batch_size)
        if not exists(model_path):
            print("Training {} node embedding model\n".format(lab))
            optimizer = torch.optim.Adam(lr=.01, params=sage.parameters(), weight_decay=1e-4)
            best_loss = 9999
            for i in range(epochs):
                loss = sage.train_sage(train_loader, optimizer=optimizer, weights=training_weights)
                #val_loss = sage.test()
                val_loss = 0
                if loss < best_loss:
                    best_loss = loss
                    print("New best model found at epoch {}. Loss: {}, val_loss: {}".format(i, loss, val_loss))
                    torch.save(sage.state_dict(), weights_path)
                if i % 5 == 0:
                    print("Epoch {}: train loss {}, val loss: {}".format(i, loss, val_loss))
            sage.load_state_dict(torch.load(weights_path))
            save_to_pickle(model_path, sage)
        else:
            sage = load_from_pickle(model_path)
        train_set = sage(graph, inference=True)
        train_set = train_set.detach().numpy()
        for k in features_dict:
            train_set_labels.append(train_df[train_df.id == k]['label'].values[0])
    else:
        adj_matrix = np.genfromtxt(adj_matrix_path, delimiter=',')
        if not is_square(adj_matrix):
            raise Exception("The {} adjacency matrix is not square".format(lab))
        id2idx = load_from_pickle(id2idx_path)
        if emb_technique == "pca":
            if not exists("{}/pca_{}.pkl".format(model_dir, lab)):
                print("Learning PCA")
                pca = PCA(n_components=ne_dim)
                pca.fit(adj_matrix)
                with open("{}/pca_{}.pkl".format(model_dir, lab), 'wb') as f:
                    pickle.dump(pca, f)
            else:
                with open("{}/pca_{}.pkl".format(model_dir, lab), 'rb') as f:
                    pca = pickle.load(f)
            train_set = pca.transform(adj_matrix)
        elif emb_technique == "autoencoder":
            ae = AE(X_train=adj_matrix, name="encoder_{}".format(lab), model_dir=model_dir, epochs=epochs, batch_size=128, lr=0.05).train_autoencoder_node(ne_dim)
            train_set = ae.predict(adj_matrix)
        elif emb_technique == "none":
            train_set = adj_matrix
        train_set_labels = [train_df[train_df['id'] == k]['label'].values[0] for k in id2idx.keys()]
    return train_set, train_set_labels
