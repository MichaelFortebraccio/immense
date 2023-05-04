from node_classification.graph_embeddings.node2vec import Node2VecEmbedder
from modelling.ae import AE
from sklearn.decomposition import PCA
import pickle
from os.path import exists

# Transductive
def dimensionality_reduction(node_emb_technique: str, model_dir, train_df, node_embedding_size, lab, edge_path=None, n_of_walks=None,
                             walk_length=None, p=None, q=None, n2v_epochs=None, adj_matrix=None, id2idx=None,
                             epochs=None):
    """
    This function applies one of the node dimensionality reduction techniques in order to generate the feature vectors that will be used for training
    the decision tree.
    Args:
        node_emb_technique: Can be either "node2vec", "pca", "autoencoder" or "none" (uses the whole adjacency matrix rows as feature vectors)
        model_dir: Directory where the models will be saved
        train_df: Dataframe with the training data. The IDs will be used
        node_embedding_size: Dimension of the embeddings to create
        lab: Label, can be either "spat" or "rel"
        edge_path: Path to the list of edges used by node2vec. Ignored if node_emb_technique != 'node2vec'
        n_of_walks: Number of walks that the n2v model will do. Ignored if node_emb_technique != 'node2vec'
        walk_length: Length of the walks that the n2v model will do. Ignored if node_emb_technique != 'node2vec'
        p: n2v's hyperparameter p. Ignored if node_emb_technique != 'node2vec'
        q: n2v's hyperparameter q. Ignored if node_emb_technique != 'node2vec'
        n2v_epochs: for how many epochs the n2v model will be trained. Ignored if node_emb_technique != 'node2vec'
        weighted: whether the edges are weighted. Ignored if node_emb_technique != 'node2vec'
        directed: Whether the edges are directed. Ignored if node_emb_technique != 'node2vec'
        adj_matrix: Adjacency matrix. Used only if node_emb_technique in ["pca", "autoencoder", "none"]
        id2idx: Mapping between the node IDs and the rows in the adj matrix. If you are using a technique different
        from node2vec, and the user IDs are not the index of the position of the users into the adjacency matrix,
        this parameter must be set. Otherwise, it can be left to none
        epochs: Epochs for training the autoencoder, if it is chosen as embedding technique. If another technique is
        chosen, this parameter can be ignored
    Returns:
        train_set: Array containing the node embeddings, which will be used for training the decision tree
        train_set_labels: Labels of the training vectors
    """
    node_emb_technique = node_emb_technique.lower()
    if node_emb_technique == "node2vec":
        n2v_path = "{}/n2v_{}.h5".format(model_dir, lab)
        weighted = False
        directed = True
        if lab == "spat":
            weighted = True
            directed = False
        n2v = Node2VecEmbedder(path_to_edges=edge_path, weighted=weighted, directed=directed, n_of_walks=n_of_walks,
                               walk_length=walk_length, embedding_size=node_embedding_size, p=p, q=q,
                               epochs=n2v_epochs, model_path=n2v_path).learn_n2v_embeddings(l=lab)
        mod = n2v.wv
        train_set_ids = [i for i in train_df['id'] if str(i) in mod.index_to_key]      # we use this cicle so to keep the order of the users as they appear in the df. The same applies for the next line
        train_set = [mod.vectors[mod.key_to_index[str(i)]] for i in train_set_ids]
        train_set_labels = train_df[train_df['id'].isin(train_set_ids)]['label']
    else:
        if node_emb_technique == "pca":
            if not exists("{}/pca_{}.pkl".format(model_dir, lab)):
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
