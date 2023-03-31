from node_classification.graph_embeddings.node2vec import Node2VecEmbedder
from sklearn.decomposition import PCA
import pickle

def pca(adj_matrix, components):
    pca = PCA(n_components=0.95)
    pca.fit(adj_matrix)
    return pca.transform(adj_matrix)

# Transductive
def dimensionality_reduction(node_emb_technique, model_dir, train_df, node_embedding_size, edge_path=None, n_of_walks=None, walk_length=None, p=None, q=None,
                             n2v_epochs=None, adj_matrix=None, id_to_idx=None):
    """
    This function applies one of the node dimensionality reduction techniques in order to generate the feature vectors that will be used for training
    the decision tree.
    Args:
        node_emb_technique: Can be either "node2vec", "pca", "autoencoder" or "none" (uses the whole adjacency matrix rows as feature vectors)
        model_dir: Directory where the models will be saved
        train_df: Dataframe with the training data. The IDs will be used
        node_embedding_size: Dimension of the embeddings to create
        edge_path: Path to the list of edges used by node2vec. Ignored if node_emb_technique != 'node2vec'
        n_of_walks: Number of walks that the n2v model will do. Ignored if node_emb_technique != 'node2vec'
        walk_length: Length of the walks that the n2v model will do. Ignored if node_emb_technique != 'node2vec'
        p: n2v's hyperparameter p. Ignored if node_emb_technique != 'node2vec'
        q: n2v's hyperparameter q. Ignored if node_emb_technique != 'node2vec'
        n2v_epochs: for how many epochs the n2v model will be trained. Ignored if node_emb_technique != 'node2vec'
        adj_matrix: Adjacency matrix. Used only if node_emb_technique in ["pca", "autoencoder", "none"]
        id_to_idx: Mapping between the node IDs and the rows in the adj matrix. If you are using a technique that is
         different from node2vec, and the user IDs are not the index of the position of the users into the adjacency matrix,
         this parameter must be set. Otherwise, it can be left as none
        the ID is not its index in the adj matrix. Can be

    Returns:

    """
    if node_emb_technique == "node2vec":
        n2v_path = "{}/n2v_rel.h5".format(model_dir)
        n2v = Node2VecEmbedder(path_to_edges=edge_path, weighted=False, directed=True, n_of_walks=n_of_walks,
                               walk_length=walk_length, embedding_size=node_embedding_size, p=p, q=q,
                               epochs=n2v_epochs, model_path=n2v_path).learn_n2v_embeddings()
        mod = n2v.wv
        train_set_ids = [i for i in train_df['id'] if str(i) in mod.key_to_index]
        train_set = [mod.vectors[mod.key_to_index[str(i)]] for i in train_set_ids]
        train_set_labels = train_df[train_df['id'].isin(train_set_ids)]['label']
    elif node_emb_technique == "PCA":
        if id_to_idx:
            train_ids = train_df['id'].to_list()
            train_indexes = [id_to_idx[id] for id in train_ids]
            # Ricontrollare
        '''else:
            train_indexes = train_df['id'].to_list()
            train_indexes = [int(idx) for idx in train_indexes]
        if inductive:
            train_mat = adj_matrix[train_indexes][:, train_indexes]
        else:
            train_mat = adj_matrix'''
        pca = PCA(n_components=node_embedding_size)
        pca.fit(adj_matrix)
        with open("{}/pca.pkl".format(model_dir), 'wb') as f:
            pickle.dump(pca, f)
        train_set = pca.transform(adj_matrix)
        train_set_labels = train_df['label']
    elif node_emb_technique == "autoencoder":
        # TODO Implement
        pass
    elif node_emb_technique == "None":
        train_set = adj_matrix
        train_set_labels = train_df['label']

    return train_set, train_set_labels

