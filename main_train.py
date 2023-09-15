import pandas as pd
import numpy as np
from os.path import exists
from os import makedirs
from modelling.sairus import train
import argparse
seed = 123
np.random.seed(seed)


def main_train(args=None):
    """textual_content_path = args.textual_content_path
    social_net_path = args.social_net_path
    spatial_net_path = args.spatial_net_path
    rel_adj_mat_path = args.rel_adj_mat_path
    spat_adj_mat_path = args.spat_adj_mat_path
    id2idx_rel_path = args.id2idx_rel_path
    id2idx_spat_path = args.id2idx_spat_path

    rel_technique = args.rel_technique
    spat_technique = args.spat_technique
    word_embedding_size = args.word_embedding_size
    window = args.window
    w2v_epochs = args.w2v_epochs
    spat_node_embedding_size = args.spat_node_embedding_size
    rel_node_embedding_size = args.rel_node_embedding_size
    n_of_walks_spat = args.n_of_walks_spat
    n_of_walks_rel = args.n_of_walks_rel
    walk_length_spat = args.walk_length_spat
    walk_length_rel = args.walk_length_rel
    p_spat = args.p_spat
    p_rel = args.p_rel
    q_spat = args.q_spat
    q_rel = args.q_rel
    n2v_epochs_spat = args.n2v_epochs_spat
    n2v_epochs_rel = args.n2v_epochs_rel
    rel_autoenc_epochs = args.rel_ae_epochs
    spat_autoenc_epochs = args.spat_ae_epochs
    models_dir = args.models_dir
    dataset_dir = args.dataset_dir"""
    """
    dataset_dir = "dataset/anthony"
    models_dir = "dataset/anthony/models"
    textual_content_path = "dataset/anthony/tweet_labeled_full.csv"
    train_path = "{}/train.csv".format(dataset_dir)
    test_path = "{}/test.csv".format(dataset_dir)
    """
    dataset_dir = "dataset"
    val = 20
    models_dir = "dataset/models/{}".format(val)

    textual_content_path = "dataset/tweets_labeled_089_{}.csv".format(val)
    train_path = "{}/train_089_{}.csv".format(dataset_dir, val)
    test_path = "{}/test_089_{}.csv".format(dataset_dir, val)

    rel_technique = spat_technique = "graphsage"
    rel_adj_mat_path = id2idx_rel_path = id2idx_spat_path = spat_adj_mat_path = None
    social_net_path = "dataset/graph/sn_labeled_089_20_train.edg"     #.format(models_dir)
    spatial_net_path = "dataset/graph/closeness_network.edg"     #.format(models_dir)
    n_of_walks_spat = n_of_walks_rel = walk_length_spat = walk_length_rel = 10
    spat_node_embedding_size = rel_node_embedding_size = 128
    word_embedding_size = 512
    window = 10
    w2v_epochs = 15
    p_spat = p_rel = 1
    q_spat = q_rel = 4
    n2v_epochs_spat = n2v_epochs_rel = 20
    rel_autoenc_epochs = spat_autoenc_epochs = 0

    if not exists(dataset_dir):
        makedirs(dataset_dir)
    if not exists(models_dir):
        makedirs(models_dir)

    df = pd.read_csv(textual_content_path, sep=',')

    cols = ['index', 'label', 'id', 'text_cleaned']
    if not exists(train_path) or not exists(test_path):
        df = df.sample(frac=1, random_state=1).reset_index()  # Shuffle the dataframe
        df = df.drop(columns=[col for col in df.columns if col not in cols])
        df.to_csv("{}/shuffled_content_089_{}.csv".format(dataset_dir, val))
        idx = round(len(df)*0.8)
        train_df = df[:idx]
        test_df = df[idx:]
        train_df = train_df.reset_index()
        test_df = test_df.reset_index()
        train_df = train_df.drop(columns=[col for col in train_df.columns if col not in cols])
        test_df = test_df.drop(columns=[col for col in test_df.columns if col not in cols])
        train_df.to_csv(train_path)
        test_df.to_csv(test_path)
    else:
        train_df = pd.read_csv(train_path)

    if rel_technique.lower() in ["autoencoder", "pca", "none"]:
        if not rel_adj_mat_path:
            raise Exception("You need to provide a path to the relational adjacency matrix")
        if not id2idx_rel_path:
            raise Exception("You need to provide a path to the pkl file with the matching between the IDs and the relational matrix rows")
    if spat_technique.lower() in ["autoencoder", "pca", "none"]:
        if not spat_adj_mat_path:
            raise Exception("You need to provide a path to the spatial adjacency matrix")
        if not id2idx_spat_path:
            raise Exception("You need to provide a path to the pkl file with the matching between the IDs and the spatial matrix rows")

    train(train_df=train_df, dataset_dir=dataset_dir, model_dir=models_dir, rel_path=social_net_path, spatial_path=spatial_net_path,
          word_embedding_size=word_embedding_size, window=window, w2v_epochs=w2v_epochs, n_of_walks_spat=n_of_walks_spat,
          n_of_walks_rel=n_of_walks_rel, walk_length_spat=walk_length_spat, walk_length_rel=walk_length_rel,
          spat_node_embedding_size=spat_node_embedding_size, rel_node_embedding_size=rel_node_embedding_size, p_spat=p_spat,
          p_rel=p_rel, q_spat=q_spat, q_rel=q_rel, n2v_epochs_spat=n2v_epochs_spat, n2v_epochs_rel=n2v_epochs_rel, rel_node_emb_technique=rel_technique,
          spat_node_emb_technique=spat_technique, adj_matrix_spat_path=spat_adj_mat_path, adj_matrix_rel_path=rel_adj_mat_path, id2idx_rel_path=id2idx_rel_path,
          id2idx_spat_path=id2idx_spat_path, rel_ae_epochs=rel_autoenc_epochs, spat_ae_epochs=spat_autoenc_epochs, we_model_name="w2v_{}_089_{}.pkl".format(word_embedding_size, val))


if __name__ == "__main__":
    """parser = argparse.ArgumentParser()
    parser.add_argument("--textual_content_path", type=str, required=True, help="Link to the file containing the posts")
    parser.add_argument("--social_net_path", type=str, required=False, default="", help="Link to the file containing the edges in the social network. Can be ignored if you don't want to use node2vec")
    parser.add_argument("--spatial_net_path", type=str, required=False, default="", help="Link to the file containing the edges in the spatial network. Can be ignored if you don't want to use node2vec")
    parser.add_argument("--word_embedding_size", type=int, default=512, required=True, help="Dimension of the word embeddings")
    parser.add_argument("--window", type=int, default=5, required=False, help="Dimension of the window for learning word embeddings")
    parser.add_argument("--w2v_epochs", type=int, default=50, required=False, help="For how many epochs to train the w2v model")
    parser.add_argument("--spat_technique", type=str, choices=['node2vec', 'none', 'autoencoder', 'pca'], required=True, help="Technique to adopt for learning spatial node embeddings")
    parser.add_argument("--rel_technique", type=str, choices=['node2vec', 'none', 'autoencoder', 'pca'], required=True, help="Technique to adopt for learning relational node embeddings")
    parser.add_argument("--spat_node_embedding_size", type=int, default=128, required=False, help="Dimension of the spatial node embeddings")
    parser.add_argument("--rel_node_embedding_size", type=int, default=128, required=False, help="Dimension of the relational node embeddings")
    parser.add_argument("--spat_adj_mat_path", type=str, required=False, default="", help="Link to the file containing the spatial adjacency matrix. Ignore this parameter if you want to use n2v for learning spatial node embeddings")
    parser.add_argument("--rel_adj_mat_path", type=str, required=False, default="", help="Link to the file containing the relational adjacency matrix. Ignore this parameter if you want to use n2v for learning relational node embeddings")
    parser.add_argument("--id2idx_spat_path", type=str, required=False, help="Link to the .pkl file with the matchings between node IDs and their index in the spatial adj matrix. Ignore this parameter if you want to use n2v for learning spatial node embeddings")
    parser.add_argument("--id2idx_rel_path", type=str, required=False, help="Link to the .pkl file with the matchings between node IDs and their index in the relational adj matrix. Ignore this parameter if you want to use n2v for learning relational node embeddings")
    parser.add_argument("--n_of_walks_spat", type=int, default=10, required=False, help="Number of walks for learning spatial embeddings (node2vec)")
    parser.add_argument("--n_of_walks_rel", type=int, default=10, required=False, help="Number of walks for learning relational embeddings (node2vec)")
    parser.add_argument("--walk_length_spat", type=int, default=10, required=False, help="Length of walks for learning spatial embeddings (node2vec)")
    parser.add_argument("--walk_length_rel", type=int, default=10, required=False, help="Length of walks for learning relational embeddings (node2vec)")
    parser.add_argument("--p_spat", type=int, default=1, required=False, help="Node2vec hyperparameter p for the spatial embeddings")
    parser.add_argument("--p_rel", type=int, default=4, required=False, help="Node2vec hyperparameter p for the relational embeddings")
    parser.add_argument("--q_spat", type=int, default=1, required=False, help="Node2vec hyperparameter q for the spatial embeddings")
    parser.add_argument("--q_rel", type=int, default=4, required=False, help="Node2vec hyperparameter q for the relational embeddings")
    parser.add_argument("--n2v_epochs_spat", type=int, default=100, required=False, help="Epochs for training the n2v model that will learn spatial embeddings")
    parser.add_argument("--n2v_epochs_rel", type=int, default=100, required=False, help="Epochs for training the n2v model that will learn relational embeddings")
    parser.add_argument("--rel_ae_epochs", type=int, default=50, required=False, help="Epochs for training the autoencoder that will learn relational embeddings")
    parser.add_argument("--spat_ae_epochs", type=int, default=50, required=False, help="Epochs for training the autoencoder that will learn spatial embeddings")
    parser.add_argument("--models_dir", type=str, default='models', required=False, help="Directory where the models will be saved")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Directory containing the dataset")

    args = parser.parse_args()"""
    args = None
    main_train(args)

