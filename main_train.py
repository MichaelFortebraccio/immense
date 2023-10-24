import argparse
from modelling.sairus import train
import numpy as np
import pandas as pd
from os.path import exists, join
from os import makedirs
from utils import adj_list_from_df
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
    n2v_epochs_spat = args.n2v_epochs_spat
    n2v_epochs_rel = args.n2v_epochs_rel
    rel_autoenc_epochs = args.rel_ae_epochs
    spat_autoenc_epochs = args.spat_ae_epochs
    models_dir = args.models_dir
    dataset_dir = args.dataset_dir"""
    dataset_dir = "dataset/definitive_prova"
    models_dir = "dataset/definitive_prova/models/"

    textual_content_path = join(dataset_dir, "da88.csv")
    train_path = join(dataset_dir, "train.csv")
    test_path = join(dataset_dir, "test.csv")

    rel_technique = spat_technique = "graphsage"
    rel_adj_mat_path = spat_adj_mat_path = None
    id2idx_rel_path = join(models_dir, "id2idx_rel.pkl")
    id2idx_spat_path = join(models_dir, "id2idx_spat.pkl")
    social_net_path = join(dataset_dir, "social_network.edg")
    spatial_net_path = join(dataset_dir, "spatial_network.edg")

    # w2v parameters
    word_embedding_size = 512
    w2v_epochs = 15
    # node emb parameters
    spat_node_embedding_size = rel_node_embedding_size = 512
    epochs_spat = epochs_rel = 10

    if not exists(dataset_dir):
        makedirs(dataset_dir)
    if not exists(models_dir):
        makedirs(models_dir)

    df = pd.read_csv(textual_content_path, sep=',')

    cols = ['index', 'label', 'id', 'text_cleaned']
    if not exists(train_path) or not exists(test_path):
        df = df.sample(frac=1, random_state=1).reset_index()  # Shuffle the dataframe
        df = df.drop(columns=[col for col in df.columns if col not in cols])
        df.to_csv(join(dataset_dir, "shuffled_content.csv"))
        idx = round(len(df)*0.8)
        train_df = df[:idx]
        test_df = df[idx:]
        train_df = train_df.reset_index()
        test_df = test_df.reset_index()
        train_df = train_df.drop(columns=[col for col in train_df.columns if col not in cols])
        test_df = test_df.drop(columns=[col for col in test_df.columns if col not in cols])
        train_df.to_csv(train_path)
        test_df.to_csv(test_path)
        adj_list_from_df(train_df, social_net_path, join(dataset_dir, "social_network_train.edg"))
        adj_list_from_df(test_df, social_net_path, join(dataset_dir, "social_network_test.edg"))
        adj_list_from_df(train_df, spatial_net_path, "{}/spatial_network_train.edg".format(dataset_dir))
        adj_list_from_df(test_df, spatial_net_path, "{}/spatial_network_test.edg".format(dataset_dir))
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

    train(train_df=train_df, dataset_dir=dataset_dir, model_dir=models_dir, w2v_epochs=w2v_epochs,
          rel_path="{}/social_network_train.edg".format(dataset_dir), word_embedding_size=word_embedding_size,
          spatial_path="{}/spatial_network_train.edg".format(dataset_dir), spat_node_emb_technique=spat_technique,
          rel_node_emb_technique=rel_technique, spat_node_embedding_size=spat_node_embedding_size,
          rel_node_embedding_size=rel_node_embedding_size, epochs_spat_nembs=epochs_spat, epochs_rel_nembs=epochs_rel,
          adj_matrix_spat_path=spat_adj_mat_path, adj_matrix_rel_path=rel_adj_mat_path, id2idx_rel_path=id2idx_rel_path,
          id2idx_spat_path=id2idx_spat_path, we_model_name="w2v_{}.pkl".format(word_embedding_size), batch_size=64)


if __name__ == "__main__":
    """parser = argparse.ArgumentParser()
    parser.add_argument("--textual_content_path", type=str, required=True, help="Link to the file containing the posts")
    parser.add_argument("--social_net_path", type=str, required=False, default="", help="Link to the file containing the edges in the social network. Can be ignored if you don't want to use node2vec")
    parser.add_argument("--spatial_net_path", type=str, required=False, default="", help="Link to the file containing the edges in the spatial network. Can be ignored if you don't want to use node2vec")
    parser.add_argument("--word_embedding_size", type=int, default=512, required=True, help="Dimension of the word embeddings")
    parser.add_argument("--w2v_epochs", type=int, default=50, required=False, help="For how many epochs to train the w2v model")
    parser.add_argument("--spat_technique", type=str, choices=['node2vec', 'none', 'autoencoder', 'pca'], required=True, help="Technique to adopt for learning spatial node embeddings")
    parser.add_argument("--rel_technique", type=str, choices=['node2vec', 'none', 'autoencoder', 'pca'], required=True, help="Technique to adopt for learning relational node embeddings")
    parser.add_argument("--spat_node_embedding_size", type=int, default=128, required=False, help="Dimension of the spatial node embeddings")
    parser.add_argument("--rel_node_embedding_size", type=int, default=128, required=False, help="Dimension of the relational node embeddings")
    parser.add_argument("--spat_adj_mat_path", type=str, required=False, default="", help="Link to the file containing the spatial adjacency matrix. Ignore this parameter if you want to use n2v for learning spatial node embeddings")
    parser.add_argument("--rel_adj_mat_path", type=str, required=False, default="", help="Link to the file containing the relational adjacency matrix. Ignore this parameter if you want to use n2v for learning relational node embeddings")
    parser.add_argument("--id2idx_spat_path", type=str, required=False, help="Link to the .pkl file with the matchings between node IDs and their index in the spatial adj matrix. Ignore this parameter if you want to use n2v for learning spatial node embeddings")
    parser.add_argument("--id2idx_rel_path", type=str, required=False, help="Link to the .pkl file with the matchings between node IDs and their index in the relational adj matrix. Ignore this parameter if you want to use n2v for learning relational node embeddings")
    parser.add_argument("--n2v_epochs_spat", type=int, default=100, required=False, help="Epochs for training the n2v model that will learn spatial embeddings")
    parser.add_argument("--n2v_epochs_rel", type=int, default=100, required=False, help="Epochs for training the n2v model that will learn relational embeddings")
    parser.add_argument("--rel_ae_epochs", type=int, default=50, required=False, help="Epochs for training the autoencoder that will learn relational embeddings")
    parser.add_argument("--spat_ae_epochs", type=int, default=50, required=False, help="Epochs for training the autoencoder that will learn spatial embeddings")
    parser.add_argument("--models_dir", type=str, default='models', required=False, help="Directory where the models will be saved")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Directory containing the dataset")

    args = parser.parse_args()"""
    args = None
    main_train(args)

