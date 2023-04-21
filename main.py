import pandas as pd
import numpy as np
from os.path import exists
from os import makedirs
from modelling.sairus import train, test, predict_user
import gdown
from gensim.models import Word2Vec
from keras.models import load_model
import pickle
import argparse
from node_classification.decision_tree import load_decision_tree
from utils import is_square, load_from_pickle
seed = 123
np.random.seed(seed)


def main(textual_content_link, rel_technique="node2vec", spat_technique="node2vec", social_net_url=None, spatial_net_url=None,
         rel_adj_mat_path=None, spat_adj_mat_path=None, id2idx_rel_path=None, id2idx_spat_path=None, word_embedding_size=512, window=5,
         w2v_epochs=10, spat_node_embedding_size=128, rel_node_embedding_size=128, n_of_walks_spat=10, n_of_walks_rel=10,
         walk_length_spat=10, walk_length_rel=10, p_spat=1, p_rel=1, q_spat=4, q_rel=4, n2v_epochs_spat=100, n2v_epochs_rel=100,
         rel_autoenc_epochs=20, spat_autoenc_epochs=20):
    dataset_dir = "dataset"
    models_dir = "models"
    if not exists(dataset_dir):
        makedirs(dataset_dir)
    if not exists(models_dir):
        makedirs(models_dir)
    posts_path = "{}/posts_labeled.csv".format(dataset_dir)
    social_path = "{}/social_network.edg".format(dataset_dir)
    closeness_path = "{}/spatial_network.edg".format(dataset_dir)
    if not exists(posts_path):
        gdown.download(url=textual_content_link, output=posts_path, quiet=False, fuzzy=True)
    if not exists(social_path):
        gdown.download(url=social_net_url, output=social_path, quiet=False, fuzzy=True)
    if not exists(closeness_path):
        gdown.download(url=spatial_net_url, output=closeness_path, quiet=False, fuzzy=True)
    train_path = "{}/train.csv".format(dataset_dir)
    test_path = "{}/test.csv".format(dataset_dir)
    df = pd.read_csv(posts_path, sep=',')
    cols = ['index', 'label', 'id', 'text_cleaned']
    if not exists(train_path) or not exists(test_path):
        shuffled_df = df.sample(frac=1, random_state=1).reset_index()    # Shuffle the dataframe
        shuffled_df = shuffled_df.drop(columns=[col for col in shuffled_df.columns if col not in cols])
        shuffled_df.to_csv("dataset/shuffled_content.csv")
        idx = round(len(shuffled_df)*0.8)
        train_df = shuffled_df[:idx]
        test_df = shuffled_df[idx:]
        train_df = train_df.reset_index()
        test_df = test_df.reset_index()
        train_df = train_df.drop(columns=[col for col in train_df.columns if col not in cols])
        test_df = test_df.drop(columns=[col for col in test_df.columns if col not in cols])
        train_df.to_csv(train_path)
        test_df.to_csv(test_path)
    else:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

    adj_mat_rel = None
    id2idx_rel = None
    adj_mat_spat = None
    id2idx_spat = None
    if rel_technique.lower() in ["autoencoder", "pca", "none"]:
        if not rel_adj_mat_path or not id2idx_rel_path:
            raise Exception("You need to provide a path to the relational adjacency matrix and to the pkl file with the matching between the IDs and the matrix rows")
        adj_mat_rel = np.genfromtxt(rel_adj_mat_path, delimiter=",")
        if not is_square(adj_mat_rel):
            raise Exception("The relational adjacency matrix is not square")
        id2idx_rel = load_from_pickle(id2idx_rel_path)
    if spat_technique.lower() in ["autoencoder", "pca", "none"]:
        if not spat_adj_mat_path or not id2idx_spat_path:
            raise Exception("You need to provide a path to the spatial adjacency matrix and to the pkl file with the matching between the IDs and the matrix rows")
        adj_mat_spat = np.genfromtxt(spat_adj_mat_path, delimiter=",")
        if not is_square(adj_mat_spat):
            raise Exception("The spatial adjacency matrix is not square")
        id2idx_spat = load_from_pickle(id2idx_spat_path)

    dang_ae, safe_ae, w2v_model, mlp, mod_dir_rel, mod_dir_spat = train(
        train_df=train_df, full_df=df, dataset_dir=dataset_dir, model_dir=models_dir, rel_path=social_path, spatial_path=closeness_path,
        word_embedding_size=word_embedding_size, window=window, w2v_epochs=w2v_epochs, n_of_walks_spat=n_of_walks_spat,
        n_of_walks_rel=n_of_walks_rel, walk_length_spat=walk_length_spat, walk_length_rel=walk_length_rel,
        spat_node_embedding_size=spat_node_embedding_size, rel_node_embedding_size=rel_node_embedding_size, p_spat=p_spat,
        p_rel=p_rel, q_spat=q_spat, q_rel=q_rel, n2v_epochs_spat=n2v_epochs_spat, n2v_epochs_rel=n2v_epochs_rel, rel_node_emb_technique=rel_technique,
        spat_node_emb_technique=spat_technique, adj_matrix_spat=adj_mat_spat, adj_matrix_rel=adj_mat_rel, id2idx_rel=id2idx_rel,
        id2idx_spat=id2idx_spat, rel_ae_epochs=rel_autoenc_epochs, spat_ae_epochs=spat_autoenc_epochs)

    n2v_spat = None
    n2v_rel = None
    pca_spat = None
    pca_rel = None
    ae_spat = None
    ae_rel = None
    tree_rel = load_decision_tree("{}/dtree.h5".format(mod_dir_rel))
    tree_spat = load_decision_tree("{}/dtree.h5".format(mod_dir_spat))
    if rel_technique == "node2vec":
        n2v_rel = Word2Vec.load("{}/n2v_rel.h5".format(mod_dir_rel))
        id2idx_rel = n2v_rel.wv.key_to_index
    elif rel_technique == "autoencoder":
        ae_rel = load_model("{}/encoder_rel.h5".format(mod_dir_rel))
    elif rel_technique == "pca":
        with open("{}/pca_rel.pkl".format(mod_dir_rel), 'rb') as f:
            pca_rel = pickle.load(f)

    if spat_technique == "node2vec":
        n2v_spat = Word2Vec.load("{}/n2v_spat.h5".format(mod_dir_spat))
        id2idx_spat = n2v_spat.wv.key_to_index
    elif spat_technique == "autoencoder":
        ae_spat = load_model("{}/encoder_spat.h5".format(mod_dir_spat))
    elif spat_technique == "pca":
        with open("{}/pca_spat.pkl".format(mod_dir_spat), 'rb') as f:
            pca_spat = pickle.load(f)
    """test(train_df=train_df, test_df=test_df, w2v_model=w2v_model, dang_ae=dang_ae, safe_ae=safe_ae, tree_rel=tree_rel, tree_spat=tree_spat, mlp=mlp, ae_rel=ae_rel,
         spat_node_emb_technique=spat_technique, rel_node_emb_technique=rel_technique, id2idx_rel=id2idx_rel, id2idx_spat=id2idx_spat, ae_spat=ae_spat,
         adj_matrix_spat=adj_mat_spat, adj_matrix_rel=adj_mat_rel, n2v_rel=n2v_rel, n2v_spat=n2v_spat, pca_rel=pca_rel, pca_spat=pca_spat)"""
    pred = predict_user(test_df.iloc[3], w2v_model=w2v_model, dang_ae=dang_ae, safe_ae=safe_ae, df=test_df, tree_rel=tree_rel, tree_spat=tree_spat, mlp=mlp,
                        rel_node_emb_technique="none", spat_node_emb_technique="pca", id2idx_rel=id2idx_rel, id2idx_spat=id2idx_spat, ae_spat=ae_spat, ae_rel=ae_rel,
                        adj_matrix_spat=adj_mat_spat, adj_matrix_rel=adj_mat_rel, n2v_rel=n2v_rel, n2v_spat=n2v_spat, pca_rel=pca_rel, pca_spat=pca_spat)
    print("The user is: {}".format("risky" if pred == 1 else "safe"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("textual_content_link", type=str, required=True, help="Link to the file containing the posts")
    parser.add_argument("social_net_url", type=str, required=False, help="Link to the file containing the edges in the social network. Can be ignored if you don't want to use node2vec")
    parser.add_argument("spatial_net_url", type=str, required=False, help="Link to the file containing the edges in the spatial network. Can be ignored if you don't want to use node2vec")
    parser.add_argument("word_embedding_size", type=int, default=512, required=True, help="Dimension of the word embeddings")
    parser.add_argument("window", type=int, default=5, required=True, help="Dimension of the window for learning word embeddings")
    parser.add_argument("w2v_epochs", type=int, default=50, required=True, help="For how many epochs to train the w2v model")
    parser.add_argument("spat_technique", type=str, choices=['node2vec', 'none', 'autoencoder', 'pca'], required=True, help="Technique to adopt for learning spatial node embeddings")
    parser.add_argument("rel_technique", type=str, choices=['node2vec', 'none', 'autoencoder', 'pca'], required=True, help="Technique to adopt for learning relational node embeddings")
    parser.add_argument("spat_node_embedding_size", type=int, default=128, required=True, help="Dimension of the spatial node embeddings")
    parser.add_argument("rel_node_embedding_size", type=int, default=128, required=True, help="Dimension of the relational node embeddings")
    parser.add_argument("spat_adj_mat_path", type=str, required=False, help="Link to the file containing the spatial adjacency matrix. Ignore this parameter if you want to use n2v for learning spatial node embeddings")
    parser.add_argument("rel_adj_mat_path", type=str, required=False, help="Link to the file containing the relational adjacency matrix. Ignore this parameter if you want to use n2v for learning relational node embeddings")
    parser.add_argument("id2idx_spat_path", type=str, required=False, help="Link to the .pkl file with the matchings between node IDs and their index in the spatial adj matrix. Ignore this parameter if you want to use n2v for learning spatial node embeddings")
    parser.add_argument("id2idx_rel_path", type=str, required=False, help="Link to the .pkl file with the matchings between node IDs and their index in the relational adj matrix. Ignore this parameter if you want to use n2v for learning relational node embeddings")
    parser.add_argument("n_of_walks_spat", type=int, default=10, required=False, help="Number of walks for learning spatial embeddings (node2vec)")
    parser.add_argument("n_of_walks_rel", type=int, default=10, required=False, help="Number of walks for learning relational embeddings (node2vec)")
    parser.add_argument("walk_length_spat", type=int, default=10, required=True, help="Length of walks for learning spatial embeddings (node2vec)")
    parser.add_argument("walk_length_rel", type=int, default=10, required=True, help="Length of walks for learning relational embeddings (node2vec)")
    parser.add_argument("p_spat", type=int, default=1, required=False, help="Node2vec hyperparameter p for the spatial embeddings")
    parser.add_argument("p_rel", type=int, default=4, required=False, help="Node2vec hyperparameter p for the relational embeddings")
    parser.add_argument("q_spat", type=int, default=1, required=False, help="Node2vec hyperparameter q for the spatial embeddings")
    parser.add_argument("q_rel", type=int, default=4, required=False, help="Node2vec hyperparameter q for the relational embeddings")
    parser.add_argument("n2v_epochs_spat", type=int, default=100, required=False, help="Epochs for training the n2v model that will learn spatial embeddings")
    parser.add_argument("n2v_epochs_rel", type=int, default=100, required=False, help="Epochs for training the n2v model that will learn relational embeddings")
    parser.add_argument("rel_ae_epochs", type=int, default=50, required=True, help="Epochs for training the autoencoder that will learn relational embeddings")
    parser.add_argument("spat_ae_epochs", type=int, default=50, required=True, help="Epochs for training the autoencoder that will learn spatial embeddings")

    args = parser.parse_args()
    main(args)

    main(="textual_content_link",
         social_net_url="https://drive.google.com/file/d/1MhSo9tMDkfnlvZXPKgxv-HBLP4fSwvmN/view?usp=sharing",
         spatial_net_url="https://drive.google.com/file/d/1fVipJMfIoqVqnImc9l79tLqzTWlhhXCq/view?usp=sharing", word_embedding_size=512, window=5, w2v_epochs=1,
         spat_node_embedding_size=128, rel_node_embedding_size=128, n_of_walks_spat=10, n_of_walks_rel=10, walk_length_spat=10, walk_length_rel=10, p_spat=1,
         p_rel=1, q_spat=4, q_rel=4, n2v_epochs_spat=100, n2v_epochs_rel=100, spat_technique="pca", rel_technique="none",
         spat_adj_mat_path="node_classification/graph_embeddings/stuff/spat_adj_net.csv",
         rel_adj_mat_path="node_classification/graph_embeddings/stuff/adj_net.csv",
         id2idx_rel_path="node_classification/graph_embeddings/stuff/id2idx_rel.pkl",
         id2idx_spat_path="node_classification/graph_embeddings/stuff/id2idx_spat.pkl", rel_autoenc_epochs=20, spat_autoenc_epochs=20)
