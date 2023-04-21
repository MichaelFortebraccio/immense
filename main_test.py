import argparse

from modelling.sairus import test
from keras.models import load_model
from node_classification.decision_tree import load_decision_tree
from gensim.models import Word2Vec
from utils import load_from_pickle
import numpy as np
import pandas as pd

def main_test(args):
    spat_technique = args.spat_technique
    rel_technique = args.rel_technique
    dataset_dir = args.dataset_dir
    models_dir = args.models_dir
    adj_mat_rel_path = args.rel_adj_mat_path
    adj_mat_spat_path = args.spat_adj_mat_path
    we_size = args.word_embedding_size

    train_df = pd.read_csv("{}/train.csv".format(dataset_dir))
    test_df = pd.read_csv("{}/test.csv".format(dataset_dir))
    dang_ae = load_model("{}/autoencoderdang_{}.h5".format(models_dir, we_size))
    safe_ae = load_model("{}/autoencodersafe_{}.h5".format(models_dir, we_size))
    mlp = load_from_pickle("{}/mlp.pkl".format(models_dir))

    mod_dir_rel = "{}/node_embeddings/rel/{}".format(models_dir, rel_technique)
    mod_dir_spat = "{}/node_embeddings/spat/{}".format(models_dir, spat_technique)
    tree_rel = load_decision_tree("{}/dtree.h5".format(mod_dir_rel))
    tree_spat = load_decision_tree("{}/dtree.h5".format(mod_dir_spat))

    n2v_spat = None
    n2v_rel = None
    pca_spat = None
    pca_rel = None
    ae_spat = None
    ae_rel = None
    if rel_technique == "node2vec":
        n2v_rel = Word2Vec.load("{}/n2v_rel.h5".format(mod_dir_rel))
        id2idx_rel = n2v_rel.wv.key_to_index
        adj_mat_rel = None
    elif rel_technique == "autoencoder":
        ae_rel = load_model("{}/encoder_rel.h5".format(mod_dir_rel))
        adj_mat_rel = np.genfromtxt(adj_mat_rel_path, delimiter=",")
    elif rel_technique == "pca":
        adj_mat_rel = np.genfromtxt(adj_mat_rel_path, delimiter=",")
        pca_rel = load_from_pickle("{}/pca_rel.pkl".format(mod_dir_rel))
    elif rel_technique == "none":
        adj_mat_rel = np.genfromtxt(adj_mat_rel_path, delimiter=',')

    if spat_technique == "node2vec":
        n2v_spat = Word2Vec.load("{}/n2v_spat.h5".format(mod_dir_spat))
        adj_mat_spat = None
        id2idx_spat = n2v_spat.wv.key_to_index
    elif spat_technique == "autoencoder":
        adj_mat_spat = np.genfromtxt(adj_mat_spat_path, delimiter=",")
        ae_spat = load_model("{}/encoder_spat.h5".format(mod_dir_spat))
    elif spat_technique == "pca":
        adj_mat_spat = np.genfromtxt(adj_mat_spat_path, delimiter=",")
        pca_spat = load_from_pickle("{}/pca_spat.pkl".format(mod_dir_spat))
    elif spat_technique == "none":
        adj_mat_spat = np.genfromtxt(adj_mat_spat_path, delimiter=',')

    w2v_model = load_from_pickle("{}/w2v_{}.pkl".format(models_dir, we_size))

    test(train_df=train_df, test_df=test_df, w2v_model=w2v_model, dang_ae=dang_ae, safe_ae=safe_ae, tree_rel=tree_rel, tree_spat=tree_spat, mlp=mlp, ae_rel=ae_rel,
         spat_node_emb_technique=spat_technique, rel_node_emb_technique=rel_technique, id2idx_rel=id2idx_rel, id2idx_spat=id2idx_spat, ae_spat=ae_spat,
         adj_matrix_spat=adj_mat_spat, adj_matrix_rel=adj_mat_rel, n2v_rel=n2v_rel, n2v_spat=n2v_spat, pca_rel=pca_rel, pca_spat=pca_spat)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--spat_technique", type=str, choices=['node2vec', 'none', 'autoencoder', 'pca'], required=True, help="Technique adopted for learning spatial node embeddings")
    parser.add_argument("--rel_technique", type=str, choices=['node2vec', 'none', 'autoencoder', 'pca'], required=True, help="Technique adopted for learning relational node embeddings")
    parser.add_argument("--dataset_dir", type=str, default="", required=True, help="Directory containing the train and test set")
    parser.add_argument("--models_dir", type=str, default="", required=True, help="Directory where the models are saved")
    parser.add_argument("--spat_adj_mat_path", type=str, required=False, default="", help="Link to the file containing the spatial adjacency matrix. Ignore this parameter if you used n2v for learning spatial node embeddings")
    parser.add_argument("--rel_adj_mat_path", type=str, required=False, default="", help="Link to the file containing the relational adjacency matrix. Ignore this parameter if you used n2v for learning relational node embeddings")
    parser.add_argument("--word_embedding_size", type=int, required=True, default=128, help="Dimension of the word embeddings learned during the training process")

    args = parser.parse_args()
    main_test(args)
