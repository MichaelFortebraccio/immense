from gensim.models import Word2Vec
from keras.models import load_model
from os.path import exists, join
from sklearn.model_selection import train_test_split
import pickle
import numpy as np


def save_to_pickle(name, c):
    with open(name, 'wb') as f:
        pickle.dump(c, f)

def load_from_pickle(name):
    with open(name, 'rb') as f:
        return pickle.load(f)

def prepare_for_decision_tree(df, mod: Word2Vec):
    y = []
    for k in mod.wv.key_to_index.keys():
        try:
            y.append(df.loc[df['id']==int(k)]['label'].item())
        except ValueError:
            print(k)
            continue
    X_train, X_test, y_train, y_test = train_test_split(mod.wv.vectors, y, test_size=0.2, train_size=0.8)
    return X_train, X_test, y_train, y_test


def convert_ids(df):
    """Use the matches file for converting the IDs"""
    d = {}
    with open("node_classification/graph_embeddings/stuff/closeness_matches", 'r') as f:
        for l in f.readlines():
            l = l.split("\t")
            d[(l[0])] = str(l[1]).strip()
    d2 = {}
    for k in d.keys():
        d2[int(k)] = d[k]
    df['id'] = df['id'].replace(d2)
    return df


def correct_edg_format(fname):
    l = []
    with open(fname, 'r') as f:
        for line in f.readlines():
            l.append(line.split("\t\t"))
        f.close()
    with open(fname+"2", 'w') as f:
        for e in l:
            str = ""
            for el in e:
                str += "{}\t".format(el.strip())
            str += "\n"
            f.write(str)
        f.close()


def create_or_load_post_list(path, w2v_model, tokenized_list):
    if exists(path):
        with open(path, 'rb') as handle:
            post_list = pickle.load(handle)
    else:
        post_list = w2v_model.text_to_vec(tokenized_list)
        with open(path, 'wb') as handle:
            pickle.dump(post_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return post_list


def is_square(m):
    return m.shape[0] == m.shape[1]

def get_ne_models(models_dir, rel_technique, spat_technique, adj_mat_rel_path=None, id2idx_rel_path=None, adj_mat_spat_path=None, id2idx_spat_path=None):
    """
    Depending on the chosen node embedding techniques, loads and returns the corresponding models needed for doing inference
    Args:
        models_dir: Directory containing the models
        rel_technique:
        spat_technique:
        adj_mat_rel_path: Path to the relational adj matrix
        id2idx_rel_path:
        adj_mat_spat_path: Path to the spatial adj matrix
        id2idx_spat_path:

    Returns:

    """
    mod_dir_rel = join(models_dir, "node_embeddings", "rel", rel_technique)
    mod_dir_spat = join(models_dir, "node_embeddings", "spat", spat_technique)

    n2v_rel = None
    n2v_spat = None
    pca_rel = None
    pca_spat = None
    ae_rel = None
    ae_spat = None
    adj_mat_rel = None
    adj_mat_spat = None
    id2idx_rel = None
    id2idx_spat = None
    if rel_technique == "node2vec":
        n2v_rel = Word2Vec.load(join(mod_dir_rel, "n2v_rel.h5"))
        id2idx_rel = n2v_rel.wv.key_to_index
    elif rel_technique == "autoencoder":
        ae_rel = load_model(join(mod_dir_rel, "encoder_rel.h5"))
        if not adj_mat_rel_path:
            raise Exception("You need to provide the path to the relational adjacency matrix")
        if not id2idx_rel_path:
            raise Exception("You need to provide the path to the file with the matchings between node IDs and the index of their row in the relational adjacency matrix")
        adj_mat_rel = np.genfromtxt(adj_mat_rel_path, delimiter=",")
        id2idx_rel = load_from_pickle(id2idx_rel_path)
    elif rel_technique == "pca":
        pca_rel = load_from_pickle(join(mod_dir_rel, "pca_rel.pkl"))
        if not adj_mat_rel_path:
            raise Exception("You need to provide the path to the relational adjacency matrix")
        if not id2idx_rel_path:
            raise Exception("You need to provide the path to the file with the matchings between node IDs and the index of their row in the relational adjacency matrix")
        adj_mat_rel = np.genfromtxt(adj_mat_rel_path, delimiter=",")
        id2idx_rel = load_from_pickle(id2idx_rel_path)
    elif rel_technique == "none":
        if not adj_mat_rel_path:
            raise Exception("You need to provide the path to the relational adjacency matrix")
        if not id2idx_rel_path:
            raise Exception("You need to provide the path to the file with the matchings between node IDs and the index of their row in the relational adjacency matrix")
        adj_mat_rel = np.genfromtxt(adj_mat_rel_path, delimiter=',')
        id2idx_rel = load_from_pickle(id2idx_rel_path)

    if spat_technique == "node2vec":
        n2v_spat = Word2Vec.load(join(mod_dir_spat, "n2v_spat.h5"))
        id2idx_spat = n2v_spat.wv.key_to_index
    elif spat_technique == "autoencoder":
        ae_spat = load_model(join(mod_dir_spat, "encoder_spat.h5"))
        if not adj_mat_spat_path:
            raise Exception("You need to provide the path to the spatial adjacency matrix")
        if not id2idx_rel_path:
            raise Exception("You need to provide the path to the file with the matchings between node IDs and the index of their row in the spatial adjacency matrix")
        adj_mat_spat = np.genfromtxt(adj_mat_spat_path, delimiter=",")
        id2idx_spat = load_from_pickle(id2idx_spat_path)
    elif spat_technique == "pca":
        pca_spat = load_from_pickle(join(mod_dir_spat, "pca_spat.pkl"))
        if not adj_mat_spat_path:
            raise Exception("You need to provide the path to the spatial adjacency matrix")
        if not id2idx_rel_path:
            raise Exception("You need to provide the path to the file with the matchings between node IDs and the index of their row in the spatial adjacency matrix")
        adj_mat_spat = np.genfromtxt(adj_mat_spat_path, delimiter=",")
        id2idx_spat = load_from_pickle(id2idx_spat_path)
    elif spat_technique == "none":
        if not adj_mat_spat_path:
            raise Exception("You need to provide the path to the spatial adjacency matrix")
        if not id2idx_rel_path:
            raise Exception("You need to provide the path to the file with the matchings between node IDs and the index of their row in the spatial adjacency matrix")
        adj_mat_spat = np.genfromtxt(adj_mat_spat_path, delimiter=',')
        id2idx_spat = load_from_pickle(id2idx_spat_path)

    return n2v_rel, n2v_spat, pca_rel, pca_spat, ae_rel, ae_spat, adj_mat_rel, id2idx_rel, adj_mat_spat, id2idx_spat
