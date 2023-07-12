import numpy as np
import pickle
from gensim.models import Word2Vec
from keras.models import load_model
from os.path import exists, join
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn

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


def get_ne_models(models_dir, rel_technique, spat_technique, spat_ne_dim, rel_ne_dim, mod_dir_rel=None, mod_dir_spat=None, adj_mat_rel_path=None, id2idx_rel_path=None, adj_mat_spat_path=None, id2idx_spat_path=None):
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

    """
    if not mod_dir_rel and not mod_dir_spat:
        mod_dir_rel = join(models_dir, "node_embeddings", "rel", rel_technique, str(rel_ne_dim))
        mod_dir_spat = join(models_dir, "node_embeddings", "spat", spat_technique, str(spat_ne_dim))

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
        n2v_rel = Word2Vec.load(join(mod_dir_rel, "n2v.h5"))
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
        n2v_spat = Word2Vec.load(join(mod_dir_spat, "n2v.h5"))
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


########### UTILITY FUNCTIONS NOT USED IN THE API ###########
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


def kfold(dim, splits):
    prog = int(dim/splits)
    folds_idx = np.arange(start=0, stop=dim, step=prog)
    return folds_idx


def plot_confusion_matrix(y_true, y_pred):
    mat = confusion_matrix(y_true=y_true, y_pred=y_pred)
    plt.figure(figsize=(6, 4))
    ax = plt.subplot()
    sn.heatmap(mat, annot=True, cmap="CMRmap", linewidths=0.5, cbar=False, fmt="d", ax=ax)
    #'Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'crest', 'crest_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'flare', 'flare_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'hot', 'hot_r', 'hsv', 'hsv_r', 'icefire', 'icefire_r', 'inferno', 'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r', 'mako', 'mako_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r', 'rocket', 'rocket_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'turbo', 'turbo_r', 'twilight', 'twilight_r', 'twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r', 'vlag', 'vlag_r', 'winter', 'winter_r'

    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(['0', '1'])
    ax.yaxis.set_ticklabels(['0', '1'])

    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.show()