import matplotlib.pyplot as plt
import numpy as np
import pickle
import seaborn as sn
from exceptions import AdjMatException, Id2IdxException
from gensim.models import Word2Vec
from os.path import exists, join
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def save_to_pickle(name, c):
    with open(name, 'wb') as f:
        pickle.dump(c, f)


def load_from_pickle(name):
    with open(name, 'rb') as f:
        return pickle.load(f)


def is_square(m):
    return m.shape[0] == m.shape[1]


def embeddings_pca(emb_model, emb_technique, dst_dir):
    if emb_technique == "node2vec":
        vectors = emb_model.wv.vectors
        k2i = emb_model.wv.key_to_index
    pca = PCA(n_components=2, random_state=42)
    pca_embs = pca.fit_transform(vectors)
    d = {}
    for k in k2i:
        d[k] = pca_embs[k2i[k]]
    save_to_pickle(join(dst_dir, "reduced_embs.pkl"), d)


def adj_list_from_df(df, path_to_src_edg, path_to_dst_edg, spatial=False, mode="graphsage"):
    """
    Given a dataframe and an edge list, create a new edge list containing only the ids in the dataframe, and write it to
    a new file. This function is used for creating the training and testing social networks from the training and testing dataframes
    """
    ids = list(df.id)
    ids = [int(id) for id in ids]
    edgs_to_keep = []
    with open(path_to_src_edg, 'r') as f:
        for l in tqdm(f.readlines()):
            if not spatial:
                id1, id2 = l.split("\t")
            else:
                spl = l.split("\t")
                id1, id2, weight = spl[0], spl[1], spl[2]
                if mode == "graphsage" and weight == 0.0:
                    continue
            id1 = int(id1.strip())
            id2 = int(id2.strip())
            if id1 in ids and id2 in ids:
                if not spatial:
                    edgs_to_keep.append((id1, id2))
                else:
                    edgs_to_keep.append((id1, id2, float(weight.strip())))
    with open(path_to_dst_edg, 'w') as f:
        for l in edgs_to_keep:
            if not spatial:
                f.write("{}\t{}\n".format(l[0], l[1]))
            else:
                f.write("{}\t{}\t{}\n".format(l[0], l[1], l[2]))


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


def create_id_mapping(path_to_edgelist, dst_path):
    """
    Pytorch requires IDs to be named in progressive way starting from 0. This function creates a dictionary with a mapping
    between original IDs and progressive IDs
    :param path_to_edgelist: Path to the file with the edgelist
    :return:
    """
    lis = []
    d = {}
    with open(path_to_edgelist, 'r') as f:
        for l in tqdm(f.readlines()):
            i1, i2 = l.split("\t")
            lis.append(i1.strip())
            lis.append(i2.strip())
    lst = list(set(lis))
    i = 0
    for el in lst:
        d[i] = el
        i += 1
    save_to_pickle(dst_path, d)
