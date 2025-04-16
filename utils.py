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
