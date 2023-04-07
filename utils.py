from gensim.models import Word2Vec
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
from os.path import exists

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
            #print(df.loc[df['id'] == int(k)]['label'].item())
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

