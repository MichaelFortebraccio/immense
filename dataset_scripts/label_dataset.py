import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors, Word2Vec
from modelling.text_preprocessing import TextPreprocessing
from os.path import join
from tqdm import tqdm
from utils import save_to_pickle, load_from_pickle, kfold


def cosine_similarity(matrix, vector):
    # Normalize the matrix rows and the vector
    normat = np.linalg.norm(matrix, axis=1, keepdims=True)
    norvec = np.linalg.norm(vector)
    matrix_norm = matrix / normat
    vector_norm = vector / norvec

    # Calculate the cosine similarity using broadcasting
    similarity = np.dot(matrix_norm, vector_norm)

    return similarity


def text_to_vec(mod: KeyedVectors, posts):
    """
    Obtain a vector from a textual content. This function is needed even if it already exists in the WordEmb class,
    because this function works using a pretrained word embedding model. Also, this function returns a list and a number
    while the function in WordEmb returns a dictionary
    :param mod: Word2Vec model
    :param posts: List of posts
    :return list_tot: A list having the same length of posts. The elements are arrays containing the sum of the
    embeddings of the words in that post
    :return a: Number of words found in the posts list that are not included in the w2v model
    """
    list_tot = []
    null_idxs = []       # List containing the indexes of the elements containing empty posts
    i = 0
    e = {w.lower(): w for w in mod.key_to_index.keys() if w.isalpha() and not w.islower()}
    for tw in tqdm(posts):
        list_temp = []
        if tw:
            for t in tw:
                try:
                    learned_embedding = mod[t]
                    f_l = True
                except KeyError:
                    try:
                        learned_embedding = mod[e[t]]
                        f_l = True
                    except KeyError:
                        f_l = False
                if f_l:
                    list_temp.append(learned_embedding)
                else:
                    list_temp.append(np.zeros(shape=(300,)))
        else:
            null_idxs.append(i)
        if list_temp:
            list_temp = np.array(list_temp)
            list_temp = np.sum(list_temp, axis=0)
            list_tot.append(list_temp)
        i += 1
    list_tot = np.asarray(list_tot)
    return list_tot, null_idxs


def text_to_vec_w2v(mod: Word2Vec, posts):
    """
    Obtain a vector from a textual content. This function is needed even if it already exists in the WordEmb class,
    because this function works using a pretrained word embedding model while the other works on list
    :param mod: Gensim word2vec model trained on the dataset
    :param posts: List of posts
    """
    list_tot = []
    wv = mod.wv
    for post in posts:
        list_temp = []
        if post:
            for t in post:
                try:
                    learned_embedding = wv[t]
                    list_temp.append(learned_embedding)
                except KeyError:
                    list_temp.append(np.zeros(shape=(300,)))
        else:
            continue
        if list_temp:
            list_temp = np.array(list_temp)
            list_temp = np.sum(list_temp, axis=0)
            list_tot.append(list_temp)
    list_tot = np.asarray(list_tot)
    return list_tot


def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))


def label_based_on_relationships(df, path_to_rel="../dataset/big_dataset/graph/social_net.edg", factor=0.1):
    """
    A user can be risky even if he doesn't post malicious content, but directly follow many users who do that. This
    function detects the users that fall in this category
    :param df: dataframe containing the users and their associated content
    :param path_to_rel: Path to the edge list depicting the social network
    :param factor: If the ratio of risky users followed by a safe user is > factor, then the user is relabeled as risky
    """
    safe = df[df.label == 0]
    risky = df[df.label == 1]
    safe_ids = [int(i) for i in safe.id.values]
    risky_ids = [int(i) for i in risky.id.values]
    with open(path_to_rel, 'r') as f:
        fol_dict = {}
        for l in f.readlines():
            follower, followed = l.split("\t")
            follower = int(follower.strip())
            followed = int(followed.strip())
            if follower not in fol_dict.keys():
                fol_dict[follower] = [followed]
            else:
                fol_dict[follower].append(followed)
    neg = []
    for k in fol_dict.keys():
        inters = intersection(risky_ids, fol_dict[k])
        if k in safe_ids and len(inters) > len(fol_dict[k])*factor:       # Segue almeno il 10% di utenti risky
            neg.append(int(k))
    return neg


def plot_values(sorted_values, type, l):
    # Create a list of indices for the elements
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    plt.xlabel('Index')
    plt.ylabel(type)
    plt.title('{} to Base Array'.format(type))
    labels = [l, "risky"]
    vals_imdb = sorted_values[0][0]
    vals_evil = sorted_values[1][0]
    indices = list(range(len(vals_imdb)+len(vals_evil)))
    ax1.scatter(indices[:len(vals_imdb)], vals_imdb, label=labels[0])
    ax1.scatter(indices[len(vals_imdb):], vals_evil, label=labels[1])
    plt.legend()
    plt.show()


def plot_single_values(values, type="cosine", l="safe"):
    # Compare the points of safe posts to the points of risky posts
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    plt.xlabel('Index')
    plt.ylabel(type)
    plt.title('{} to Base Array'.format(type))
    labels = [l, "risky"]
    vals_unknown = values[0][0]
    indices = np.arange(len(vals_unknown))
    ax1.scatter(indices[:len(vals_unknown)], vals_unknown, label=labels[0])
    plt.legend()
    plt.show()


def plot_sim(values: list, type="cosine"):
    # Display the points in a plot
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    plt.xlabel('Index')
    plt.ylabel(type)
    plt.title('{} to Base Array'.format(type))
    ax1.scatter(np.arange(len(values)), values)
    plt.legend()
    plt.show()


def main_whole(risky_ds_path, dataset_to_label_path, model_path, sim_fname):
    """
    label your dataset
    :param risky_ds_path: Path to the dataset containing posts that are known to be risky
    :param dataset_to_label_path: Path to the dataset that has to be labeled
    :param model_path: Path to the word embedding model
    :param sim_fname: Name of the file where we will serialize an array that, for each user, will contain the
    value of the highest similarity among that user embeddings and each of the risky posts
    """
    df_negative = pd.read_csv(risky_ds_path)
    df_to_label = pd.read_csv(dataset_to_label_path)

    tok = TextPreprocessing()
    posts_content_negative = tok.token_list(df_negative, text_field_name="text")
    posts_content_to_label = tok.token_list(df_to_label, text_field_name="text_cleaned")
    model_negative = KeyedVectors.load_word2vec_format(model_path, binary=True)
    print("model loaded")
    similarities = {}
    t2v_to_label, idxs = text_to_vec(posts=posts_content_to_label, mod=model_negative)
    if idxs:
        df_to_label = df_to_label.drop(idxs)
        df_to_label = df_to_label.drop(columns=[c for c in df_to_label.columns if c not in ['id', 'text_cleaned']])
        df_to_label.to_csv(dataset_to_label_path)
    # t2v_to_label = load_from_pickle("t2vtl.pkl")
    t2v_negative, _ = text_to_vec(posts=posts_content_negative, mod=model_negative)
    i = 0
    for j in tqdm(range(t2v_to_label.shape[0])):
        sim = cosine_similarity(t2v_negative, t2v_to_label[j, :])
        similarities[j] = np.nanmax(sim)
        i += 1
    save_to_pickle(name=sim_fname, c=np.array(list(similarities.values())))
    plot_sim(list(similarities.values()))


def add_label(df, ratio, sim_array):
    # If sim(user_emb, negative_emb) > ratio, then user = risky else user = safe
    l = [0 if sim_array[i] < ratio else 1 for i in range(len(sim_array))]
    df['label'] = l
    return df


if __name__ == "__main__":
    dataset_dir = join("..", "dataset", "big_dataset")

    allowed_columns = ['id', 'text_cleaned', 'label']   # Used when deleting columns named 'Unnamed: 0' and so on
    risky_df = join(dataset_dir, "evil_preprocessed.csv")   # csv containing the risky content used for labeling our dataset
    # to_label = join(dataset_dir, "unlabelled_dataset.csv")
    to_label = join(dataset_dir, "df_with_rel.csv")
    path_to_rel = join(dataset_dir, "graph", "social_net.edg")
    model_path = "../evil/google_w2v.bin"
    sim2label_fname = "sim_to_label.pkl"

    main_whole(risky_ds_path=risky_df, dataset_to_label_path=to_label, model_path=model_path, sim_fname=sim2label_fname)
    df = pd.read_csv(to_label)
    print("Adding label column to the dataset")


    #df = pd.read_csv("../dataset/big_dataset/ppp.csv")
    ar = load_from_pickle(sim2label_fname)
    df = add_label(df, ratio=0.88, sim_array=ar)
    factors = [0.1]
    for factor in factors:
        neg = label_based_on_relationships(df=df, factor=factor, path_to_rel=path_to_rel)
        neg_idxs = []
        for index, r in df.iterrows():
            if r.id in neg:
                neg_idxs.append(index)
        df = df.drop(columns=[c for c in df.columns if c not in allowed_columns])
        df.to_csv(join(dataset_dir, "dataset_labeled88_full.csv"))

        ## THE REDUCED DATASET CONTAINS THE TOP 2% ELEMENTS WITH HIGHEST SIMILARITY, AS RISKY (=1) POINTS, AND THE BOTTOM 25% AS SAFE (=0) POINTS. THE REMAINING IS IGNORED
        sorted_indexes = ar.argsort()[::-1]
        """twoperc = int(np.ceil(len(ar)*2/100))
        twperc = int(np.ceil(len(ar)*25/100))
        l1 = sorted_indexes[:twoperc].tolist()
        l2 = sorted_indexes[-twperc:].tolist()
        indexes = list(set(l1 + l2 + neg_idxs))
        reduced_dataset = df.iloc[indexes]
        reduced_dataset = reduced_dataset.drop(columns=[c for c in reduced_dataset.columns if c not in allowed_columns])
        reduced_dataset.to_csv(join(dataset_dir, "tweets_labeled_089_{}_27perc.csv".format(int(factor * 100))))
        """