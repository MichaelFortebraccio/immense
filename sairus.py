import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
from os.path import exists
from text_preprocessing import TextPreprocessing
from word_embedding import WordEmb
from ae import AE
from node_classification.graph_embeddings.node2vec import Node2VecEmbedder
from node_classification.decision_tree import *
from utils import create_or_load_tweets_list
from tqdm import tqdm
from mlp import MLP

seed = 123
np.random.seed(seed)


def train(train_df, l):
    train_df = train_df.reset_index()
    # train W2V on Twitter dataset
    tok = TextPreprocessing()
    tweet = tok.token_list(train_df['text_cleaned'].tolist())
    w2v_model = WordEmb(tweet, l)

    # split tweet in safe and dangerous
    dang_tweets = train_df.loc[train_df['label'] == 1]['text_cleaned']
    safe_tweets = train_df.loc[train_df['label'] == 0]['text_cleaned']

    list_dang_tweets = tok.token_list(dang_tweets)
    list_safe_tweets = tok.token_list(safe_tweets)

    if not exists('model/w2v_text_{}.h5'.format(l)):
        w2v_model.train_w2v()


    # convert text to vector
    list_dang_tweets = create_or_load_tweets_list(path='dataset/textual_data/list_dang_tweets.pickle', w2v_model=w2v_model,
                                                  tokenized_list=list_dang_tweets)
    list_safe_tweets = create_or_load_tweets_list(path='dataset/textual_data/list_safe_tweets.pickle', w2v_model=w2v_model,
                                                  tokenized_list=list_safe_tweets)

    #################TRAIN AND LOAD SAFE AND DANGEROUS AUTOENCODER####################
    dang_ae = AE(input_len=512, X_train=list_dang_tweets, label='dang').train_autoencoder() # .load_autoencoder()
    safe_ae = AE(input_len=512, X_train=list_safe_tweets, label='safe').train_autoencoder()
    '''dang_ae = AE(input_len=512, X_train=list_dang_tweets, label='dang').train_autoencoder()
    safe_ae = AE(input_len=512, X_train=list_safe_tweets, label='safe').train_autoencoder()'''

    ################# TRAIN OR LOAD DECISION TREES ####################
    rel_tree_path = "model/dtree_rel.h5"
    clos_tree_path = "model/dtree_clos.h5"
    path_to_edges_rel = "node_classification/graph_embeddings/stuff/network.edg"
    path_to_edges_clos = "node_classification/graph_embeddings/stuff/closeness_network.edg"
    rel_n2v_path = "model/n2v_rel.h5"
    clos_n2v_path = "model/n2v_clos.h5"

    n2v_rel = Node2VecEmbedder(path_to_edges=path_to_edges_rel, weighted=False, directed=True, n_of_walks=10,
                               walk_length=10, embedding_size=128, p=1, q=4, epochs=100, model_path=rel_n2v_path,
                               name="relationships").learn_n2v_embeddings()  # .load_model()
    if not exists(rel_tree_path):  # IF THE DECISION TREE HAS NOT BEEN LEARNED, LOAD/TRAIN THE N2V MODEL
        train_set_ids_rel = [i for i in train_df['id'] if str(i) in n2v_rel.wv.key_to_index]
        train_decision_tree(train_set_ids=train_set_ids_rel, save_path=rel_tree_path, n2v_model=n2v_rel,
                            train_set_labels=train_df[train_df['id'].isin(train_set_ids_rel)]['label'],
                            name="relationships")

    n2v_clos = Node2VecEmbedder(path_to_edges=path_to_edges_clos, weighted=True, directed=False, n_of_walks=10,
                                walk_length=10, embedding_size=128, p=1, q=4, epochs=100, model_path=clos_n2v_path,
                                name="closeness").learn_n2v_embeddings()   #.load_model()
    if not exists(clos_tree_path):
        train_set_ids_clos = [i for i in train_df['id'] if str(i) in n2v_clos.wv.key_to_index]
        train_decision_tree(train_set_ids=train_set_ids_clos, save_path=clos_tree_path, n2v_model=n2v_clos,
                            train_set_labels=train_df[train_df['id'].isin(train_set_ids_clos)]['label'],
                            name="closeness")

    tree_rel = load_decision_tree(rel_tree_path)
    tree_clos = load_decision_tree(clos_tree_path)

    ################# NOW THAT WE HAVE THE MODELS WE CAN OBTAIN THE TRAINING SET FOR THE MLP #################
    dataset = np.zeros((len(train_df), 7))

    list_tweets_embs = w2v_model.text_to_vec(tweet)
    prediction_dang = dang_ae.predict(list_tweets_embs)
    prediction_safe = safe_ae.predict(list_tweets_embs)

    tweets_sigmoid = tf.keras.activations.sigmoid(tf.constant(list_tweets_embs, dtype=tf.float32)).numpy()
    prediction_loss_dang = tf.keras.losses.mse(prediction_dang, tweets_sigmoid).numpy()
    prediction_loss_safe = tf.keras.losses.mse(prediction_safe, tweets_sigmoid).numpy().tolist()
    labels = [1 if i < j else 0 for i, j in zip(prediction_loss_dang, prediction_loss_safe)]
    dataset[:, 0] = prediction_loss_dang
    dataset[:, 1] = prediction_loss_safe
    dataset[:, 2] = np.array(labels)

    for index, row in tqdm(train_df.iterrows()):
        id = row['id']
        if id in n2v_rel.wv.key_to_index:
            pr, conf = test_decision_tree(test_set_ids=[str(id)], test_set_labels=row['label'], cls=tree_rel,
                                          n2v_model=n2v_rel)
        else:
            pr, conf = row['label'], 1.0
        dataset[index, 3] = pr
        dataset[index, 4] = conf
        if id in n2v_clos.wv.key_to_index:
            pr, conf = test_decision_tree(test_set_ids=[str(id)], test_set_labels=row['label'], cls=tree_clos,
                                          n2v_model=n2v_clos)
        else:
            pr, conf = row['label'], 1.0
        dataset[index, 5] = pr
        dataset[index, 6] = conf

    mlp = MLP(X_train=dataset, y_train=np.array(train_df['label']))
    mlp.train()
    return dang_ae, safe_ae, n2v_rel, n2v_clos, tree_rel, tree_clos, mlp


def test(test_df, train_df, dang_ae, safe_ae, tree_rel, tree_clos, n2v_rel, n2v_clos, mlp, l):
    test_df = test_df.reset_index()
    test_set = np.zeros(shape=(len(test_df), 7))

    tok = TextPreprocessing()
    tweets = tok.token_list(test_df['text_cleaned'].tolist())
    w2v_model = WordEmb(tweets, l)
    test_tweets_embs = w2v_model.text_to_vec(tweets)
    pred_dang = dang_ae.predict(test_tweets_embs)
    pred_safe = safe_ae.predict(test_tweets_embs)

    test_tweets_sigmoid = tf.keras.activations.sigmoid(tf.constant(test_tweets_embs, dtype=tf.float32)).numpy()
    pred_loss_dang = tf.keras.losses.mse(pred_dang, test_tweets_sigmoid).numpy()
    pred_loss_safe = tf.keras.losses.mse(pred_safe, test_tweets_sigmoid).numpy()
    labels = [1 if i < j else 0 for i, j in zip(pred_loss_dang, pred_loss_safe)]
    test_set[:, 0] = pred_loss_dang
    test_set[:, 1] = pred_loss_safe
    test_set[:, 2] = np.array(labels)

    # At test time, if we meet an instance that doesn't have information about relationships or closeness, we will
    # replace the decision tree prediction with the most frequent label in the training set
    pred_missing_info = train_df['label'].value_counts().argmax()
    conf_missing_info = max(train_df['label'].value_counts()) / len(train_df)  # ratio

    for index, row in tqdm(test_df.iterrows()):
        id = row['id']
        if id in n2v_rel.wv.key_to_index:
            pr, conf = test_decision_tree(test_set_ids=[str(id)], test_set_labels=row['label'], cls=tree_rel,
                                          n2v_model=n2v_rel)
        else:
            pr, conf = pred_missing_info, conf_missing_info
        test_set[index, 3] = pr
        test_set[index, 4] = conf
        if id in n2v_clos.wv.key_to_index:
            pr, conf = test_decision_tree(test_set_ids=[str(id)], test_set_labels=row['label'], cls=tree_clos,
                                          n2v_model=n2v_clos)
        else:
            pr, conf = pred_missing_info, conf_missing_info
        test_set[index, 5] = pr
        test_set[index, 6] = conf

    return mlp.test(test_set, np.array(test_df['label']))


def cross_validation(dataset_path, n_folds):
    df = pd.read_csv(dataset_path, sep=',')
    X = df
    y = df['label']

    st = StratifiedKFold(n_splits=n_folds)
    folds = st.split(X=X, y=y)
    l = []
    for k, (train_idx, test_idx) in enumerate(folds):
        dang_ae, safe_ae, n2v_rel, n2v_clos, tree_rel, tree_clos, mlp = train(df.iloc[train_idx], k)
        p, r, f1, s = test(df.iloc[test_idx], df.iloc[train_idx], dang_ae, safe_ae, tree_rel, tree_clos, n2v_rel,
                           n2v_clos, mlp, k)
        l.append((p, r, f1, s))