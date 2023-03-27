import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
from os.path import exists
from os import makedirs
from text_preprocessing import TextPreprocessing
from word_embedding import WordEmb
from ae import AE
from node_classification.graph_embeddings.node2vec import Node2VecEmbedder
from node_classification.decision_tree import *
from utils import create_or_load_tweets_list
from tqdm import tqdm
from mlp import MLP
from node_classification.decision_tree import load_decision_tree
from keras.models import load_model
import celery
import gdown
seed = 123
np.random.seed(seed)


@celery.task(bind=True)
def train(tweets_url, social_network_url, spatial_network_url, word_embedding_size, window, w2v_epochs,
          rel_node_embedding_size, spat_node_embedding_size, n_of_walks_spat, n_of_walks_rel, walk_length_spat,
          walk_length_rel, p_spat, p_rel, q_spat, q_rel, n2v_epochs_spat, n2v_epochs_rel):
    job_id = train.task.id
    dataset_dir = "{}/dataset".format(job_id)
    model_dir = "{}/model".format(job_id)
    makedirs(dataset_dir, exist_ok=True)
    makedirs(model_dir, exist_ok=True)
    tweets_path = "{}/tweet_labeled.csv".format(dataset_dir)
    social_path = "{}/social_network.edg".format(dataset_dir)
    closeness_path = "{}/closeness_network.edg".format(dataset_dir)
    if not exists(tweets_path):
        gdown.download(url=tweets_url, output=tweets_path, quiet=False, fuzzy=True)
    if not exists(social_path):
        gdown.download(url=social_network_url, output=social_path, quiet=False, fuzzy=True)
    if not exists(closeness_path):
        gdown.download(url=spatial_network_url, output=closeness_path, quiet=False, fuzzy=True)

    train_df = pd.read_csv('{}/tweet_labeled_full.csv'.format(dataset_dir), sep=',').reset_index()
    # train W2V on Twitter dataset
    tok = TextPreprocessing()
    tweet = tok.token_list(train_df['text_cleaned'].tolist())
    w2v_model = WordEmb(tweet, embedding_size=word_embedding_size, window=window, epochs=w2v_epochs)

    # split tweet in safe and dangerous
    dang_tweets = train_df.loc[train_df['label'] == 1]['text_cleaned']
    safe_tweets = train_df.loc[train_df['label'] == 0]['text_cleaned']

    list_dang_tweets = tok.token_list(dang_tweets)
    list_safe_tweets = tok.token_list(safe_tweets)

    if not exists('{}/w2v_text_{}.h5'.format(model_dir, word_embedding_size)):
        w2v_model.train_w2v()

    # convert text to vector
    list_dang_tweets = create_or_load_tweets_list(path='{}/list_dang_tweets.pickle'.format(dataset_dir), w2v_model=w2v_model,
                                                  tokenized_list=list_dang_tweets)
    list_safe_tweets = create_or_load_tweets_list(path='{}/list_safe_tweets.pickle'.format(dataset_dir), w2v_model=w2v_model,
                                                  tokenized_list=list_safe_tweets)

    ################# TRAIN AND LOAD SAFE AND DANGEROUS AUTOENCODER ####################
    dang_ae = AE(input_len=word_embedding_size, X_train=list_dang_tweets, label='dang').train_autoencoder() # .load_autoencoder()
    safe_ae = AE(input_len=word_embedding_size, X_train=list_safe_tweets, label='safe').train_autoencoder()

    ################# TRAIN OR LOAD DECISION TREES ####################
    rel_tree_path = "{}/dtree_rel.h5".format(model_dir)
    clos_tree_path = "{}/dtree_clos.h5".format(model_dir)
    rel_n2v_path = "{}/n2v_rel.h5".format(model_dir)
    clos_n2v_path = "{}/n2v_clos.h5".format(model_dir)

    n2v_rel = Node2VecEmbedder(path_to_edges=social_path, weighted=False, directed=True, n_of_walks=n_of_walks_rel,
                               walk_length=walk_length_rel, embedding_size=rel_node_embedding_size, p=p_rel, q=q_rel,
                               epochs=n2v_epochs_rel, model_path=rel_n2v_path, name="relationships").learn_n2v_embeddings()  # .load_model()
    if not exists(rel_tree_path):  # IF THE DECISION TREE HAS NOT BEEN LEARNED, LOAD/TRAIN THE N2V MODEL
        train_set_ids_rel = [i for i in train_df['id'] if str(i) in n2v_rel.wv.key_to_index]
        train_decision_tree(train_set_ids=train_set_ids_rel, save_path=rel_tree_path, n2v_model=n2v_rel,
                            train_set_labels=train_df[train_df['id'].isin(train_set_ids_rel)]['label'],
                            name="relationships")

    n2v_spat = Node2VecEmbedder(path_to_edges=closeness_path, weighted=True, directed=False, n_of_walks=n_of_walks_spat,
                                walk_length=walk_length_spat, embedding_size=spat_node_embedding_size, p=p_spat, q=q_spat,
                                epochs=n2v_epochs_spat, model_path=clos_n2v_path, name="closeness").learn_n2v_embeddings()   #.load_model()
    if not exists(clos_tree_path):
        train_set_ids_clos = [i for i in train_df['id'] if str(i) in n2v_spat.wv.key_to_index]
        train_decision_tree(train_set_ids=train_set_ids_clos, save_path=clos_tree_path, n2v_model=n2v_spat,
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
            pr, conf = test_decision_tree(test_set_ids=[str(id)], cls=tree_rel, n2v_model=n2v_rel)
        else:
            pr, conf = row['label'], 1.0
        dataset[index, 3] = pr
        dataset[index, 4] = conf
        if id in n2v_spat.wv.key_to_index:
            pr, conf = test_decision_tree(test_set_ids=[str(id)], cls=tree_clos, n2v_model=n2v_spat)
        else:
            pr, conf = row['label'], 1.0
        dataset[index, 5] = pr
        dataset[index, 6] = conf

    mlp = MLP(X_train=dataset, y_train=np.array(train_df['label']))
    mlp.train()
    # return dang_ae, safe_ae, w2v_model, n2v_rel, n2v_clos, tree_rel, tree_clos, mlp


def test(test_df, train_df, w2v_model, dang_ae, safe_ae, tree_rel, tree_clos, n2v_rel, n2v_clos, mlp):
    test_df = test_df.reset_index()
    test_set = np.zeros(shape=(len(test_df), 7))

    tok = TextPreprocessing()
    tweets = tok.token_list(test_df['text_cleaned'].tolist())
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
            pr, conf = test_decision_tree(test_set_ids=[str(id)], cls=tree_rel, n2v_model=n2v_rel)
        else:
            pr, conf = pred_missing_info, conf_missing_info
        test_set[index, 3] = pr
        test_set[index, 4] = conf
        if id in n2v_clos.wv.key_to_index:
            pr, conf = test_decision_tree(test_set_ids=[str(id)], cls=tree_clos, n2v_model=n2v_clos)
        else:
            pr, conf = pred_missing_info, conf_missing_info
        test_set[index, 5] = pr
        test_set[index, 6] = conf

    return mlp.test(test_set, np.array(test_df['label']))

def classify_users(job_id, user_ids):
    if not exists(str(job_id)):
        return 400
    else:
        tok = TextPreprocessing()
        w2v_model = WordEmb("", embedding_size=0, window=0, epochs=0)   # The actual values are not important since we will load the model
        dang_ae = load_model("{}/model/autoencoderdang.h5".format(job_id))
        safe_ae = load_model("{}/model/autoencodersafe.h5".format(job_id))
        n2v_rel = Word2Vec.load("{}/model/n2v_rel.h5".format(job_id))
        n2v_clos = Word2Vec.load("{}/model/n2v_clos.h5".format(job_id))
        mlp = load_model("{}/model/mlp.h5".format(job_id))
        tree_rel = load_decision_tree("{}/model/dtree_rel.h5".format(job_id))
        tree_clos = load_decision_tree("{}/model/dtree_clos.h5".format(job_id))
        out = {}
        for id in user_ids:
            if id not in df['id'].tolist():
                out[id] = "not found"
            else:
                pred = predict_user(user=df[df.id==id].reset_index(), w2v_model=w2v_model, dang_ae=dang_ae, tok=tok,
                                    safe_ae=safe_ae, n2v_rel=n2v_rel, n2v_clos=n2v_clos, mlp=mlp, tree_rel=tree_rel,
                                    tree_clos=tree_clos, df=df)
                if round(pred[0][0]) == 0:
                    out[id] = "risky"
                elif round(pred[0][0]) == 1:
                    out[id] = "safe"
        return out


def predict_user(user:pd.DataFrame, w2v_model, dang_ae, tok, safe_ae, n2v_rel, n2v_clos, mlp, tree_rel, tree_clos, df):
    test_array = np.zeros(shape=(1, 7))
    tweet = tok.token_list(user['text_cleaned'].tolist())
    tweets_embs = w2v_model.text_to_vec(tweet)
    pred_dang = dang_ae.predict(tweets_embs)
    pred_safe = safe_ae.predict(tweets_embs)
    tweets_sigmoid = tf.keras.activations.sigmoid(tf.constant(tweets_embs, dtype=tf.float32)).numpy()
    pred_loss_dang = tf.keras.losses.mse(pred_dang, tweets_sigmoid).numpy()
    pred_loss_safe = tf.keras.losses.mse(pred_safe, tweets_sigmoid).numpy()
    label = [1 if i < j else 0 for i, j in zip(pred_loss_dang, pred_loss_safe)]
    test_array[0, 0] = pred_loss_dang
    test_array[0, 1] = pred_loss_safe
    test_array[0, 2] = label[0]

    # If the instance doesn't have information about spatial or social relationships, we will replace the decision tree
    # prediction with the most frequent label in the training set
    pred_missing_info = df['label'].value_counts().argmax()
    conf_missing_info = max(df['label'].value_counts()) / len(df)  # ratio

    id = user['id'].values[0]
    if id in n2v_rel.wv.key_to_index:
        pr, conf = test_decision_tree(test_set_ids=[str(id)], cls=tree_rel, n2v_model=n2v_rel)
    else:
        pr, conf = pred_missing_info, conf_missing_info
    test_array[0, 3] = pr
    test_array[0, 4] = conf
    if id in n2v_clos.wv.key_to_index:
        pr, conf = test_decision_tree(test_set_ids=[str(id)], cls=tree_clos, n2v_model=n2v_clos)
    else:
        pr, conf = pred_missing_info, conf_missing_info
    test_array[0, 5] = pr
    test_array[0, 6] = conf

    pred = mlp.predict(test_array, verbose=0)
    return pred


def cross_validation(dataset_path, n_folds):
    df = pd.read_csv(dataset_path, sep=',')
    X = df
    y = df['label']

    st = StratifiedKFold(n_splits=n_folds)
    folds = st.split(X=X, y=y)
    l = []
    for k, (train_idx, test_idx) in enumerate(folds):
        dang_ae, safe_ae, w2v_model, n2v_rel, n2v_clos, tree_rel, tree_clos, mlp = train(df.iloc[train_idx], k)
        p, r, f1, s = test(test_df=df.iloc[test_idx], train_df=df.iloc[train_idx], dang_ae=dang_ae, safe_ae=safe_ae,
                           tree_rel=tree_rel, tree_clos=tree_clos, n2v_rel=n2v_rel, n2v_clos=n2v_clos, mlp=mlp,
                           w2v_model=w2v_model)
        l.append((p, r, f1, s))
