import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
from os.path import exists
import modelling.mlp
from modelling.text_preprocessing import TextPreprocessing
from modelling.word_embedding import WordEmb
from modelling.ae import AE
from node_classification.graph_embeddings.node2vec import Node2VecEmbedder
from node_classification.reduce_dimension import dimensionality_reduction
from node_classification.decision_tree import *
from utils import create_or_load_post_list
from tqdm import tqdm
from modelling.mlp import MLP
from keras.models import load_model

np.random.seed(123)

def train_w2v_model(train_df, embedding_size, window, epochs, model_dir, dataset_dir):
    tok = TextPreprocessing()
    posts_content = tok.token_list(train_df['text_cleaned'].tolist())
    w2v_model = WordEmb(posts_content, embedding_size=embedding_size, window=window, epochs=epochs, model_dir=model_dir)

    # split content in safe and dangerous
    dang_posts = train_df.loc[train_df['label'] == 1]['text_cleaned']
    safe_posts = train_df.loc[train_df['label'] == 0]['text_cleaned']

    w2v_model.train_w2v()

    list_dang_posts = create_or_load_post_list(path='{}/list_dang_posts.pickle'.format(dataset_dir), w2v_model=w2v_model,
                                               tokenized_list=tok.token_list(dang_posts))
    list_safe_posts = create_or_load_post_list(path='{}/list_safe_posts.pickle'.format(dataset_dir), w2v_model=w2v_model,
                                               tokenized_list=tok.token_list(safe_posts))
    list_embs = w2v_model.text_to_vec(posts_content)
    return list_dang_posts, list_safe_posts, list_embs, w2v_model


def learn_mlp(train_df, list_embs, dang_ae, safe_ae, n2v_rel, n2v_spat, tree_rel, tree_spat, model_dir):
    dataset = np.zeros((len(train_df), 7))
    prediction_dang = dang_ae.predict(list_embs)
    prediction_safe = safe_ae.predict(list_embs)

    posts_sigmoid = tf.keras.activations.sigmoid(tf.constant(list_embs, dtype=tf.float32)).numpy()
    prediction_loss_dang = tf.keras.losses.mse(prediction_dang, posts_sigmoid).numpy()
    prediction_loss_safe = tf.keras.losses.mse(prediction_safe, posts_sigmoid).numpy().tolist()
    labels = [1 if i < j else 0 for i, j in zip(prediction_loss_dang, prediction_loss_safe)]
    dataset[:, 0] = prediction_loss_dang
    dataset[:, 1] = prediction_loss_safe
    dataset[:, 2] = np.array(labels)

    for index, row in train_df.iterrows():
        id = row['id']
        if id in n2v_rel.wv.key_to_index:
            pr, conf = test_decision_tree(test_set_ids=[str(id)], cls=tree_rel, n2v_model=n2v_rel)
        else:
            pr, conf = row['label'], 1.0
        dataset[index, 3] = pr
        dataset[index, 4] = conf
        if id in n2v_spat.wv.key_to_index:
            pr, conf = test_decision_tree(test_set_ids=[str(id)], cls=tree_spat, n2v_model=n2v_spat)
        else:
            pr, conf = row['label'], 1.0
        dataset[index, 5] = pr
        dataset[index, 6] = conf

    mlp = MLP(X_train=dataset, y_train=np.array(train_df['label']), model_dir=model_dir)
    mlp.train()
    return mlp

def train(train_df, dataset_dir, model_dir, rel_path, spatial_path, word_embedding_size, window, w2v_epochs, node_emb_technique,
          rel_node_embedding_size, spat_node_embedding_size, n_of_walks_spat, n_of_walks_rel, walk_length_spat,
          walk_length_rel, p_spat, p_rel, q_spat, q_rel, n2v_epochs_spat, n2v_epochs_rel):

    list_dang_posts, list_safe_posts, list_embs, w2v_model = train_w2v_model(train_df=train_df, embedding_size=word_embedding_size, window=window,
                                                         epochs=w2v_epochs, model_dir=model_dir, dataset_dir=dataset_dir)

    ################# TRAIN AND LOAD SAFE AND DANGEROUS AUTOENCODER ####################
    dang_ae = AE(input_len=word_embedding_size, X_train=list_dang_posts, label='dang', model_dir=model_dir).load_autoencoder()  # .train_autoencoder()
    safe_ae = AE(input_len=word_embedding_size, X_train=list_safe_posts, label='safe', model_dir=model_dir).load_autoencoder()

    ################# TRAIN OR LOAD DECISION TREES ####################
    rel_tree_path = "{}/dtree_rel.h5".format(model_dir)
    spat_tree_path = "{}/dtree_spat.h5".format(model_dir)
    train_set_rel, train_set_labels_rel = dimensionality_reduction(node_emb_technique, model_dir=model_dir, edge_path=rel_path, n_of_walks=n_of_walks_rel,
                                                                   walk_length=walk_length_rel, node_embedding_size=rel_node_embedding_size,
                                                                   p=p_rel, q=q_rel, n2v_epochs=n2v_epochs_rel, train_df=train_df)
    train_set_spat, train_set_labels_spat = dimensionality_reduction(node_emb_technique, model_dir=model_dir, edge_path=spatial_path, n_of_walks=n_of_walks_spat,
                                                                   walk_length=walk_length_spat, node_embedding_size=spat_node_embedding_size,
                                                                   p=p_spat, q=q_spat, n2v_epochs=n2v_epochs_spat, train_df=train_df)

    if not exists(rel_tree_path):
        train_decision_tree(train_set=train_set_rel, save_path=rel_tree_path, train_set_labels=train_set_labels_rel, name="rel")

    if not exists(spat_tree_path):
        train_decision_tree(train_set=train_set_spat, save_path=spat_tree_path, train_set_labels=train_set_labels_spat, name="spatial")


    tree_rel = load_decision_tree(rel_tree_path)
    tree_spat = load_decision_tree(spat_tree_path)

    ################# NOW THAT WE HAVE THE MODELS WE CAN OBTAIN THE TRAINING SET FOR THE MLP #################
    mlp = learn_mlp(train_df=train_df, list_embs=list_embs, dang_ae=dang_ae, safe_ae=safe_ae, n2v_rel=n2v_rel, n2v_spat=n2v_spat, tree_rel=tree_rel, tree_spat=tree_spat, model_dir=model_dir)
    return dang_ae, safe_ae, w2v_model, n2v_rel, n2v_spat, tree_rel, tree_spat, mlp

def classify_users(job_id, user_ids, CONTENT_FILENAME, model_dir):
    df = pd.read_csv("{}/dataset/{}".format(job_id, CONTENT_FILENAME))
    tok = TextPreprocessing()
    w2v_model = WordEmb("", embedding_size=0, window=0, epochs=0, model_dir=model_dir)   # The actual values are not important since we will load the model. Only the model dir is important
    dang_ae = load_model("{}/autoencoderdang.h5".format(model_dir))
    safe_ae = load_model("{}/autoencodersafe.h5".format(model_dir))
    n2v_rel = Word2Vec.load("{}/n2v_rel.h5".format(model_dir))
    n2v_spat = Word2Vec.load("{}/n2v_spat.h5".format(model_dir))
    mlp = load_model("{}/mlp.h5".format(model_dir))
    tree_rel = load_decision_tree("{}/dtree_rel.h5".format(model_dir))
    tree_spat = load_decision_tree("{}/dtree_spat.h5".format(model_dir))
    out = {}
    for id in user_ids:
        if id not in df['id'].tolist():
            out[id] = "not found"
        else:
            pred = predict_user(user=df[df.id==id].reset_index(), w2v_model=w2v_model, dang_ae=dang_ae, tok=tok,
                                safe_ae=safe_ae, n2v_rel=n2v_rel, n2v_spat=n2v_spat, mlp=mlp, tree_rel=tree_rel,
                                tree_spat=tree_spat, df=df)
            if round(pred[0][0]) == 0:
                out[id] = "risky"
            elif round(pred[0][0]) == 1:
                out[id] = "safe"
    return out

def predict_user(user: pd.DataFrame, w2v_model, dang_ae, tok, safe_ae, n2v_rel, n2v_spat, mlp, tree_rel, tree_spat, df):
    test_array = np.zeros(shape=(1, 7))
    posts = tok.token_list(user['text_cleaned'].tolist())
    posts_embs = w2v_model.text_to_vec(posts)
    pred_dang = dang_ae.predict(posts_embs)
    pred_safe = safe_ae.predict(posts_embs)
    posts_sigmoid = tf.keras.activations.sigmoid(tf.constant(posts_embs, dtype=tf.float32)).numpy()
    pred_loss_dang = tf.keras.losses.mse(pred_dang, posts_sigmoid).numpy()
    pred_loss_safe = tf.keras.losses.mse(pred_safe, posts_sigmoid).numpy()
    label = [1 if i < j else 0 for i, j in zip(pred_loss_dang, pred_loss_safe)]
    test_array[0, 0] = pred_loss_dang
    test_array[0, 1] = pred_loss_safe
    test_array[0, 2] = label[0]

    # If the instance doesn't have information about spatial or social relationships, we will replace the decision tree
    # prediction with the most frequent label in the training set
    pred_missing_info = df['label'].value_counts().argmax()
    conf_missing_info = max(df['label'].value_counts()) / len(df)  # ratio

    id = user['id'].values[0]
    if str(id) in n2v_rel.wv.key_to_index:
        pr, conf = test_decision_tree(test_set_ids=[str(id)], cls=tree_rel, n2v_model=n2v_rel)
    else:
        pr, conf = pred_missing_info, conf_missing_info
    test_array[0, 3] = pr
    test_array[0, 4] = conf
    if str(id) in n2v_spat.wv.key_to_index:
        pr, conf = test_decision_tree(test_set_ids=[str(id)], cls=tree_spat, n2v_model=n2v_spat)
    else:
        pr, conf = pred_missing_info, conf_missing_info
    test_array[0, 5] = pr
    test_array[0, 6] = conf

    pred = mlp.predict(test_array, verbose=0)
    return pred

####### THE FOLLOWING FUNCTIONS ARE CURRENTLY NOT USED IN THE API #######

def test(test_df, train_df, w2v_model, dang_ae, safe_ae, tree_rel, tree_spat, n2v_rel, n2v_spat, mlp:modelling.mlp.MLP):
    test_set = np.zeros(shape=(len(test_df), 7))

    tok = TextPreprocessing()
    posts = tok.token_list(test_df['text_cleaned'].tolist())
    test_posts_embs = w2v_model.text_to_vec(posts)
    pred_dang = dang_ae.predict(test_posts_embs)
    pred_safe = safe_ae.predict(test_posts_embs)
    print("Prediction autoencoders")
    test_posts_sigmoid = tf.keras.activations.sigmoid(tf.constant(test_posts_embs, dtype=tf.float32)).numpy()
    pred_loss_dang = tf.keras.losses.mse(pred_dang, test_posts_sigmoid).numpy()
    pred_loss_safe = tf.keras.losses.mse(pred_safe, test_posts_sigmoid).numpy()
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
        if str(id) in n2v_rel.wv.key_to_index:
            pr, conf = test_decision_tree(test_set_ids=[str(id)], cls=tree_rel, n2v_model=n2v_rel)
        else:
            pr, conf = pred_missing_info, conf_missing_info
        test_set[index, 3] = pr
        test_set[index, 4] = conf
        if str(id) in n2v_spat.wv.key_to_index:
            pr, conf = test_decision_tree(test_set_ids=[str(id)], cls=tree_spat, n2v_model=n2v_spat)
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
        dang_ae, safe_ae, w2v_model, n2v_rel, n2v_spat, tree_rel, tree_spat, mlp = train(df.iloc[train_idx], k)
        p, r, f1, s = test(test_df=df.iloc[test_idx], train_df=df.iloc[train_idx], dang_ae=dang_ae, safe_ae=safe_ae,
                           tree_rel=tree_rel, tree_spat=tree_spat, n2v_rel=n2v_rel, n2v_spat=n2v_spat, mlp=mlp,
                           w2v_model=w2v_model)
        l.append((p, r, f1, s))



