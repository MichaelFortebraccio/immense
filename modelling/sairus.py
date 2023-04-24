import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
import modelling.mlp
from modelling.text_preprocessing import TextPreprocessing
from modelling.word_embedding import WordEmb
from modelling.ae import AE
from node_classification.reduce_dimension import dimensionality_reduction
from node_classification.decision_tree import *
from utils import create_or_load_post_list, save_to_pickle, load_from_pickle
from gensim.models import Word2Vec
from tqdm import tqdm
from modelling.mlp import MLP
from keras.models import load_model

np.random.seed(123)

def train_w2v_model(train_df, embedding_size, window, epochs, model_dir, dataset_dir):
    """
    Train the Word2Vc model that will be used for learning the embeddings of the content
    """
    tok = TextPreprocessing()
    posts_content = tok.token_list(train_df['text_cleaned'].tolist())
    if not os.path.exists("{}/w2v_{}.pkl".format(model_dir, embedding_size)):
        w2v_model = WordEmb(posts_content, embedding_size=embedding_size, window=window, epochs=epochs, model_dir=model_dir)
        w2v_model.train_w2v()
        save_to_pickle("{}/w2v_{}.pkl".format(model_dir, embedding_size), w2v_model)
    else:
        w2v_model = load_from_pickle("{}/w2v_{}.pkl".format(model_dir, embedding_size))
    # split content in safe and dangerous
    dang_posts = train_df.loc[train_df['label'] == 1]['text_cleaned']
    safe_posts = train_df.loc[train_df['label'] == 0]['text_cleaned']
    list_dang_posts = create_or_load_post_list(path='{}/list_dang_posts_{}.pickle'.format(dataset_dir, embedding_size), w2v_model=w2v_model,
                                               tokenized_list=tok.token_list(dang_posts))
    list_safe_posts = create_or_load_post_list(path='{}/list_safe_posts_{}.pickle'.format(dataset_dir, embedding_size), w2v_model=w2v_model,
                                               tokenized_list=tok.token_list(safe_posts))
    list_embs = w2v_model.text_to_vec(posts_content)
    return list_dang_posts, list_safe_posts, list_embs, w2v_model


def learn_mlp(train_df, content_embs, dang_ae, safe_ae, tree_rel, tree_spat, spat_node_embs, rel_node_embs, id2idx_spat_path: dict, id2idx_rel_path: dict, model_dir):
    """
    Train the MLP aimed at fusing the models
    Args:
        train_df: Dataframe with the IDs of the users in the training set
        content_embs: np array containing the word embeddings of the content posted by the users
        dang_ae: Dangerous autoencoder model
        safe_ae: Safe autoencoder model
        tree_rel: Relational decision tree
        tree_spat: Spatial decision tree
        spat_node_embs: Spatial node embeddings
        rel_node_embs: Relational node embeddings
        id2idx_spat_path: Path to the dictionary having as keys the user IDs and as value their index in the spatial adjacency matrix
        id2idx_rel_path: Path to the dictionary having as keys the user IDs and as value their index in the relational adjacency matrix
        model_dir:
    Returns: The learned MLP
    """
    id2idx_rel = load_from_pickle(id2idx_rel_path)
    id2idx_spat = load_from_pickle(id2idx_spat_path)

    dataset = np.zeros((len(train_df), 7))
    prediction_dang = dang_ae.predict(content_embs)
    prediction_safe = safe_ae.predict(content_embs)

    posts_sigmoid = tf.keras.activations.sigmoid(tf.constant(content_embs, dtype=tf.float32)).numpy()      # Apply the sigmoid to the posts and make them comparable with the autoencoder predictions (the autoencoder uses the sigmoid activation function)
    prediction_loss_dang = tf.keras.losses.mse(prediction_dang, posts_sigmoid).numpy()
    prediction_loss_safe = tf.keras.losses.mse(prediction_safe, posts_sigmoid).numpy().tolist()
    labels = [1 if i < j else 0 for i, j in zip(prediction_loss_dang, prediction_loss_safe)]
    dataset[:, 0] = prediction_loss_dang
    dataset[:, 1] = prediction_loss_safe
    dataset[:, 2] = np.array(labels)

    for index, row in train_df.iterrows():
        id = row['id']
        if id in id2idx_rel.keys():
            idx = id2idx_rel[id]
            dtree_input = np.expand_dims(rel_node_embs[idx], axis=0)
            pr, conf = test_decision_tree(test_set=dtree_input, cls=tree_rel)
        else:
            pr, conf = row['label'], 1.0
        dataset[index, 3] = pr
        dataset[index, 4] = conf
        if id in id2idx_spat.keys():
            idx = id2idx_spat[id]
            try:
                pr, conf = test_decision_tree(test_set=np.expand_dims(spat_node_embs[idx], axis=0), cls=tree_spat)
            except IndexError:
                print("error")
        else:
            pr, conf = row['label'], 1.0
        dataset[index, 5] = pr
        dataset[index, 6] = conf

    mlp = MLP(X_train=dataset, y_train=np.array(train_df['label']), model_dir=model_dir)
    mlp.train()
    return mlp


def train(train_df, full_df, dataset_dir, model_dir, word_embedding_size, window, w2v_epochs, rel_node_emb_technique:str,
          spat_node_emb_technique:str, rel_node_embedding_size, spat_node_embedding_size, rel_path=None, spatial_path=None,
          n_of_walks_spat=None, n_of_walks_rel=None, walk_length_spat=None, walk_length_rel=None, p_spat=None, p_rel=None,
          q_spat=None, q_rel=None, n2v_epochs_spat=None, n2v_epochs_rel=None, spat_ae_epochs=None, rel_ae_epochs=None,
          adj_matrix_spat_path=None, adj_matrix_rel_path=None, id2idx_rel_path=None, id2idx_spat_path=None):
    """
    Builds and trains the independent modules that analyze content, social relationships and spatial relationships, and then fuses them with the MLP
    Args:
        train_df: Dataframe with the posts used for the MLP training
        full_df: Dataframe containing the whole set of posts. It is used for training the node embedding models, since
        the setting is transductive, hence we need to know in advance information about all the nodes
        dataset_dir: Directory containing the dataset
        model_dir: Directory where the models will be saved
        word_embedding_size: size of the word embeddings that will be created
        window:
        w2v_epochs:
        rel_node_emb_technique: technique to adopt for learning relational node embeddings
        spat_node_emb_technique: technique to adopt for learning spatial node embeddings
        rel_node_embedding_size: Dimension of the node embeddings that will be created from the relational network
        spat_node_embedding_size: Dimension of the node embeddings that will be created from the spatial network
        rel_path: Path to the file stating the social relationships among the users
        spatial_path: Path to the file stating the spatial relationships among the users
        n_of_walks_spat: n2v
        n_of_walks_rel: n2v
        walk_length_spat: n2v
        walk_length_rel: n2v
        p_spat: n2v
        p_rel: n2v
        q_spat: n2v
        q_rel: n2v
        n2v_epochs_spat: n2v
        n2v_epochs_rel: n2v
        rel_ae_epochs: autoencoder
        spat_ae_epochs: autoencoder
        adj_matrix_spat: pca, none, autoencoder
        adj_matrix_rel: pca, none, autoencoder
        id2idx_rel: matching between the node IDs and their index in the relational adj matrix. pca, autoencoder
        id2idx_spat: matching between the node IDs and their index in the spatial adj matrix. pca, autoencoder
    Returns:
    """
    list_dang_posts, list_safe_posts, list_content_embs, w2v_model = train_w2v_model(train_df=train_df, embedding_size=word_embedding_size, window=window,
                                                         epochs=w2v_epochs, model_dir=model_dir, dataset_dir=dataset_dir)


    ################# TRAIN AND LOAD SAFE AND DANGEROUS AUTOENCODER ####################
    dang_ae = AE(X_train=list_dang_posts, name='autoencoderdang', model_dir=model_dir, epochs=100, batch_size=128, lr=0.05).train_autoencoder_content()
    safe_ae = AE(X_train=list_safe_posts, name='autoencodersafe', model_dir=model_dir, epochs=100, batch_size=128, lr=0.05).train_autoencoder_content()
    ################# TRAIN OR LOAD DECISION TREES ####################
    model_dir_rel = "{}/node_embeddings/rel/{}".format(model_dir, rel_node_emb_technique)
    model_dir_spat = "{}/node_embeddings/spat/{}".format(model_dir, spat_node_emb_technique)
    try:
        os.makedirs(model_dir_rel, exist_ok=False)
        os.makedirs(model_dir_spat, exist_ok=False)
    except OSError:
        pass
    rel_tree_path = "{}/dtree.h5".format(model_dir_rel)
    spat_tree_path = "{}/dtree.h5".format(model_dir_spat)

    train_set_rel, train_set_labels_rel = dimensionality_reduction(rel_node_emb_technique, model_dir=model_dir_rel, edge_path=rel_path,
                                                                   n_of_walks=n_of_walks_rel, walk_length=walk_length_rel, lab="rel", epochs=rel_ae_epochs,
                                                                   node_embedding_size=rel_node_embedding_size, p=p_rel, q=q_rel, id2idx_path=id2idx_rel_path,
                                                                   n2v_epochs=n2v_epochs_rel, train_df=full_df, adj_matrix_path=adj_matrix_rel_path)

    train_set_spat, train_set_labels_spat = dimensionality_reduction(spat_node_emb_technique, model_dir=model_dir_spat, edge_path=spatial_path,
                                                                     n_of_walks=n_of_walks_spat, walk_length=walk_length_spat, epochs=spat_ae_epochs,
                                                                     node_embedding_size=spat_node_embedding_size, p=p_spat, q=q_spat, lab="spat",
                                                                     n2v_epochs=n2v_epochs_spat, train_df=full_df, adj_matrix_path=adj_matrix_spat_path, id2idx_path=id2idx_spat_path)

    train_decision_tree(train_set=train_set_rel, save_path=rel_tree_path, train_set_labels=train_set_labels_rel, name="rel")
    train_decision_tree(train_set=train_set_spat, save_path=spat_tree_path, train_set_labels=train_set_labels_spat, name="spat")

    tree_rel = load_decision_tree(rel_tree_path)
    tree_spat = load_decision_tree(spat_tree_path)

    ################# NOW THAT WE HAVE THE MODELS WE CAN OBTAIN THE TRAINING SET FOR THE MLP #################
    if rel_node_emb_technique == "node2vec":
        mod = Word2Vec.load("{}/n2v_rel.h5".format(model_dir_rel))
        d = mod.wv.key_to_index
        id2idx_rel = {int(k): d[k] for k in d.keys()}
    if spat_node_emb_technique == "node2vec":
        mod = Word2Vec.load("{}/n2v_spat.h5".format(model_dir_spat))
        d = mod.wv.key_to_index
        id2idx_spat = {int(k): d[k] for k in d.keys()}
    mlp = learn_mlp(train_df=train_df, content_embs=list_content_embs, dang_ae=dang_ae, safe_ae=safe_ae, tree_rel=tree_rel, tree_spat=tree_spat,
                    rel_node_embs=train_set_rel, spat_node_embs=train_set_spat, model_dir=model_dir, id2idx_rel=id2idx_rel, id2idx_spat=id2idx_spat)
    save_to_pickle("{}/mlp.pkl".format(model_dir), mlp)

    # return dang_ae, safe_ae, w2v_model, mlp, model_dir_rel, model_dir_spat

def predict_user(user: pd.DataFrame, w2v_model, dang_ae, safe_ae, df, tree_rel, tree_spat, mlp: modelling.mlp.MLP, rel_node_emb_technique, spat_node_emb_technique,
                 id2idx_rel=None, id2idx_spat=None, n2v_rel=None, n2v_spat=None, pca_rel=None, pca_spat=None, ae_rel=None, ae_spat=None, adj_matrix_rel=None, adj_matrix_spat=None):
    test_array = np.zeros(shape=(1, 7))
    posts = user['text_cleaned'].values[0].split(" ")
    posts_embs = w2v_model.text_to_vec([posts])
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

    id = user['id']

    if str(id) in id2idx_rel.keys():
        idx = id2idx_rel[id]
        dtree_input = get_testset(rel_node_emb_technique, idx, adj_matrix=adj_matrix_rel, n2v=n2v_rel, pca=pca_rel, ae=ae_rel)
        pr_rel, conf_rel = test_decision_tree(test_set=dtree_input, cls=tree_rel)
    else:
        pr_rel, conf_rel = pred_missing_info, conf_missing_info
    test_array[0, 3] = pr_rel
    test_array[0, 4] = conf_rel
    if str(id) in id2idx_spat.keys():
        idx = id2idx_spat[id]
        dtree_input = get_testset(spat_node_emb_technique, idx, adj_matrix=adj_matrix_spat, n2v=n2v_spat, pca=pca_spat, ae=ae_spat)
        pr_spat, conf_spat = test_decision_tree(test_set=dtree_input, cls=tree_spat)
    else:
        pr_spat, conf_spat = pred_missing_info, conf_missing_info
    test_array[0, 5] = pr_spat
    test_array[0, 6] = conf_spat

    pred = mlp.model.predict(test_array, verbose=0)
    if round(pred[0][0]) == 0:
        return 1
    elif round(pred[0][0]) == 1:
        return 0

def get_testset(node_emb_technique, idx, adj_matrix=None, n2v=None, pca=None, ae=None):
    """
    Depending on the node embedding technique adopted, provide the processed array that will then be used by the decision tree
    """
    if node_emb_technique == "node2vec":
        # if str(id) in n2v_rel.wv.key_to_index:
        mod = n2v.wv
        test_set = mod.vectors[mod.key_to_index[idx]]
        test_set = np.expand_dims(test_set, axis=0)
    elif node_emb_technique == "pca":
        row = adj_matrix[idx]
        row = np.expand_dims(row, axis=0)
        test_set = pca.transform(row)
        #test_set = np.expand_dims(test_set, axis=0)
    elif node_emb_technique == "autoencoder":
        row = adj_matrix[idx]
        row = np.expand_dims(row, axis=0)
        test_set = ae.predict(row)
    else:
        test_set = np.expand_dims(adj_matrix[idx], axis=0)
    return test_set


def test(rel_node_emb_technique, spat_node_emb_technique, test_df, train_df, w2v_model, dang_ae, safe_ae, tree_rel, tree_spat, mlp: modelling.mlp.MLP, id2idx_rel=None,
         id2idx_spat=None, n2v_rel=None, n2v_spat=None, pca_rel=None, pca_spat=None, ae_rel=None, ae_spat=None, adj_matrix_spat=None, adj_matrix_rel=None):
    test_set = np.zeros(shape=(len(test_df), 7))

    tok = TextPreprocessing()
    posts = tok.token_list(test_df['text_cleaned'].tolist())
    test_posts_embs = w2v_model.text_to_vec(posts)
    pred_dang = dang_ae.predict(test_posts_embs)
    pred_safe = safe_ae.predict(test_posts_embs)
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
        if id in id2idx_rel.keys():
            idx = id2idx_rel[id]
            dtree_input = get_testset(rel_node_emb_technique, idx, adj_matrix=adj_matrix_rel, n2v=n2v_rel, pca=pca_rel, ae=ae_rel)
            pr_rel, conf_rel = test_decision_tree(test_set=dtree_input, cls=tree_rel)
        else:
            pr_rel, conf_rel = pred_missing_info, conf_missing_info
        if id in id2idx_spat.keys():
            idx = id2idx_spat[id]
            dtree_input = get_testset(spat_node_emb_technique, idx, adj_matrix=adj_matrix_spat, n2v=n2v_spat, pca=pca_spat, ae=ae_spat)
            pr_spat, conf_spat = test_decision_tree(test_set=dtree_input, cls=tree_spat)
        else:
            pr_spat, conf_spat = pred_missing_info, conf_missing_info

        test_set[index, 3] = pr_rel
        test_set[index, 4] = conf_rel
        test_set[index, 5] = pr_spat
        test_set[index, 6] = conf_spat
    return mlp.test(test_set, np.array(test_df['label']))

####### THESE FUNCTIONS ARE CURRENTLY NOT USED #######
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
            pred = predict_user(user=df[df.id==id].reset_index(), w2v_model=w2v_model, dang_ae=dang_ae,
                                safe_ae=safe_ae, n2v_rel=n2v_rel, n2v_spat=n2v_spat, mlp=mlp, tree_rel=tree_rel,
                                tree_spat=tree_spat, df=df)
            if round(pred[0][0]) == 0:
                out[id] = "safe"
            elif round(pred[0][0]) == 1:
                out[id] = "risky"
    return out
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


