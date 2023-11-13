import numpy as np
import pandas as pd
import tensorflow as tf
from torch.optim import SGD, Adam
from exceptions import Id2IdxException, MissingParamException
from gensim.models import Word2Vec
from keras.models import load_model
from modelling.ae import AE
from modelling.mlp import MLP
from modelling.text_preprocessing import TextPreprocessing
from modelling.word_embedding import WordEmb
from node_classification.decision_tree import *
from node_classification.graph_embeddings.sage import create_graph, create_mappers
from node_classification.reduce_dimension import reduce_dimension
from torch.nn import MSELoss
from os.path import exists, join
import torch
from os import makedirs
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from utils import load_from_pickle, save_to_pickle
np.random.seed(123)


def train_w2v_model(embedding_size, epochs, id_field_name, model_dir, text_field_name, train_df):
    """
    Train the Word2Vc model that will be used for learning the embeddings of the content.
    :param embedding_size:
    :param epochs:
    :param id_field_name: Name of the field containing the ID in the training dataframe
    :param model_dir: Directory where the word embedding model will be saved
    :param text_field_name: Name of the field containing the text in the training dataframe
    :param train_df: training dataframe
    :return: dang_posts_array: Array of shape [n_dang_users, embedding_size] with the embeddings of the dangerous users
    :return: safe_posts_array: Array of shape [n_safe_users, embedding_size] with the embeddings of the safe users
    :return: users_embeddings (although the variable name is safe_users_embeddings): Dictionary having as keys the
        users' IDs and, for each of them, the embedding array given by the sum of the words in their posts
    """
    tok = TextPreprocessing()
    posts_content = tok.token_list(text_field_name=text_field_name, df=train_df)
    name = "w2v_{}.pkl".format(embedding_size)
    if not exists(join(model_dir, name)):
        print("Training word2vec model")
        w2v_model = WordEmb(posts_content, embedding_size=embedding_size, window=10, epochs=epochs, model_dir=model_dir)
        w2v_model.train_w2v()
        save_to_pickle(join(model_dir, name), w2v_model)
    else:
        print("Loading word2vec model")
        w2v_model = load_from_pickle(join(model_dir, name))
    # split content in safe and dangerous
    all_users_tokens = tok.token_dict(train_df, text_field_name=text_field_name, id_field_name=id_field_name)
    all_users_embeddings = w2v_model.text_to_vec(users=all_users_tokens)  # Get a dict of all the embeddings of each user, keeping the association with the key
    # dang_posts_array, safe_posts_array,
    return all_users_embeddings


def learn_mlp(ae_dang, ae_safe, content_embs, id2idx_rel, id2idx_spat, model_dir, node_embs_rel, node_embs_spat,
              tec_rel, tec_spat, train_df, tree_rel, tree_spat, y_train, consider_rel=True, consider_spat=True,
              n2v_rel=None, n2v_spat=None):
    """
    Train the MLP aimed at fusing the models
    Args:
        train_df: Dataframe with the IDs of the users in the training set
        ae_dang: Dangerous autoencoder model
        ae_safe: Safe autoencoder model
        content_embs: torch tensor containing the word embeddings of the content posted by the users. The features are z-score normalized
        id2idx_rel: Dictionary having as keys the user IDs and as value their index in the relational adjacency matrix
        id2idx_spat: Dictionary having as keys the user IDs and as value their index in the spatial adjacency matrix
        model_dir:
        node_embs_rel: Relational node embeddings
        node_embs_spat: Spatial node embeddings
        tec_rel: Relational node embedding technique
        tec_spat: Spatial node embedding technique
        train_df: Training dataframe needed to get the predictions for the node embeddings when using n2v since it keeps
        the mapping between node ID and label
        tree_rel: Relational decision tree
        tree_spat: Spatial decision tree
        y_train: Train labels
        n2v_spat: spatial node2vec model
        n2v_rel: relational node2vec model
    Returns: The learned MLP
    """
    dataset = torch.zeros((content_embs.shape[0], 7))
    prediction_dang = ae_dang.predict(content_embs)
    prediction_safe = ae_safe.predict(content_embs)

    loss = MSELoss()
    prediction_loss_dang = []
    prediction_loss_safe = []
    for i in range(content_embs.shape[0]):
        prediction_loss_dang.append(loss(content_embs[i], prediction_dang[i]))
        prediction_loss_safe.append(loss(content_embs[i], prediction_safe[i]))
    labels = [1 if i < j else 0 for i, j in zip(prediction_loss_dang, prediction_loss_safe)]
    dataset[:, 0] = torch.tensor(prediction_loss_dang, dtype=torch.float32)
    dataset[:, 1] = torch.tensor(prediction_loss_safe, dtype=torch.float32)
    dataset[:, 2] = torch.tensor(labels, dtype=torch.float32)

    cmi = 1.0
    if consider_rel:
        rel_part = get_relational_preds(technique=tec_rel, df=train_df, node_embs=node_embs_rel, tree=tree_rel,
                                        id2idx=id2idx_rel, n2v=n2v_rel, cmi=cmi)
        dataset[:, 3], dataset[:, 4] = rel_part[:, 0], rel_part[:, 1]
    if consider_spat:
        spat_part = get_relational_preds(technique=tec_spat, df=train_df, node_embs=node_embs_spat, tree=tree_spat,
                                         id2idx=id2idx_spat, n2v=n2v_spat, cmi=cmi)
        dataset[:, 5], dataset[:, 6] = spat_part[:, 0], spat_part[:, 1]

    #############   IMPORTANTE  #############
    #### STO TRAINANDO IL MODELLO 3%      ###
    #### CONSIDERANDO SOLO IL TESTO       ###
    #########################################
    mlp = MLP(X_train=dataset, y_train=y_train, model_dir=model_dir)
    optim = Adam(mlp.parameters(), lr=0.003, weight_decay=1e-4)
    mlp.train_mlp(optim)


def get_relational_preds(technique, df, tree, node_embs, id2idx: dict, n2v, cmi, pmi=None):
    dataset = torch.zeros(node_embs.shape[0], 2)
    if technique == "graphsage":
        pr, conf = test_decision_tree(test_set=node_embs, cls=tree)
        dataset[:, 0] = torch.tensor(pr, dtype=torch.float32)
        dataset[:, 1] = torch.tensor(conf, dtype=torch.float32)
    else:
        for index, row in tqdm(df.iterrows()):
            id = row['id']
            if technique == "node2vec":
                try:
                    dtree_input = np.expand_dims(n2v.wv[str(id)], axis=0)
                    pr, conf = test_decision_tree(test_set=dtree_input, cls=tree)
                except KeyError:
                    if not pmi:
                        pmi = row['label']
                    pr, conf = pmi, cmi
            else:
                if id in id2idx.keys():
                    idx = id2idx[id]
                    dtree_input = np.expand_dims(node_embs[idx], axis=0)
                    pr, conf = test_decision_tree(test_set=dtree_input, cls=tree)
                else:
                    if not pmi:
                        pmi = row['label']
                    pr, conf = pmi, cmi
            dataset[index, 3] = pr
            dataset[index, 4] = conf
            dataset[index, 5] = pr
            dataset[index, 6] = conf
    return dataset


def train(field_name_id, field_name_text, model_dir, node_emb_technique_rel: str, node_emb_technique_spat: str,
          node_emb_size_rel, node_emb_size_spat, path_safe_ids, path_dang_ids, train_df, w2v_epochs, word_emb_size,
          adj_matrix_path_rel=None, adj_matrix_path_spat=None, batch_size=None, consider_rel=True, consider_spat=True,
          eps_nembs_rel=None, eps_nembs_spat=None, id2idx_path_rel=None, id2idx_path_spat=None, path_rel=None,
          path_spat=None):
    """
    Builds and trains the independent modules that analyze content, social relationships and spatial relationships, and
    then fuses them with the MLP
    :param field_name_id: Name of the field containing the id
    :param field_name_text: Name of the field containing the text
    :param model_dir: Directory where the models will be saved
    :param node_emb_technique_rel: Technique to adopt for learning relational node embeddings
    :param node_emb_technique_spat: Technique to adopt for learning spatial node embeddings
    :param node_emb_size_rel: Dimension of the relational node embeddings to learn
    :param node_emb_size_spat: Dimension of the spatial node embeddings to learn
    :param train_df: Dataframe with the posts used for the MLP training
    :param w2v_epochs:
    :param word_emb_size: Dimension of the word embeddings to create
    :param adj_matrix_path_rel: Path to the relational adj matrix (pca, none, autoencoder)
    :param adj_matrix_path_spat: Path to the spatial adj matrix (pca, none, autoencoder)
    :param batch_size:
    :param eps_nembs_rel: Epochs for training the relational node embedding model
    :param eps_nembs_spat: Epochs for training the spatial node embedding model
    :param id2idx_path_rel: Path to the file containing the dictionary that matches the node IDs to their index in the relational adj matrix (graphsage, pca, autoencoder)
    :param id2idx_path_spat: Path to the file containing the dictionary that matches the node IDs to their index in the spatial adj matrix (graphsage, pca, autoencoder)
    :param path_rel: Path to the file stating the social relationships among the users
    :param path_spat: Path to the file stating the spatial relationships among the users
    :return: Nothing, the learned mlp will be saved in the file "mlp.h5" and put in the model directory
    """
    #dang_posts_ids = list(train_df.loc[train_df['label'] == 1][field_name_id])
    #safe_posts_ids = list(train_df.loc[train_df['label'] == 0][field_name_id])
    dang_posts_ids = load_from_pickle(path_dang_ids)
    safe_posts_ids = load_from_pickle(path_safe_ids)
    #dang_posts_array, safe_posts_array,
    users_embs_dict = train_w2v_model(embedding_size=word_emb_size, epochs=w2v_epochs, id_field_name=field_name_id,
                                      model_dir=model_dir, text_field_name=field_name_text, train_df=train_df)

    embs = np.array(list(users_embs_dict.values()))
    mean = embs.mean(axis=0)
    std = embs.std(axis=0)
    keys = list(users_embs_dict.keys())
    zscored_embs = {}
    for k in keys:
        zscored_embs[k] = (users_embs_dict[k] - mean) / std

    dang_users_ar = np.array([zscored_embs[k] for k in keys if k in dang_posts_ids])
    safe_users_ar = np.array([zscored_embs[k] for k in keys if k in safe_posts_ids])

    """together = np.vstack([dang_users_ar, safe_users_ar])
    normalized = (together - together.mean(axis=0))/together.std(axis=0)
    dang_users_ar = torch.tensor(normalized[:dang_users_ar.shape[0]], dtype=torch.float32)
    safe_users_ar_norm = torch.tensor(normalized[dang_users_ar_norm.shape[0]:], dtype=torch.float32)
    normalized = torch.tensor(normalized, dtype=torch.float32)
    y_train = [1] * dang_users_ar_norm.shape[0] + [0] * safe_users_ar_norm.shape[0]"""


    ################# TRAIN AND LOAD SAFE AND DANGEROUS AUTOENCODER ####################
    dang_ae_name = join(model_dir, "autoencoderdang_{}.pkl".format(word_emb_size))
    safe_ae_name = join(model_dir, "autoencodersafe_{}.pkl".format(word_emb_size))
    #content_embs = np.array(list(users_embs_dict.values()))

    if not exists(dang_ae_name):
        dang_ae = AE(X_train=dang_users_ar, epochs=100, batch_size=64, lr=0.002, name=dang_ae_name)
        dang_ae.train_autoencoder_content()
    else:
        dang_ae = load_from_pickle(dang_ae_name)
    if not exists(safe_ae_name):
        safe_ae = AE(X_train=safe_users_ar, epochs=100, batch_size=128, lr=0.002, name=safe_ae_name)
        safe_ae.train_autoencoder_content()
    else:
        safe_ae = load_from_pickle(safe_ae_name)

    ################# TRAIN OR LOAD DECISION TREES ####################
    model_dir_rel = join(model_dir, "node_embeddings", "rel")
    model_dir_spat = join(model_dir, "node_embeddings", "spat")
    try:
        makedirs(model_dir_rel, exist_ok=False)
        makedirs(model_dir_spat, exist_ok=False)
    except OSError:
        pass
    rel_tree_path = join(model_dir_rel, "dtree_{}_{}.h5".format(node_emb_technique_rel, node_emb_size_rel))
    spat_tree_path = join(model_dir_spat, "dtree_{}_{}.h5".format(node_emb_technique_rel, node_emb_size_rel))

    x_rel, y_rel = reduce_dimension(node_emb_technique_rel, model_dir=model_dir_rel, edge_path=path_rel, lab="rel",
                                    id2idx_path=id2idx_path_rel, node_embedding_size=node_emb_size_rel,
                                    train_df=train_df, epochs=eps_nembs_rel, adj_matrix_path=adj_matrix_path_rel,
                                    sizes=[2, 3], features_dict=users_embs_dict, batch_size=batch_size)

    """x_spat, y_spat = reduce_dimension(spat_node_emb_technique, model_dir=model_dir_spat, edge_path=spatial_path, lab="spat",
                                      id2idx_path=id2idx_spat_path, node_embedding_size=spat_node_embedding_size,
                                      train_df=train_df, epochs=spat_nembs_eps, adj_matrix_path=adj_matrix_spat_path,
                                      sizes=[2, 3], features_dict=users_embs_dict, batch_size=batch_size)"""
    if not exists(rel_tree_path):
        train_decision_tree(train_set=x_rel, save_path=rel_tree_path, train_set_labels=y_rel, name="rel")
    """if not exists(spat_tree_path):
        train_decision_tree(train_set=x_spat, save_path=spat_tree_path, train_set_labels=y_spat, name="spat")"""

    tree_rel = load_decision_tree(rel_tree_path)
    #tree_spat = load_decision_tree(spat_tree_path)

    # WE CAN NOW OBTAIN THE TRAINING SET FOR THE MLP
    n2v_rel = None
    n2v_spat = None
    id2idx_rel = None
    id2idx_spat = None
    if node_emb_technique_rel == "node2vec":
        n2v_rel = Word2Vec.load(join(model_dir_rel, "n2v.h5"))
    elif node_emb_technique_rel != "graphsage":
        id2idx_rel = load_from_pickle(id2idx_path_rel)
    if node_emb_technique_spat == "node2vec":
        n2v_spat = Word2Vec.load(join(model_dir_spat, "n2v.h5"))
    elif node_emb_technique_spat != "graphsage":
        id2idx_spat = load_from_pickle(id2idx_path_spat)
    #tree_rel = tree_spat = x_rel = x_spat = n2v_rel = n2v_spat = id2idx_spat = id2idx_rel = None
    tree_spat = x_spat = None
    y_train = list(train_df['label'])
    normalized = torch.tensor(list(zscored_embs.values()), dtype=torch.float32)
    learn_mlp(ae_dang=dang_ae, ae_safe=safe_ae, content_embs=normalized, consider_rel=consider_rel,
              consider_spat=consider_spat, id2idx_rel=id2idx_rel, id2idx_spat=id2idx_spat, model_dir=model_dir,
              node_embs_rel=x_rel, node_embs_spat=x_spat, tec_rel=node_emb_technique_rel,
              tec_spat=node_emb_technique_spat, train_df=train_df, tree_rel=tree_rel, tree_spat=tree_spat,
              y_train=y_train, n2v_rel=n2v_rel, n2v_spat=n2v_spat)


def predict_user(user: pd.DataFrame, w2v_model, dang_ae, safe_ae, df, tree_rel, tree_spat, mlp: MLP, rel_node_emb_technique, spat_node_emb_technique,
                 id2idx_rel=None, id2idx_spat=None, n2v_rel=None, n2v_spat=None, pca_rel=None, pca_spat=None, ae_rel=None, ae_spat=None, adj_matrix_rel=None, adj_matrix_spat=None):
    test_array = np.zeros(shape=(1, 7))
    posts = user['text_cleaned'].values[0].split(" ")
    posts_embs = w2v_model.text_to_vec([posts])
    posts_embs = list(posts_embs.values())[0]
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
    if n2v_rel:
        try:
            dtree_input = np.expand_dims(n2v_rel.wv[str(id)], axis=0)
            pr_rel, conf_rel = test_decision_tree(test_set=dtree_input, cls=tree_rel)
        except KeyError:
            pr_rel, conf_rel = pred_missing_info, conf_missing_info
    else:
        idx = id2idx_rel[id]
        dtree_input = get_testset_dtree(rel_node_emb_technique, idx, adj_matrix=adj_matrix_rel, n2v=n2v_rel, pca=pca_rel, ae=ae_rel)
        pr_rel, conf_rel = test_decision_tree(test_set=dtree_input, cls=tree_rel)

    test_array[0, 3] = pr_rel
    test_array[0, 4] = conf_rel
    if n2v_spat:
        try:
            dtree_input = np.expand_dims(n2v_spat.wv[str(id)], axis=0)
            pr_spat, conf_spat = test_decision_tree(test_set=dtree_input, cls=tree_spat)
        except KeyError:
            pr_spat, conf_spat = pred_missing_info, conf_missing_info
            pass
    else:
        if id in id2idx_spat.keys():
            idx = id2idx_spat[id]
            dtree_input = get_testset_dtree(spat_node_emb_technique, idx, adj_matrix=adj_matrix_spat, n2v=n2v_spat, pca=pca_spat, ae=ae_spat)
            pr_spat, conf_spat = test_decision_tree(test_set=dtree_input, cls=tree_spat)
        else:
            print("missing")
            pr_spat, conf_spat = pred_missing_info, conf_missing_info
    test_array[0, 5] = pr_spat
    test_array[0, 6] = conf_spat
    print(test_array)

    pred = mlp.model.predict(test_array, verbose=0)
    if round(pred[0][0]) == 0:
        return 1
    elif round(pred[0][0]) == 1:
        return 0


def get_testset_dtree(node_emb_technique, idx, adj_matrix=None, n2v=None, pca=None, ae=None):
    """
    Depending on the node embedding technique adopted, provide the array that will be used by the decision tree
    """
    idx = str(idx)
    if node_emb_technique == "node2vec":
        mod = n2v.wv
        test_set = mod.vectors[mod.key_to_index[idx]]
        test_set = np.expand_dims(test_set, axis=0)
    elif node_emb_technique == "pca":
        row = adj_matrix[idx]
        row = np.expand_dims(row, axis=0)
        test_set = pca.transform(row)
    elif node_emb_technique == "autoencoder":
        row = adj_matrix[idx]
        row = np.expand_dims(row, axis=0)
        test_set = ae.predict(row)
    else:
        test_set = np.expand_dims(adj_matrix[idx], axis=0)
    return test_set


def classify_users(preprocess_job_id, train_job_id, user_ids, content_filename, id2idx_rel_fname, id2idx_spat_fname,
                   rel_adj_mat_fname, spat_adj_mat_fname, rel_technique, spat_technique, we_dim):
    dataset_dir = join(preprocess_job_id, "dataset")
    models_dir = join(train_job_id, "models")
    adj_mat_rel_path = join(dataset_dir, rel_adj_mat_fname)
    adj_mat_spat_path = join(dataset_dir, spat_adj_mat_fname)
    id2idx_rel_path = join(dataset_dir, id2idx_rel_fname)
    id2idx_spat_path = join(dataset_dir, id2idx_spat_fname)

    df = pd.read_csv(join(dataset_dir, content_filename))
    w2v_model = load_from_pickle(join(models_dir, "w2v.pkl"))
    dang_ae = load_model(join(models_dir, "autoencoderdang_{}.h5".format(we_dim)))
    safe_ae = load_model(join(models_dir, "autoencodersafe_{}.h5".format(we_dim)))
    mod_dir_rel = join(models_dir, "node_embeddings", "rel", rel_technique)
    mod_dir_spat = join(models_dir, "node_embeddings", "spat", spat_technique)
    tree_rel = load_decision_tree(join(mod_dir_rel, "dtree.h5"))
    tree_spat = load_decision_tree(join(mod_dir_spat, "dtree.h5"))

    n2v_rel, n2v_spat, pca_rel, pca_spat, ae_rel, ae_spat, adj_mat_rel, id2idx_rel, adj_mat_spat, id2idx_spat = get_ne_models(
        rel_technique=rel_technique, spat_technique=spat_technique, adj_mat_rel_path=adj_mat_rel_path, id2idx_rel_path=id2idx_rel_path,
        adj_mat_spat_path=adj_mat_spat_path, id2idx_spat_path=id2idx_spat_path, mod_dir_rel=mod_dir_rel, mod_dir_spat=mod_dir_spat)

    mlp = load_from_pickle(join(models_dir, "mlp.pkl"))
    out = {}
    for id in user_ids:
        if id not in df['id'].tolist():
            out[id] = "not found"
        else:
            pred = predict_user(user=df[df.id==id].reset_index(), w2v_model=w2v_model, dang_ae=dang_ae, safe_ae=safe_ae, n2v_rel=n2v_rel,
                                n2v_spat=n2v_spat, mlp=mlp, tree_rel=tree_rel, tree_spat=tree_spat, df=df, rel_node_emb_technique=rel_technique,
                                spat_node_emb_technique=spat_technique, id2idx_rel=id2idx_rel, id2idx_spat=id2idx_spat, pca_rel=pca_rel,
                                pca_spat=pca_spat, ae_rel=ae_rel, ae_spat=ae_spat, adj_matrix_rel=adj_mat_rel, adj_matrix_spat=adj_mat_spat)
            out[id] = pred
    return out


# THESE FUNCTIONS ARE NOT USED IN THE API #
def test(ae_dang, ae_safe, df, df_train, field_id, field_text, mlp: MLP, ne_technique_rel, ne_technique_spat, tree_rel,
         tree_spat, w2v_model, ae_rel=None, ae_spat=None, adj_matrix_rel=None, adj_matrix_spat=None, consider_rel=True,
         consider_spat=True, id2idx_rel=None, id2idx_spat=None, mod_rel=None, mod_spat=None, pca_rel=None,
         pca_spat=None, rel_net_path=None, spat_net_path=None):
    test_set = torch.zeros(len(df), 7)
    tok = TextPreprocessing()
    posts = tok.token_dict(df, text_field_name=field_text, id_field_name=field_id)
    posts_embs_dict = w2v_model.text_to_vec(posts)
    posts_embs = torch.tensor(list(posts_embs_dict.values()), dtype=torch.float)
    posts_embs_norm = (posts_embs - posts_embs.mean(axis=0))/posts_embs.std(axis=0)
    pred_dang = ae_dang.predict(posts_embs_norm)
    pred_safe = ae_safe.predict(posts_embs_norm)
    loss = MSELoss()
    pred_loss_dang = []
    pred_loss_safe = []
    for i in range(posts_embs.shape[0]):
        pred_loss_dang.append(loss(posts_embs_norm[i], pred_dang[i]))
        pred_loss_safe.append(loss(posts_embs_norm[i], pred_safe[i]))

    labels = [1 if i < j else 0 for i, j in zip(pred_loss_dang, pred_loss_safe)]
    test_set[:, 0] = torch.tensor(pred_loss_dang, dtype=torch.float32)
    test_set[:, 1] = torch.tensor(pred_loss_safe, dtype=torch.float32)
    test_set[:, 2] = torch.tensor(labels, dtype=torch.float32)

    # At test time, if we meet an instance that doesn't have information about relationships or closeness, we will
    # replace the decision tree prediction with the most frequent label in the training set
    pred_missing_info = df_train['label'].value_counts().argmax()
    #conf_missing_info = max(train_df['label'].value_counts()) / len(train_df)  # ratio
    conf_missing_info = 0.5

    sage_rel_embs = None
    sage_spat_embs = None
    inv_map_rel = None
    inv_map_sp = None
    if consider_rel:
        if ne_technique_rel == "graphsage":
            mapper, inv_map_rel = create_mappers(posts_embs_dict)
            graph = create_graph(inv_map=inv_map_rel, weighted=False, features=posts_embs_dict, edg_dir=rel_net_path, df=df)
            with torch.no_grad():
                graph = graph.to(mod_rel.device)
                sage_rel_embs = mod_rel(graph, inference=True).detach().numpy()
        social_part = get_relational_preds(technique=ne_technique_rel, df=df_train, tree=tree_rel,
                                           node_embs=sage_rel_embs,
                                           id2idx=id2idx_rel, n2v=mod_rel, cmi=conf_missing_info, pmi=pred_missing_info)
        test_set[:, 3], test_set[:, 4] = social_part[:, 0], social_part[:, 1]
    if consider_spat:
        if ne_technique_spat == "graphsage":
            mapper, inv_map_sp = create_mappers(posts_embs_dict)
            graph = create_graph(inv_map=inv_map_sp, weighted=True, features=posts_embs_dict, edg_dir=spat_net_path, df=df)
            with torch.no_grad():
                graph = graph.to(mod_rel.device)
                sage_spat_embs = mod_rel(graph).to("cpu").detach().numpy()
            spatial_part = get_graph_based_predictions(df, pmi=pred_missing_info, cmi=conf_missing_info, tree=tree_spat,
                                                       ne_technique=ne_technique_spat, test_set=test_set, mode="spat",
                                                       model=mod_spat, adj_mat=adj_matrix_spat, pca=pca_spat, ae=ae_spat,
                                                       id2idx=id2idx_spat, inv_mapper=inv_map_sp, node_embs=sage_spat_embs)
            test_set[:, 3], test_set[:, 4] = spatial_part[:, 0], spatial_part[:, 1]

    print(mlp.test(test_set, np.array(df['label'])))
    d = "uj"


def get_graph_based_predictions(test_df, pmi, cmi, ne_technique, tree, test_set, mode, model=None, adj_mat=None,
                                pca=None, ae=None, id2idx=None, inv_mapper=None, node_embs=None):
    """
    This function takes care of making the predictions based on the graphs. If the predictions concern the relational
    component, it fills the 3rd and 4th columns of the test set. If the predictions concern the spatial component, the
    5th and 6th columns are filled
    :param net_path: Path to the edgelist containing the test network (graphsage)
    :param embs: User embeddings (graphsage)
    :param test_df:
    :param pmi: What prediction to give in case we don't have relational (or spatial) information for a node
    :param cmi: What confidence to give in case we don't have relational (or spatial) information for a node
    :param ne_technique:
    :param tree: Decision tree
    :param test_set: Numpy array where the function will place its predictions
    :param mode: Either "rel" or "spat"
    :param model: Model (node2vec or graphsage)
    :param adj_mat: Adjacency matrix (autoencoder, none, pca)
    :param pca: Pca model
    :param ae: Autoencoder
    :param id2idx:
    :return:
    """
    i1, i2, weighted = 3, 4, False
    if mode == "spat":
        i1, i2, weighted = 5, 6, True

    if ne_technique == "graphsage":
        if not inv_mapper:
            MissingMapperException

        for index, row in test_df.iterrows():
            id = row.id
            dtree_input = node_embs[inv_mapper[id]]
            pr, conf = test_decision_tree(test_set=dtree_input, cls=tree)
            for index, row in tqdm(test_df.iterrows()):
                id = row['id']
                if id in id2idx.keys():
                    idx = id2idx[id]
                    test_df[index, i1] = pr[idx]
                    test_df[index, i2] = conf[idx]
    else:
        for index, row in tqdm(test_df.iterrows()):
            id = row['id']
            if ne_technique == "node2vec":
                if not model:
                    raise MissingParamException(ne_technique, "model")
                try:
                    dtree_input = np.expand_dims(model.wv[str(id)], axis=0)
                    pr, conf = test_decision_tree(test_set=dtree_input, cls=tree)
                except KeyError:
                    pr, conf = pmi, cmi
            else:
                if not id2idx:
                    raise Id2IdxException(mode)
                if id in id2idx.keys():
                    idx = id2idx[id]
                    dtree_input = get_testset_dtree(ne_technique, idx, adj_matrix=adj_mat, n2v=model, pca=pca, ae=ae)
                    pr, conf = test_decision_tree(test_set=dtree_input, cls=tree)
                else:
                    pr, conf = pmi, cmi
            test_set[index, i1] = pr
            test_set[index, i2] = conf
    return test_set
    #mlp.test(test_set, np.array(test_df['label']))


def cross_validation(dataset_path, n_folds):
    df = pd.read_csv(dataset_path, sep=',')
    X = df
    y = df['label']

    st = StratifiedKFold(n_splits=n_folds)
    folds = st.split(X=X, y=y)
    l = []
    for k, (train_idx, test_idx) in enumerate(folds):
        dang_ae, safe_ae, w2v_model, n2v_rel, n2v_spat, tree_rel, tree_spat, mlp = train(df.iloc[train_idx], k)
        p, r, f1, s = test(df=df.iloc[test_idx], df_train=df.iloc[train_idx], ae_dang=dang_ae, ae_safe=safe_ae,
                           tree_rel=tree_rel, tree_spat=tree_spat, mod_rel=n2v_rel, mod_spat=n2v_spat, mlp=mlp,
                           w2v_model=w2v_model)
        l.append((p, r, f1, s))


