import numpy as np
import pandas as pd
import tensorflow as tf
import torch
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
from os.path import exists, join
from os import makedirs
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from utils import load_from_pickle, save_to_pickle
np.random.seed(123)


def train_w2v_model(train_df, embedding_size, epochs, model_dir, dataset_dir, name):
    """
    Train the Word2Vc model that will be used for learning the embeddings of the content.
    :param train_df: Dataframe with training data
    :param embedding_size:
    :param epochs:
    :param model_dir:
    :param dataset_dir:
    :param name: name of the file containing the word2vec mmodel
    :return: dang_posts_array: Array of shape [n_dang_users, embedding_size] with the embeddings of the dangerous users
    :return: safe_posts_array: Array of shape [n_safe_users, embedding_size] with the embeddings of the safe users
    :return: users_embeddings (although the variable name is safe_users_embeddings): Dictionary having as keys the
        users' IDs and, for each of them, the embedding array given by the sum of the words in their posts
    """
    tok = TextPreprocessing()
    posts_content = tok.token_list(train_df)
    if not exists(join(model_dir, name)):
        print("Training word2vec model")
        w2v_model = WordEmb(posts_content, embedding_size=embedding_size, window=10, epochs=epochs, model_dir=model_dir)
        w2v_model.train_w2v()
        save_to_pickle(join(model_dir, name), w2v_model)
    else:
        print("Loading word2vec model")
        w2v_model = load_from_pickle(join(model_dir, name))
    # split content in safe and dangerous
    dang_posts = train_df.loc[train_df['label'] == 1]
    safe_posts = train_df.loc[train_df['label'] == 0]
    users = tok.token_list(dang_posts)
    path = join(dataset_dir, "list_dang_posts_{}.pickle".format(embedding_size))
    dang_users_embeddings = w2v_model.text_to_vec(users=users, path=path)
    safe_users_embeddings = w2v_model.text_to_vec(users=tok.token_list(safe_posts), path=join(dataset_dir, "list_safe_posts_{}.pickle".format(embedding_size)))
    dang_posts_array = np.array(list(dang_users_embeddings.values()))
    safe_posts_array = np.array(list(safe_users_embeddings.values()))
    safe_users_embeddings.update(dang_users_embeddings)     # merge dang_users_embeddings and safe_users_embeddings, so we have a dict with all the users. Doing dang_users_embeddings.update(safe_users_embeddings) has the same output. Ugly but effective
    return dang_posts_array, safe_posts_array, safe_users_embeddings


def learn_mlp(train_df, content_embs, dang_ae, safe_ae, tree_rel, tree_spat, spat_node_embs, rel_node_embs,
              id2idx_spat: dict, id2idx_rel: dict, model_dir, n2v_rel=None, n2v_spat=None):
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
        id2idx_spat: Dictionary having as keys the user IDs and as value their index in the spatial adjacency matrix
        id2idx_rel: Dictionary having as keys the user IDs and as value their index in the relational adjacency matrix
        model_dir:
        n2v_spat: spatial node2vec model
        n2v_rel: relational node2vec model
    Returns: The learned MLP
    """
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
    for index, row in tqdm(train_df.iterrows()):
        id = row['id']
        if n2v_rel:
            try:
                dtree_input = np.expand_dims(n2v_rel.wv[str(id)], axis=0)
                pr_rel, conf_rel = test_decision_tree(test_set=dtree_input, cls=tree_rel)
            except KeyError:
                pr_rel, conf_rel = row['label'], 1.0
        else:
            if id in id2idx_rel.keys():
                idx = id2idx_rel[id]
                dtree_input = np.expand_dims(rel_node_embs[idx], axis=0)
                pr_rel, conf_rel = test_decision_tree(test_set=dtree_input, cls=tree_rel)
            else:
                pr_rel, conf_rel = row['label'], 1.0
        """if n2v_spat:
            try:
                pass
                #dtree_input = np.expand_dims(n2v_spat.wv[str(id)], axis=0)
                #pr_spat, conf_spat = test_decision_tree(test_set=dtree_input, cls=tree_spat)
            except KeyError:
                pr_spat, conf_spat = row['label'], 1.0
        else:
            if id in id2idx_spat.keys():
                idx = id2idx_spat[id]
                pr_spat, conf_spat = test_decision_tree(test_set=np.expand_dims(spat_node_embs[idx], axis=0), cls=tree_spat)
            else:
                pr_spat, conf_spat = row['label'], 1.0"""
        pr_spat = conf_spat = 1
        #pr_rel = conf_rel = 1
        dataset[index, 3] = pr_rel
        dataset[index, 4] = conf_rel
        dataset[index, 5] = pr_spat
        dataset[index, 6] = conf_spat
    #############   IMPORTANTE  #############
    #### STO TRAINANDO IL MODELLO 3%      ###
    #### CONSIDERANDO SOLO IL TESTO       ###
    #########################################
    mlp = MLP(X_train=dataset, y_train=np.array(train_df['label']), model_dir=model_dir)
    mlp.train()
    return mlp


def train(train_df, dataset_dir, model_dir, word_embedding_size, w2v_epochs, rel_node_emb_technique:str,
          spat_node_emb_technique:str, rel_node_embedding_size, spat_node_embedding_size, we_model_name, rel_path=None,
          spatial_path=None, epochs_spat_nembs=None, epochs_rel_nembs=None, adj_matrix_spat_path=None,
          adj_matrix_rel_path=None, id2idx_rel_path=None, id2idx_spat_path=None, batch_size=None):
    """
    Builds and trains the independent modules that analyze content, social relationships and spatial relationships, and
    then fuses them with the MLP
    :param train_df: Dataframe with the posts used for the MLP training
    :param dataset_dir: Directory containing the dataset
    :param model_dir: Directory where the models will be saved
    :param word_embedding_size: Dimension of the word embeddings to create
    :param w2v_epochs:
    :param rel_node_emb_technique: Technique to adopt for learning relational node embeddings
    :param spat_node_emb_technique: Technique to adopt for learning spatial node embeddings
    :param rel_node_embedding_size: Dimension of the relational node embeddings to learn
    :param spat_node_embedding_size: Dimension of the spatial node embeddings to learn
    :param we_model_name: Name of node embedding model
    :param rel_path: Path to the file stating the social relationships among the users
    :param spatial_path: Path to the file stating the spatial relationships among the users
    :param epochs_rel_nembs: Epochs for training the relational node embedding model
    :param epochs_spat_nembs: Epochs for training the spatial node embedding model
    :param adj_matrix_rel_path: Path to the relational adj matrix (pca, none, autoencoder)
    :param adj_matrix_spat_path: Path to the spatial adj matrix (pca, none, autoencoder)
    :param id2idx_rel_path: Path to the file containing the dictionary that matches the node IDs to their index in the relational adj matrix (graphsage, pca, autoencoder)
    :param id2idx_spat_path: Path to the file containing the dictionary that matches the node IDs to their index in the spatial adj matrix (graphsage, pca, autoencoder)
    :param batch_size:
    :return: Nothing, the learned mlp will be saved in the file "mlp.h5" and put in the model directory
    """
    dang_users_arrays, safe_users_arrays, users_embeddings_dict = train_w2v_model(train_df=train_df, embedding_size=word_embedding_size,
                                                         epochs=w2v_epochs, model_dir=model_dir, dataset_dir=dataset_dir, name=we_model_name)


    ################# TRAIN AND LOAD SAFE AND DANGEROUS AUTOENCODER ####################
    dang_ae = AE(X_train=dang_users_arrays, name='autoencoderdang', model_dir=model_dir, epochs=100, batch_size=128, lr=0.05).train_autoencoder_content()
    safe_ae = AE(X_train=safe_users_arrays, name='autoencodersafe', model_dir=model_dir, epochs=100, batch_size=128, lr=0.05).train_autoencoder_content()
    ################# TRAIN OR LOAD DECISION TREES ####################
    model_dir_rel = join(model_dir, "node_embeddings", "rel")
    model_dir_spat = join(model_dir, "node_embeddings", "spat")
    try:
        makedirs(model_dir_rel, exist_ok=False)
        makedirs(model_dir_spat, exist_ok=False)
    except OSError:
        pass
    rel_tree_path = join(model_dir_rel, "dtree_{}_{}.h5".format(rel_node_emb_technique, rel_node_embedding_size))
    spat_tree_path = join(model_dir_spat, "dtree_{}_{}.h5".format(rel_node_emb_technique, rel_node_embedding_size))

    x_rel, y_rel = reduce_dimension(rel_node_emb_technique, model_dir=model_dir_rel, edge_path=rel_path, lab="rel",
                                    id2idx_path=id2idx_rel_path, node_embedding_size=rel_node_embedding_size,
                                    train_df=train_df, epochs=epochs_rel_nembs, adj_matrix_path=adj_matrix_rel_path,
                                    sizes=[2, 3], features_dict=users_embeddings_dict, batch_size=batch_size)

    x_spat, y_spat = reduce_dimension(spat_node_emb_technique, model_dir=model_dir_spat, edge_path=spatial_path, lab="spat",
                                      id2idx_path=id2idx_spat_path, node_embedding_size=spat_node_embedding_size,
                                      train_df=train_df, epochs=epochs_spat_nembs, adj_matrix_path=adj_matrix_spat_path,
                                      sizes=[5, 5], features_dict=users_embeddings_dict, batch_size=batch_size)
    if not exists(rel_tree_path):
        train_decision_tree(train_set=x_rel, save_path=rel_tree_path, train_set_labels=y_rel, name="rel")
    if not exists(spat_tree_path):
        train_decision_tree(train_set=x_spat, save_path=spat_tree_path, train_set_labels=y_spat, name="spat")

    tree_rel = load_decision_tree(rel_tree_path)
    tree_spat = load_decision_tree(spat_tree_path)

    ################# NOW THAT WE HAVE THE MODELS WE CAN OBTAIN THE TRAINING SET FOR THE MLP #################
    if rel_node_emb_technique == "node2vec":
        n2v_rel = Word2Vec.load(join(model_dir_rel, "n2v.h5"))
        id2idx_rel = None
    else:
        n2v_rel = None
        id2idx_rel = load_from_pickle(id2idx_rel_path)
    if spat_node_emb_technique == "node2vec":
        n2v_spat = Word2Vec.load(join(model_dir_spat, "n2v.h5"))
        id2idx_spat = None
    else:
        n2v_spat = None
        id2idx_spat = load_from_pickle(id2idx_spat_path)
    content_embs = np.array(list(users_embeddings_dict.values()))
    mlp = learn_mlp(train_df=train_df, content_embs=content_embs, dang_ae=dang_ae, safe_ae=safe_ae, tree_rel=tree_rel,
                    tree_spat=tree_spat, rel_node_embs=x_rel, spat_node_embs=x_spat, model_dir=model_dir,
                    id2idx_rel=id2idx_rel, id2idx_spat=id2idx_spat, n2v_rel=n2v_rel, n2v_spat=n2v_spat)
    save_to_pickle(join(model_dir, "mlp.pkl"), mlp)


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
def test(rel_ne_technique, spat_ne_technique, test_df, train_df, w2v_model, dang_ae, safe_ae, tree_rel, tree_spat,
         mlp: MLP, id2idx_rel=None, id2idx_spat=None, mod_rel=None, mod_spat=None, pca_rel=None, pca_spat=None,
         ae_rel=None, ae_spat=None, adj_matrix_spat=None, adj_matrix_rel=None, rel_net_path=None, spat_net_path=None):
    test_set = np.zeros(shape=(len(test_df), 7))
    tok = TextPreprocessing()
    posts = tok.token_list(test_df)
    test_posts_embs_dict = w2v_model.text_to_vec(posts)
    test_posts_embs = np.array(list(test_posts_embs_dict.values()))
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
    #conf_missing_info = max(train_df['label'].value_counts()) / len(train_df)  # ratio
    conf_missing_info = 0.5

    # Social part
    test_set = obtain_graph_based_predictions(test_df, pred_missing_info, conf_missing_info, rel_ne_technique, tree_rel,
                                              test_set, "rel", mod_rel, adj_matrix_rel, pca_rel, ae_rel,
                                              id2idx_rel, test_posts_embs_dict, net_path=rel_net_path)

    # Spatial part
    test_set = obtain_graph_based_predictions(test_df, pred_missing_info, conf_missing_info, rel_ne_technique, tree_spat,
                                              test_set, "spat", mod_spat, adj_matrix_spat, pca_spat, ae_spat,
                                              id2idx_spat, test_posts_embs_dict, net_path=spat_net_path)

    print(mlp.test(test_set, np.array(test_df['label'])))


def obtain_graph_based_predictions(test_df, pmi, cmi, ne_technique, tree, test_set, mode, model=None, adj_mat=None,
                                   pca=None, ae=None, id2idx=None, embs=None, net_path=None):
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
        if not net_path:
            raise MissingParamException(ne_technique, "path to the edgelist")
        if not model:
            raise MissingParamException(ne_technique, "model")
        if not embs:
            raise MissingParamException(ne_technique, "user embeddings")
        mapper, inv_mapper = create_mappers(embs)
        graph = create_graph(inv_map=inv_mapper, weighted=weighted, features=embs, edg_dir=net_path)
        with torch.no_grad():
            graph = graph.to(model.device)
            preds = model(graph)
            preds = preds.to("cpu").numpy
            for index, row in test_df.iterrows():
                id = row.id
                dtree_input = preds[inv_mapper[id]]
                pr, conf = test_decision_tree(test_set=dtree_input, cls=tree)
                # TODO PROSEGUIRE
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
                    #TODO BISOGNA MODIFICARE GET TEST SET DTREE VISTO CHE ORA IL MODELLO NON è SOLO N2V
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
        p, r, f1, s = test(test_df=df.iloc[test_idx], train_df=df.iloc[train_idx], dang_ae=dang_ae, safe_ae=safe_ae,
                           tree_rel=tree_rel, tree_spat=tree_spat, mod_rel=n2v_rel, mod_spat=n2v_spat, mlp=mlp,
                           w2v_model=w2v_model)
        l.append((p, r, f1, s))


