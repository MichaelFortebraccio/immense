import numpy as np
import pandas as pd
import torch
from exceptions import Id2IdxException, MissingParamException
from gensim.models import Word2Vec
from modelling.ae import AE
from modelling.mlp import MLP
from modelling.text_preprocessing import TextPreprocessing
from modelling.word_embedding import WordEmb
from node_classification.random_forest import *
from node_classification.graph_embeddings.sage import create_graph, create_mappers
from os import makedirs
from os.path import exists, join
from sklearn.model_selection import StratifiedKFold
from node_classification.reduce_dimension import reduce_dimension
from sklearn.metrics import classification_report
from torch.nn import MSELoss
from torch.optim import Adam
from tqdm import tqdm
from utils import load_from_pickle, save_to_pickle, plot_confusion_matrix
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
    import time
    name = "w2v_{}.pkl".format(embedding_size)
    if not exists(join(model_dir, name)):
        start_emb = time.time()
        print("Training word2vec model")
        w2v_model = WordEmb(posts_content, embedding_size=embedding_size, window=10, epochs=epochs, model_dir=model_dir)
        w2v_model.train_w2v()
        save_to_pickle(join(model_dir, name), w2v_model)
        print("Elapsed time for training w2v: {}".format(time.time() - start_emb))
    else:
        print("Loading word2vec model")
        w2v_model = load_from_pickle(join(model_dir, name))
    # split content in safe and dangerous
    all_users_tokens = tok.token_dict(train_df, text_field_name=text_field_name, id_field_name=id_field_name)
    all_users_embeddings = w2v_model.text_to_vec(users=all_users_tokens)  # Get a dict of all the embeddings of each user, keeping the association with the key
    return all_users_embeddings


def learn_mlp(ae_dang, ae_safe, content_embs, id2idx_rel, id2idx_spat, model_dir, node_embs_rel, node_embs_spat,
              tec_rel, tec_spat, train_df, tree_rel, tree_spat, y_train, we_dim, rel_dim, spat_dim, consider_rel=True,
              consider_spat=True, n2v_rel=None, n2v_spat=None, weights=None):
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
    name = "mlp_{}".format(we_dim)
    if consider_rel:
        name += "_rel_{}".format(rel_dim)
    if consider_spat:
        name += "_spat_{}".format(spat_dim)
    name += ".pkl"
    print(name)
    mlp = MLP(X_train=dataset, y_train=y_train, model_path=join(model_dir, name), weights=weights)
    optim = Adam(mlp.parameters(), lr=0.004, weight_decay=1e-4)
    mlp.train_mlp(optim)


def get_relational_preds(technique, df, tree, node_embs, id2idx: dict, n2v, cmi, pmi=None):
    dataset = torch.zeros(np.array(node_embs).shape[0], 2)
    if technique == "graphsage":
        pr, conf = test_random_forest(test_set=node_embs, cls=tree)
        dataset[:, 0] = torch.tensor(pr, dtype=torch.float32)
        dataset[:, 1] = torch.tensor(conf, dtype=torch.float32)
    else:
        for index, row in tqdm(df.iterrows()):
            id = row['id']
            if technique == "node2vec":
                try:
                    dtree_input = np.expand_dims(n2v.wv[str(id)], axis=0)
                    pr, conf = test_random_forest(test_set=dtree_input, cls=tree)
                except KeyError:
                    if not pmi:
                        pmi = row['label']
                    pr, conf = pmi, cmi
            else:
                if id in id2idx.keys():
                    idx = id2idx[id]
                    dtree_input = np.expand_dims(node_embs[idx], axis=0)
                    pr, conf = test_random_forest(test_set=dtree_input, cls=tree)
                else:
                    if not pmi:
                        pmi = row['label']
                    pr, conf = pmi, cmi
            dataset[index, 0] = torch.tensor(pr, dtype=torch.float64)
            dataset[index, 1] = torch.tensor(conf, dtype=torch.float64)
    return dataset


def train(field_name_id, model_dir, node_emb_technique_rel: str, node_emb_technique_spat: str,
          node_emb_size_rel, node_emb_size_spat, train_df, word_emb_size, users_embs_dict, adj_matrix_path_rel=None,
          adj_matrix_path_spat=None, batch_size=None, consider_content=True, consider_rel=True, consider_spat=True,
          eps_nembs_rel=None, eps_nembs_spat=None, id2idx_path_rel=None, id2idx_path_spat=None, path_rel=None,
          path_spat=None, weights=None, competitor=False):
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
    :param batch_size: Batch size for learning node embedding models
    :param eps_nembs_rel: Epochs for training the relational node embedding model
    :param eps_nembs_spat: Epochs for training the spatial node embedding model
    :param id2idx_path_rel: Path to the file containing the dictionary that matches the node IDs to their index in the relational adj matrix (graphsage, pca, autoencoder)
    :param id2idx_path_spat: Path to the file containing the dictionary that matches the node IDs to their index in the spatial adj matrix (graphsage, pca, autoencoder)
    :param path_rel: Path to the file stating the social relationships among the users
    :param path_spat: Path to the file stating the spatial relationships among the users
    :return: Nothing, the learned mlp will be saved in the file "mlp.h5" and put in the model directory
    """
    y_train = list(train_df['label'])
    dang_posts_ids = list(train_df.loc[train_df['label'] == 1][field_name_id])
    safe_posts_ids = list(train_df.loc[train_df['label'] == 0][field_name_id])

    posts_embs = np.array(list(users_embs_dict.values()))
    keys = list(users_embs_dict.keys())

    dang_users_ar = np.array([users_embs_dict[k] for k in keys if k in dang_posts_ids])
    safe_users_ar = np.array([users_embs_dict[k] for k in keys if k in safe_posts_ids])
    posts_embs = torch.tensor(posts_embs, dtype=torch.float32)


    ################# TRAIN AND LOAD SAFE AND DANGEROUS AUTOENCODER ####################
    dang_ae_name = join(model_dir, "autoencoderdang_{}.pkl".format(word_emb_size))
    safe_ae_name = join(model_dir, "autoencodersafe_{}.pkl".format(word_emb_size))

    if not exists(dang_ae_name):
        dang_ae = AE(X_train=dang_users_ar, epochs=150, batch_size=32, lr=0.002, name=dang_ae_name)
        dang_ae.train_autoencoder_content()
    else:
        dang_ae = load_from_pickle(dang_ae_name)
    if not exists(safe_ae_name):
        safe_ae = AE(X_train=safe_users_ar, epochs=100, batch_size=64, lr=0.002, name=safe_ae_name)
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
    rel_forest_path = join(model_dir_rel, "forest_{}_{}.h5".format(node_emb_size_rel, word_emb_size))
    spat_forest_path = join(model_dir_spat, "forest_{}_{}.h5".format(node_emb_size_spat, word_emb_size))

    tree_rel = tree_spat = x_rel = x_spat = n2v_rel = n2v_spat = id2idx_rel = id2idx_spat = None
    if consider_rel:
        x_rel, y_rel = reduce_dimension(node_emb_technique_rel, model_dir=model_dir_rel, edge_path=path_rel, lab="rel",
                                        id2idx_path=id2idx_path_rel, ne_dim=node_emb_size_rel, train_df=train_df,
                                        epochs=eps_nembs_rel, adj_matrix_path=adj_matrix_path_rel, sizes=[2, 3],
                                        features_dict=users_embs_dict, batch_size=batch_size, training_weights=weights,
                                        we_dim=word_emb_size)
        if not exists(rel_forest_path):
            train_random_forest(train_set=x_rel, dst_dir=rel_forest_path, train_set_labels=y_rel, name="rel")
        tree_rel = load_from_pickle(rel_forest_path)

    if consider_spat:
        x_spat, y_spat = reduce_dimension(node_emb_technique_spat, model_dir=model_dir_spat, edge_path=path_spat,
                                          lab="spat", id2idx_path=id2idx_path_spat, ne_dim=node_emb_size_spat,
                                          train_df=train_df, epochs=eps_nembs_spat, adj_matrix_path=adj_matrix_path_spat,
                                          sizes=[3, 5], features_dict=users_embs_dict, batch_size=batch_size,
                                          training_weights=weights, we_dim=word_emb_size)
        if not exists(spat_forest_path):
            train_random_forest(train_set=x_spat, dst_dir=spat_forest_path, train_set_labels=y_spat, name="spat")
        tree_spat = load_from_pickle(spat_forest_path)
    # WE CAN NOW OBTAIN THE TRAINING SET FOR THE MLP
    if node_emb_technique_rel == "node2vec":
        n2v_rel = Word2Vec.load(join(model_dir_rel, "n2v.h5"))
    elif node_emb_technique_rel != "graphsage":
        id2idx_rel = load_from_pickle(id2idx_path_rel)
    if node_emb_technique_spat == "node2vec":
        n2v_spat = Word2Vec.load(join(model_dir_spat, "n2v.h5"))
    elif node_emb_technique_spat != "graphsage":
        id2idx_spat = load_from_pickle(id2idx_path_spat)

    if not competitor:
        print("Learning MLP...\n")
        learn_mlp(ae_dang=dang_ae, ae_safe=safe_ae, content_embs=posts_embs, consider_rel=consider_rel,
                  consider_spat=consider_spat, id2idx_rel=id2idx_rel, id2idx_spat=id2idx_spat, model_dir=model_dir,
                  node_embs_rel=x_rel, node_embs_spat=x_spat, tec_rel=node_emb_technique_rel,
                  tec_spat=node_emb_technique_spat, train_df=train_df, tree_rel=tree_rel, tree_spat=tree_spat,
                  y_train=y_train, n2v_rel=n2v_rel, n2v_spat=n2v_spat, weights=weights, we_dim=word_emb_size,
                  rel_dim=node_emb_size_rel, spat_dim=node_emb_size_spat)
    else:
        train_set_forest = posts_embs
        name = "forest"
        if consider_content:
            name += "_content_{}".format(word_emb_size)
        if consider_rel:
            train_set_forest = np.hstack((train_set_forest, x_rel))
            name += "_rel_{}".format(node_emb_size_rel)
        if consider_spat:
            train_set_forest = np.hstack((train_set_forest, x_spat))
            name += "_spat_{}".format(node_emb_size_spat)
        train_random_forest(train_set=train_set_forest, dst_dir=join(model_dir, "competitors", name+".pkl"),
                            train_set_labels=y_train, name=name)


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


def test(ae_dang, ae_safe, df, df_train, field_id, field_text, field_label, mlp: MLP, ne_technique_rel,
         ne_technique_spat, tree_rel, tree_spat, w2v_model, consider_rel=True, consider_spat=True, id2idx_rel=None,
         id2idx_spat=None, mod_rel=None, mod_spat=None, rel_net_path=None, spat_net_path=None, cls_competitor=None):
    tok = TextPreprocessing()
    posts = tok.token_dict(df, text_field_name=field_text, id_field_name=field_id)
    test_set = torch.zeros(len(posts), 7)
    posts_embs_dict = w2v_model.text_to_vec(posts)
    posts_embs = torch.tensor(list(posts_embs_dict.values()), dtype=torch.float32)
    pred_dang = ae_dang.predict(posts_embs)
    pred_safe = ae_safe.predict(posts_embs)
    loss = MSELoss()
    pred_loss_dang = []
    pred_loss_safe = []
    for i in range(posts_embs.shape[0]):
        pred_loss_dang.append(loss(posts_embs[i], pred_dang[i]))
        pred_loss_safe.append(loss(posts_embs[i], pred_safe[i]))

    labels = [1 if i < j else 0 for i, j in zip(pred_loss_dang, pred_loss_safe)]
    test_set[:, 0] = torch.tensor(pred_loss_dang, dtype=torch.float32)
    test_set[:, 1] = torch.tensor(pred_loss_safe, dtype=torch.float32)
    test_set[:, 2] = torch.tensor(labels, dtype=torch.float32)

    # At test time, if we meet an instance that doesn't have information about relationships or closeness, we will
    # replace the decision tree prediction with the most frequent label in the training set
    pred_missing_info = df_train['label'].value_counts().argmax()
    conf_missing_info = 0.5

    if consider_rel:
        if ne_technique_rel == "graphsage":
            mapper, inv_map_rel = create_mappers(posts_embs_dict)
            graph = create_graph(inv_map=inv_map_rel, weighted=False, features=posts_embs_dict, edg_dir=rel_net_path, df=df)
            with torch.no_grad():
                graph = graph.to(mod_rel.device)
                sage_rel_embs = mod_rel(graph, inference=True).detach().numpy()
            if not cls_competitor:
                social_part = get_relational_preds(technique=ne_technique_rel, df=df, tree=tree_rel,
                                                   node_embs=sage_rel_embs, id2idx=id2idx_rel, n2v=mod_rel,
                                                   cmi=conf_missing_info, pmi=pred_missing_info)
                test_set[:, 3], test_set[:, 4] = social_part[:, 0], social_part[:, 1]
    if consider_spat:
        if ne_technique_spat == "graphsage":
            mapper, inv_map_sp = create_mappers(posts_embs_dict)
            graph = create_graph(inv_map=inv_map_sp, weighted=True, features=posts_embs_dict, edg_dir=spat_net_path, df=df)
            with torch.no_grad():
                graph = graph.to(mod_spat.device)
                sage_spat_embs = mod_spat(graph, inference=True).detach().numpy()
            if not cls_competitor:
                spatial_part = get_relational_preds(technique=ne_technique_spat, df=df, tree=tree_spat,
                                                    node_embs=sage_spat_embs, id2idx=id2idx_spat, n2v=mod_spat,
                                                    cmi=conf_missing_info, pmi=pred_missing_info)
                test_set[:, 5], test_set[:, 6] = spatial_part[:, 0], spatial_part[:, 1]

    if not cls_competitor:
        pred = mlp.test(test_set, np.array(df[field_label]))
    else:
        test_set_forest = posts_embs
        if consider_rel:
            test_set_forest = np.hstack((test_set_forest, sage_rel_embs))
        if consider_spat:
            test_set_forest = np.hstack((test_set_forest, sage_spat_embs))
        pred, _ = test_random_forest(test_set=test_set_forest, cls=cls_competitor)
    y_true = np.array(df['label'])
    plot_confusion_matrix(y_true=y_true, y_pred=pred)
    print(classification_report(y_true=y_true, y_pred=pred))


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
        for index, row in test_df.iterrows():
            id = row.id
            dtree_input = node_embs[inv_mapper[id]]
            pr, conf = test_random_forest(test_set=dtree_input, cls=tree)
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
                    pr, conf = test_random_forest(test_set=dtree_input, cls=tree)
                except KeyError:
                    pr, conf = pmi, cmi
            else:
                if not id2idx:
                    raise Id2IdxException(mode)
                if id in id2idx.keys():
                    idx = id2idx[id]
                    dtree_input = get_testset_dtree(ne_technique, idx, adj_matrix=adj_mat, n2v=model, pca=pca, ae=ae)
                    pr, conf = test_random_forest(test_set=dtree_input, cls=tree)
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


