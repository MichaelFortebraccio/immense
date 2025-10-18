import os
import time
from os import makedirs
from os.path import exists, join
from datetime import datetime


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from sklearn.metrics import classification_report
from torch.nn import MSELoss
from torch.optim import Adam

from modelling.ae import AE
from modelling.mlp import MLP
from modelling.sage import create_graph, create_mappers
from modelling.text_preprocessing import TextPreprocessing
from modelling.word_embedding import WordEmb

from reduce_dimension import reduce_dimension
from utils import load_from_pickle, save_to_pickle, plot_confusion_matrix

import shap


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
        # tracker = EmissionsTracker()
        # tracker.start()
        start_emb = time.time()
        print("Training word2vec model")
        w2v_model = WordEmb(posts_content, embedding_size=embedding_size, window=10, epochs=epochs, model_dir=model_dir)
        w2v_model.train_w2v()
        # tracker.stop()
        save_to_pickle(join(model_dir, name), w2v_model)
        print("Elapsed time for training {} w2v: {}".format(embedding_size, time.time() - start_emb))

    else:
        print("Loading word2vec model")
        w2v_model = load_from_pickle(join(model_dir, name))
    # split content in safe and dangerous
    all_users_tokens = tok.token_dict(train_df, text_field_name=text_field_name, id_field_name=id_field_name)
    all_users_embeddings = w2v_model.text_to_vec(users=all_users_tokens)  # Get a dict of all the embeddings of each user, keeping the association with the key
    return all_users_embeddings


def model_fusion(model_dir, y_train, content_embs, mlp_name, mlp_loss, mlp_lr=.004, ae_dang=None, ae_safe=None,
                 rel_preds=None, spat_preds=None, weights=None):
    """
    Train the MLP aimed at fusing the models
    Args:
    :param model_dir:
    :param y_train: Train labels
    :param content_embs: torch tensor containing the word embeddings of the content posted by the users. The features
    are z-score normalized
    :param mlp_lr: learning rate for the mlp
    :param ae_dang: Dangerous autoencoder model. Leave it to None if you don't want to consider the content analysis module
    :param ae_safe: Safe autoencoder model. Leave it to None if you don't want to consider the content analysis module
    :param rel_preds: Predictions from the relational node embedding module. Leave it to none if you don't want to use that module
    :param spat_preds: Predictions from the spatial node embedding module. Leave it to none if you don't want to use that module
    :param mlp_name: File name of the mlp to create
    Returns: The learned MLP
    """
    dataset = torch.zeros((content_embs.shape[0], 7))

    if ae_dang and ae_safe:
        prediction_safe = ae_safe.predict(content_embs)
        prediction_risky = ae_dang.predict(content_embs)
        mse_loss = MSELoss()
        prediction_loss_safe = []
        prediction_loss_risky = []
        for i in range(content_embs.shape[0]):
            prediction_loss_safe.append(mse_loss(content_embs[i], prediction_safe[i]))
            prediction_loss_risky.append(mse_loss(content_embs[i], prediction_risky[i]))
        labels = [0 if i < j else 1 for i, j in zip(prediction_loss_safe, prediction_loss_risky)]
        dataset[:, 0] = torch.tensor(prediction_loss_safe, dtype=torch.float32)
        dataset[:, 1] = torch.tensor(prediction_loss_risky, dtype=torch.float32)
        dataset[:, 2] = torch.tensor(labels, dtype=torch.float32)

    if rel_preds is not None:
        safe_rel_probs = torch.tensor(rel_preds[:, 0], dtype=torch.float32)
        risky_rel_probs = torch.tensor(rel_preds[:, 1], dtype=torch.float32)
        dataset[:, 3], dataset[:, 4] = safe_rel_probs, risky_rel_probs
    if spat_preds is not None:
        safe_spat_probs = torch.tensor(spat_preds[:, 0], dtype=torch.float32)
        risky_spat_probs = torch.tensor(spat_preds[:, 1], dtype=torch.float32)
        dataset[:, 5], dataset[:, 6] = safe_spat_probs, risky_spat_probs
    save_to_pickle("explainability/X_train_{}_{}.pkl".format(content_embs.shape[1], mlp_loss), dataset)
    mlp = MLP(X_train=dataset, y_train=y_train, model_path=join(model_dir, "mlp", mlp_name), weights=weights, loss=mlp_loss)
    optim = Adam(mlp.parameters(), lr=mlp_lr, weight_decay=1e-4)
    mlp.train_mlp(optim)


def get_relational_preds(df, node_embs=None):
    dataset = torch.zeros(len(df), 2)
    dataset[:, 0] = torch.tensor(node_embs[:, 0], dtype=torch.float32)
    dataset[:, 1] = torch.tensor(node_embs[:, 1], dtype=torch.float32)
    return dataset


def train(field_name_id, field_name_label, model_dir, train_df, word_emb_size, users_embs_dict, separator, loss,
          gnn_batch_size=None, consider_content=True, consider_rel=True, consider_spat=True, ne_dim_rel=None,
          ne_dim_spat=None, eps_nembs_rel=None, eps_nembs_spat=None, path_rel=None, path_spat=None, retrain=False):
    """
    Builds and trains the independent modules that analyze content, social relationships and spatial relationships, and
    then fuses them with the MLP
    :param field_name_id: Name of the field containing the id
    :param field_name_label: Name of the field containing the label
    :param model_dir: Directory where the models will be saved
    :param ne_dim_rel: Dimension of the relational node embeddings to learn
    :param ne_dim_spat: Dimension of the spatial node embeddings to learn
    :param train_df: Dataframe with the posts used for the MLP training
    :param word_emb_size: Dimension of the word embeddings to create
    :param users_embs_dict: Dictionary having as keys the user IDs and as values their associated word embedding vector
    :param separator: Separator used in the edgelist
    :param gnn_batch_size: Batch size for learning node embedding models
    :param consider_content: Whether to use the module for the semantic content analysis
    :param consider_rel: Whether to use the module for the analysis of social relationships
    :param consider_spat: Whether to use the module for the analysis of spatial relationships
    :param eps_nembs_rel: Epochs for training the relational node embedding model. Can be None if consider_rel=False
    :param eps_nembs_spat: Epochs for training the spatial node embedding model. Can be None if consider_spat=False
    :param path_rel: Path to the file stating the social relationships among the users. Can be None if consider_rel=False
    :param path_spat: Path to the file stating the spatial relationships among the users. Can be None if consider_spat=False
    :param weights: Tensor containing the weights to use during training to compensate for data imbalance
    :return: Nothing, the learned mlp will be saved in the file "mlp.h5" and put in the model directory
    """
    #retrain = True
    y_train = list(train_df[field_name_label])
    dang_posts_ids = list(train_df.loc[train_df[field_name_label] == 1][field_name_id])
    safe_posts_ids = list(train_df.loc[train_df[field_name_label] == 0][field_name_id])

    posts_embs = np.array(list(users_embs_dict.values()))
    keys = list(users_embs_dict.keys())

    dang_users_ar = np.array([users_embs_dict[k] for k in keys if k in dang_posts_ids])
    safe_users_ar = np.array([users_embs_dict[k] for k in keys if k in safe_posts_ids])
    posts_embs = torch.tensor(posts_embs, dtype=torch.float32)

    weights = None
    if loss == "weighted":
        nz = len(train_df[train_df.label == 1])
        pos_weight = len(train_df) / nz
        neg_weight = len(train_df) / (2 * (len(train_df) - nz))
        weights = torch.tensor([neg_weight, pos_weight])
        weights = weights.to()

    mlp_name = "mlp"

    safe_ae = risky_ae = None
    if consider_content:
        mlp_name += "_content_{}".format(word_emb_size)
        safe_ae_name = join(model_dir, "autoencodersafe_{}.pkl".format(word_emb_size))
        risky_ae_name = join(model_dir, "autoencoderdang_{}.pkl".format(word_emb_size))

        if not exists(safe_ae_name) or retrain:
            safe_ae = AE(X_train=safe_users_ar, epochs=100, batch_size=64, lr=0.002, name=safe_ae_name)
            safe_ae.train_autoencoder_content()
        else:
            safe_ae = load_from_pickle(safe_ae_name)

        if not exists(risky_ae_name) or retrain:
            risky_ae = AE(X_train=dang_users_ar, epochs=150, batch_size=32, lr=0.002, name=risky_ae_name)
            risky_ae.train_autoencoder_content()
        else:
            risky_ae = load_from_pickle(risky_ae_name)

    model_dir_rel = join(model_dir, "node_embeddings", "rel")
    model_dir_spat = join(model_dir, "node_embeddings", "spat")
    try:
        makedirs(model_dir_rel, exist_ok=False)
        makedirs(model_dir_spat, exist_ok=False)
    except OSError:
        pass

    x_rel = x_spat = None

    if consider_rel:
        mlp_name += "_rel_{}".format(ne_dim_rel)
        x_rel = reduce_dimension(model_dir=model_dir_rel, edge_path=path_rel, lab="rel", ne_dim=ne_dim_rel,
                                 train_df=train_df, epochs=eps_nembs_rel, sizes=[2, 3], loss=loss,
                                 features_dict=users_embs_dict, batch_size=gnn_batch_size, training_weights=weights,
                                 we_dim=word_emb_size, retrain=retrain, separator=separator,
                                 field_name_id=field_name_id, field_name_label=field_name_label)
    if consider_spat:
        mlp_name += "_spat_{}".format(ne_dim_spat)
        x_spat = reduce_dimension(model_dir=model_dir_spat, edge_path=path_spat, lab="spat", ne_dim=ne_dim_spat,
                                  train_df=train_df, epochs=eps_nembs_spat, sizes=[3, 5],
                                  features_dict=users_embs_dict, batch_size=gnn_batch_size,
                                  training_weights=weights, we_dim=word_emb_size, retrain=retrain, separator=separator,
                                  field_name_id=field_name_id, field_name_label=field_name_label, loss=loss)

    mlp_name += "_{}.pkl".format(loss)
    print("Learning MLP...\n")
    model_fusion(ae_dang=risky_ae, ae_safe=safe_ae, content_embs=posts_embs, model_dir=model_dir, rel_preds=x_rel,
                 spat_preds=x_spat, y_train=y_train, weights=weights, mlp_name=mlp_name, mlp_loss=loss)


def test(df, field_name_id, field_name_text, field_name_label, mlp: MLP, w2v_model,
         consider_content, mlp_loss, consider_rel, consider_spat,
         ae_risky=None, ae_safe=None, mod_rel=None, mod_spat=None,
         rel_net_path=None, spat_net_path=None, separator="\t"):
    """
    Build test features, run MLP, compute SHAP, save CSVs, and plot:
      - dominant perspective per user (content/rel/spat)
      - dominant feature per user (one bar per feature)
    Also prints confusion matrix + classification report and saves SHAP with predictions.
    """
    # ---------- Build test_set (7 features) ----------
    tok = TextPreprocessing()
    posts = tok.token_dict(df, text_field_name=field_name_text, id_field_name=field_name_id)

    test_set = torch.zeros(len(posts), 7)
    posts_embs_dict = w2v_model.text_to_vec(posts)

    if consider_content:
        posts_embs = torch.tensor(list(posts_embs_dict.values()), dtype=torch.float32)
        pred_safe = ae_safe.predict(posts_embs)
        pred_risky = ae_risky.predict(posts_embs)
        loss = MSELoss()
        pred_loss_safe, pred_loss_risky = [], []
        for i in range(posts_embs.shape[0]):
            pred_loss_safe.append(loss(posts_embs[i], pred_safe[i]))
            pred_loss_risky.append(loss(posts_embs[i], pred_risky[i]))
        labels = [0 if i < j else 1 for i, j in zip(pred_loss_safe, pred_loss_risky)]
        test_set[:, 0] = torch.tensor(pred_loss_safe, dtype=torch.float32)   # AE_safe_err
        test_set[:, 1] = torch.tensor(pred_loss_risky, dtype=torch.float32)  # AE_risk_err
        test_set[:, 2] = torch.tensor(labels, dtype=torch.float32)           # label_content (pseudo-label)

    if consider_rel:
        mapper, inv_map_rel = create_mappers(posts_embs_dict)
        graph = create_graph(inv_map=inv_map_rel, weighted=False, features=posts_embs_dict,
                             edg_dir=rel_net_path, df=df, separator=separator,
                             field_name_id=field_name_id, field_name_label=field_name_label)
        with torch.no_grad():
            graph = graph.to(mod_rel.device)
            rel_preds = mod_rel(graph, inference=True).cpu().detach().numpy()
        safe_rel_probs = torch.tensor(rel_preds[:, 0], dtype=torch.float32)
        risky_rel_probs = torch.tensor(rel_preds[:, 1], dtype=torch.float32)
        test_set[:, 3], test_set[:, 4] = safe_rel_probs, risky_rel_probs   # label_rel, conf_rel

    if consider_spat:
        mapper, inv_map_sp = create_mappers(posts_embs_dict)
        graph = create_graph(inv_map=inv_map_sp, weighted=True, features=posts_embs_dict,
                             edg_dir=spat_net_path, df=df, separator=separator,
                             field_name_id=field_name_id, field_name_label=field_name_label)
        with torch.no_grad():
            graph = graph.to(mod_spat.device)
            spat_preds = mod_spat(graph, inference=False).cpu().detach().numpy()
        safe_spat_probs = torch.tensor(spat_preds[:, 0], dtype=torch.float32)
        risky_spat_probs = torch.tensor(spat_preds[:, 1], dtype=torch.float32)
        test_set[:, 5], test_set[:, 6] = safe_spat_probs, risky_spat_probs  # label_spat, conf_spat

    save_to_pickle(f"explainability/x_test_{posts_embs.shape[1]}_{mlp_loss}.pkl", test_set)

    # ---------- SHAP setup ----------
    X_test_np = test_set.detach().cpu().numpy()

    def predict_proba(x):
        """Return class probabilities from the MLP (expects log-probs)."""
        x_tensor = torch.tensor(x, dtype=torch.float32)
        with torch.no_grad():
            preds = mlp(x_tensor)           # logits or log-probs (your model returns log-probs)
            return torch.exp(preds).numpy() # convert log-probs -> probs

    feature_names = [
        "AE_safe_err", "AE_risk_err", "label_content",
        "prob_safe_rel", "prob_risk_rel",
        "prob_safe_spat", "prob_risk_spat",
    ]

    explainer = shap.Explainer(predict_proba, X_test_np, feature_names=feature_names)
    shap_values = explainer(X_test_np)  # shape: (N_samples, 7, N_classes)

    # Use class index 1 ("risky") for SHAP values table
    df_shap = pd.DataFrame(shap_values.values[:, :, 1], columns=feature_names)
    if field_name_id in df.columns:
        df_shap["user_id"] = df[field_name_id].values

    # ---------- Aggregate SHAP by perspective ----------
    # Content perspective = {AE_safe_err, AE_risk_err, label_content}
    df_shap["SHAP_content"] = df_shap[["AE_safe_err", "AE_risk_err", "label_content"]].abs().sum(axis=1)

    # Relational perspective = {prob_safe_rel, prob_risk_rel}
    df_shap["SHAP_rel"] = df_shap[["prob_safe_rel", "prob_risk_rel"]].abs().sum(axis=1)
    
    # Spatial perspective = {prob_safe_spat, prob_risk_spat}
    df_shap["SHAP_spat"] = df_shap[["prob_safe_spat", "prob_risk_spat"]].abs().sum(axis=1)

    # Dominant perspective per user
    df_shap["dominant_perspective"] = df_shap[["SHAP_content", "SHAP_rel", "SHAP_spat"]].idxmax(axis=1)

    # ---------- Save plots / CSV ----------
    shap_dir = "shap"
    os.makedirs(shap_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Distribution of dominant perspective
    plot_path_persp = os.path.join(shap_dir, f"dominant_perspective_distribution_{timestamp}.png")
    plt.figure(figsize=(5, 4))
    df_shap["dominant_perspective"].value_counts().plot(kind="bar")
    plt.title("Most influential perspective per user (test set)")
    plt.ylabel("Users")
    plt.tight_layout()
    plt.savefig(plot_path_persp)
    plt.close()

    # Dominant single feature per user (largest |SHAP|)
    abs_shap = df_shap[feature_names].abs().to_numpy()   # (N, 7)
    winner_idx = abs_shap.argmax(axis=1)
    feature_winner = pd.Series([feature_names[i] for i in winner_idx])
    counts_by_feature = feature_winner.value_counts().reindex(feature_names, fill_value=0)

    plot_path_feat = os.path.join(shap_dir, f"dominant_feature_distribution_{timestamp}.png")
    plt.figure(figsize=(7, 4))
    counts_by_feature.plot(kind="bar")
    plt.title("Most influential feature per user (test set)")
    plt.ylabel("Users")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(plot_path_feat)
    plt.close()

    # ---------- Predictions & reports ----------
    pred = mlp.test(test_set)  # predicted labels from the MLP
    y_true = np.array(df[field_name_label])

    # Confusion matrix + classification report
    plot_confusion_matrix(y_true=y_true, y_pred=pred)
    print(classification_report(y_true=y_true, y_pred=pred))

    # Attach labels/predictions to the SHAP table and save
    df_shap["true_label"] = y_true
    df_shap["predicted_label"] = pred
    df_shap["correct"] = df_shap["true_label"] == df_shap["predicted_label"]

    csv_path_full = os.path.join(shap_dir, f"shap_output_with_predictions_{timestamp}.csv")
    df_shap.to_csv(csv_path_full, index=False)

    print(f"\n✔️ Files saved in '{shap_dir}':")
    print(f"- {os.path.basename(csv_path_full)} (SHAP + predictions)")
    print(f"- {os.path.basename(plot_path_persp)} (dominant perspective plot)")
    print(f"- {os.path.basename(plot_path_feat)} (dominant feature plot)")


def export_shap_feature_scores_per_user(shap_dir="shap", field_name_id="user_id", class_index=1):
    """
    Load the most recent SHAP CSV saved by `test(...)` and return per-user SHAP scores.
    Prints a preview and returns the DataFrame (no new files are created).
    """
    import os, glob, pandas as pd

    shap_files = sorted(glob.glob(os.path.join(shap_dir, "shap_output_with_predictions_*.csv")))
    if not shap_files:
        print("❌ No SHAP files found.")
        return

    last_file = shap_files[-1]
    print(f"📄 Loading latest SHAP file: {os.path.basename(last_file)}")
    df_shap = pd.read_csv(last_file)

    columns_to_keep = [
        field_name_id,
        "AE_safe_err", "AE_risk_err",
        "label_content", "label_rel", "conf_rel",
        "label_spat", "conf_spat",
        "SHAP_content", "SHAP_rel", "SHAP_spat",
        "dominant_perspective",
        "true_label", "predicted_label", "correct"
    ]
    available_columns = [c for c in columns_to_keep if c in df_shap.columns]
    df_result = df_shap[available_columns]

    print("\n✅ SHAP scores for top 5 users (with predictions and labels):")
    print(df_result.head())

    return df_result
