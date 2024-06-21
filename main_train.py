import numpy as np
import pandas as pd
import torch
import yaml
from exceptions import *
from modelling.sairus import train, train_w2v_model
from os.path import exists, join
from os import makedirs
seed = 123
np.random.seed(seed)


def main_train(args=None):
    """textual_content_path = args.textual_content_path
    social_net_path = args.social_net_path
    spatial_net_path = args.spatial_net_path
    rel_adj_mat_path = args.rel_adj_mat_path
    spat_adj_mat_path = args.spat_adj_mat_path
    id2idx_rel_path = args.id2idx_rel_path
    id2idx_spat_path = args.id2idx_spat_path

    rel_technique = args.rel_technique
    spat_technique = args.spat_technique
    word_embedding_size = args.word_embedding_size
    window = args.window
    w2v_epochs = args.w2v_epochs
    spat_node_embedding_size = args.spat_node_embedding_size
    rel_node_embedding_size = args.rel_node_embedding_size
    n2v_epochs_spat = args.n2v_epochs_spat
    n2v_epochs_rel = args.n2v_epochs_rel
    rel_autoenc_epochs = args.rel_ae_epochs
    spat_autoenc_epochs = args.spat_ae_epochs
    models_dir = args.models_dir
    dataset_dir = args.dataset_dir"""


if __name__ == "__main__":
    with open("parameters.yaml", 'r') as params_file:
        params = yaml.safe_load(params_file)
        dataset_params = params["dataset_params"]
        model_params = params["model_params"]
    dataset_dir = dataset_params["dir_dataset"]
    graph_dir = dataset_params["dir_graph"]
    models_dir = dataset_params["dir_models"]
    df_name = dataset_params["df_name"]
    field_id = dataset_params["field_id"]
    field_text = dataset_params["field_text"]
    field_label = dataset_params["field_label"]
    social_net_name = dataset_params["social_net_name"]
    spatial_net_name = dataset_params["spatial_net_name"]
    dir = dataset_params["dir"]

    epochs_rel = model_params["epochs_rel"]
    epochs_spat = model_params["epochs_spat"] #25
    mlp_batch_size = int(model_params["mlp_batch_size"])
    mlp_lr = float(model_params["mlp_lr"])
    ne_dim_rel = int(model_params["ne_dim_rel"])      #128
    ne_dim_spat = int(model_params["ne_dim_spat"])    #128
    ne_technique_rel = model_params["ne_technique_rel"]
    ne_technique_spat = model_params["ne_technique_spat"]
    w2v_path = model_params["w2v_path"]
    word_emb_size = int(model_params["word_emb_size"])
    w2v_epochs = int(model_params["w2v_epochs"])

    path_rel = join(graph_dir, social_net_name)
    path_spat = join(graph_dir, spatial_net_name)

    train_path = join(dataset_dir, df_name) #tweets.csv

    rel_adj_mat_path = spat_adj_mat_path = None
    id2idx_rel_path = join(models_dir, "id2idx_rel.pkl")
    id2idx_spat_path = join(models_dir, "id2idx_spat.pkl")

    if not exists(models_dir):
        makedirs(models_dir)
    train_df = pd.read_csv(train_path)
    nz = len(train_df[train_df.label == 1])
    pos_weight = len(train_df) / nz
    neg_weight = len(train_df) / (2*(len(train_df) - nz))

    if ne_technique_rel.lower() in ["autoencoder", "pca", "none"]:
        if not rel_adj_mat_path:
            raise AdjMatException(lab="rel")
        if not id2idx_rel_path:
            raise Id2IdxException(lab="rel")
    if ne_technique_spat.lower() in ["autoencoder", "pca", "none"]:
        if not spat_adj_mat_path:
            raise AdjMatException(lab="spat")
        if not id2idx_spat_path:
            raise Id2IdxException(lab="spat")

    competitor = False
    consider_content = True
    consider_rel = True
    consider_spat = False
    users_embs_dict = train_w2v_model(embedding_size=word_emb_size, epochs=w2v_epochs, id_field_name=field_id,
                                      model_dir=models_dir, text_field_name=field_text, train_df=train_df)

    if competitor:
        while not (consider_content and consider_rel and consider_spat):
            consider_spat = not consider_spat
            if not consider_spat:
                consider_rel = not consider_rel
                if not consider_rel:
                    consider_content = not consider_content
            print("CONTENT: {} REL: {} SPAT: {}".format(consider_content, consider_rel, consider_spat))
            train(train_df=train_df, model_dir=models_dir, batch_size=64, field_name_id=field_id,
                  id2idx_path_spat=id2idx_spat_path, path_rel=path_rel, path_spat=path_spat,
                  word_emb_size=word_emb_size, node_emb_technique_spat=ne_technique_spat,
                  node_emb_technique_rel=ne_technique_rel, node_emb_size_spat=ne_dim_spat, node_emb_size_rel=ne_dim_rel,
                  weights=torch.tensor([neg_weight, pos_weight]), eps_nembs_spat=epochs_spat, eps_nembs_rel=epochs_rel,
                  adj_matrix_path_spat=spat_adj_mat_path, adj_matrix_path_rel=rel_adj_mat_path,
                  id2idx_path_rel=id2idx_rel_path, consider_rel=consider_rel, consider_spat=consider_spat,
                  consider_content=consider_content, competitor=competitor, users_embs_dict=users_embs_dict)
    else:
        while not (consider_rel and consider_spat):
            consider_spat = True
            """consider_rel = not consider_rel
            if not consider_rel:
                consider_spat = not consider_spat
                if not consider_spat:
                    consider_content = not consider_content"""
            print("REL: {} SPAT: {}".format(consider_rel, consider_spat))
            train(train_df=train_df, model_dir=models_dir, batch_size=64, field_name_id=field_id,
                  id2idx_path_spat=id2idx_spat_path, path_rel=path_rel, path_spat=path_spat,
                  word_emb_size=word_embedding_size, node_emb_technique_spat=spat_technique,
                  node_emb_technique_rel=rel_technique, node_emb_size_spat=ne_dim_spat, node_emb_size_rel=ne_dim_rel,
                  weights=torch.tensor([neg_weight, pos_weight]), eps_nembs_spat=epochs_spat, eps_nembs_rel=epochs_rel,
                  adj_matrix_path_spat=spat_adj_mat_path, adj_matrix_path_rel=rel_adj_mat_path,
                  id2idx_path_rel=id2idx_rel_path, consider_rel=consider_rel, consider_spat=consider_spat,
                  consider_content=consider_content, competitor=competitor, users_embs_dict=users_embs_dict)

