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


if __name__ == "__main__":
    with open("parameters.yaml", 'r') as params_file:
        params = yaml.safe_load(params_file)
        dataset_general_params = params["dataset_general_params"]
        train_dataset_params = params["train_dataset_params"]
        model_params = params["model_params"]
    field_id = dataset_general_params["field_id"]
    field_text = dataset_general_params["field_text"]
    field_label = dataset_general_params["field_label"]
    train_df = dataset_general_params["train_df"]
    path_rel = train_dataset_params["train_social_net"]
    path_spat = train_dataset_params["train_spatial_net"]
    consider_content = train_dataset_params["consider_content"]
    consider_rel = train_dataset_params["consider_rel"]
    consider_spat = train_dataset_params["consider_spat"]

    models_dir = model_params["dir_models"]
    epochs_rel = model_params["epochs_rel"]
    epochs_spat = model_params["epochs_spat"]
    mlp_batch_size = int(model_params["mlp_batch_size"])
    mlp_lr = float(model_params["mlp_lr"])
    ne_dim_rel = int(model_params["ne_dim_rel"])
    ne_dim_spat = int(model_params["ne_dim_spat"])
    ne_technique_rel = model_params["ne_technique_rel"]
    ne_technique_spat = model_params["ne_technique_spat"]
    w2v_path = model_params["w2v_path"]
    word_emb_size = int(model_params["word_emb_size"])
    w2v_epochs = int(model_params["w2v_epochs"])

    rel_adj_mat_path = spat_adj_mat_path = None
    id2idx_rel_path = join(models_dir, "id2idx_rel.pkl")
    id2idx_spat_path = join(models_dir, "id2idx_spat.pkl")

    if not exists(models_dir):
        makedirs(models_dir)
    train_df = pd.read_csv(train_df)
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
    users_embs_dict = train_w2v_model(embedding_size=word_emb_size, epochs=w2v_epochs, id_field_name=field_id,
                                      model_dir=models_dir, text_field_name=field_text, train_df=train_df)

    if not competitor:
        print("REL: {} SPAT: {}".format(consider_rel, consider_spat))
        train(train_df=train_df, model_dir=models_dir, batch_size=64, field_name_id=field_id,
              id2idx_path_spat=id2idx_spat_path, path_rel=path_rel, path_spat=path_spat, word_emb_size=word_emb_size,
              node_emb_technique_spat=ne_technique_spat, node_emb_technique_rel=ne_technique_rel,
              node_emb_size_spat=ne_dim_spat, node_emb_size_rel=ne_dim_rel,
              weights=torch.tensor([neg_weight, pos_weight]), eps_nembs_spat=epochs_spat, eps_nembs_rel=epochs_rel,
              adj_matrix_path_spat=spat_adj_mat_path, adj_matrix_path_rel=rel_adj_mat_path,
              id2idx_path_rel=id2idx_rel_path, consider_rel=consider_rel, consider_spat=consider_spat,
              consider_content=consider_content, competitor=competitor, users_embs_dict=users_embs_dict)
    else:
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
