import numpy as np
import pandas as pd
import torch
import yaml
from modelling.immense import train, train_w2v_model
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
    separator = train_dataset_params["separator"]

    models_dir = model_params["dir_models"]
    epochs_rel = model_params["epochs_rel"]
    epochs_spat = model_params["epochs_spat"]
    mlp_batch_size = int(model_params["mlp_batch_size"])
    mlp_lr = float(model_params["mlp_lr"])
    ne_dim_rel = int(model_params["ne_dim_rel"])
    ne_dim_spat = int(model_params["ne_dim_spat"])
    word_emb_size = int(model_params["word_emb_size"])
    w2v_epochs = int(model_params["w2v_epochs"])
    loss = model_params["loss"]

    w2v_path = join(models_dir, "w2v_{}.pkl".format(word_emb_size))

    if not exists(models_dir):
        makedirs(models_dir)
    train_df = pd.read_csv(train_df)


    users_embs_dict = train_w2v_model(embedding_size=word_emb_size, epochs=w2v_epochs, id_field_name=field_id,
                                      model_dir=models_dir, text_field_name=field_text, train_df=train_df)

    """confs = [(True, False, False), (True, False, True), (True, True, False), (True, True, True), 
                (False, False, True), (False, True, False), (False, True, True)]
    for conf in confs:
        consider_content, consider_rel, consider_spat = conf[0], conf[1], conf[2]"""
    print("CONTENT: {} REL: {} SPAT: {}".format(consider_content, consider_rel, consider_spat))
    train(train_df=train_df, model_dir=models_dir, gnn_batch_size=64, field_name_id=field_id,
          field_name_label=field_label, path_rel=path_rel, path_spat=path_spat, word_emb_size=word_emb_size,
          ne_dim_spat=ne_dim_spat, ne_dim_rel=ne_dim_rel, eps_nembs_spat=epochs_spat, eps_nembs_rel=epochs_rel,
          consider_rel=consider_rel, separator=separator, consider_spat=consider_spat,
          consider_content=consider_content, users_embs_dict=users_embs_dict, loss=loss)
