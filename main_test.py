from modelling.sairus import test
from os.path import join
from utils import load_from_pickle
import pandas as pd
import yaml


def main_test():
    with open("parameters.yaml", 'r') as params_file:
        params = yaml.safe_load(params_file)
        dataset_general_params = params["dataset_general_params"]
        test_dataset_params = params["test_dataset_params"]
        model_params = params["model_params"]

    field_id = dataset_general_params["field_id"]
    field_text = dataset_general_params["field_text"]
    field_label = dataset_general_params["field_label"]
    test_df = test_dataset_params["test_df"]
    path_rel = test_dataset_params["test_social_net"]
    path_spat = test_dataset_params["test_spatial_net"]
    consider_content = test_dataset_params["consider_content"]
    consider_rel = test_dataset_params["consider_rel"]
    consider_spat = test_dataset_params["consider_spat"]

    models_dir = model_params["dir_models"]
    ne_dim_rel = int(model_params["ne_dim_rel"])
    ne_dim_spat = int(model_params["ne_dim_spat"])
    word_emb_size = int(model_params["word_emb_size"])

    w2v_path = join(models_dir, "w2v_{}.pkl".format(word_emb_size))
    mod_dir_rel = join(models_dir, "node_embeddings", "rel")
    mod_dir_spat = join(models_dir, "node_embeddings", "spat")

    test_df = pd.read_csv(test_df)
    w2v_model = load_from_pickle(w2v_path)

    dang_ae = load_from_pickle(join(models_dir, "autoencoderdang_{}.pkl".format(word_emb_size)))
    safe_ae = load_from_pickle(join(models_dir, "autoencodersafe_{}.pkl".format(word_emb_size)))
    mod_rel = load_from_pickle(join(mod_dir_rel, "graphsage_{}_{}.pkl".format(ne_dim_rel, word_emb_size)))
    mod_spat = load_from_pickle(join(mod_dir_spat, "graphsage_{}_{}.pkl".format(ne_dim_spat, word_emb_size)))

    """confs = [(True, False, False), (True, False, True), (True, True, False), (True, True, True), (False, False, True),
             (False, True, False), (False, True, True)]
    for conf in confs:
        consider_content, consider_rel, consider_spat = conf[0], conf[1], conf[2]"""
    mlp_name = "mlp"
    if consider_content:
        mlp_name += "_content_{}".format(word_emb_size)
    if consider_rel:
        mlp_name += "_rel_{}".format(ne_dim_rel)
    if consider_spat:
        mlp_name += "_spat_{}".format(ne_dim_spat)
    mlp = load_from_pickle(join(models_dir, mlp_name + ".pkl"))
    print(mlp_name.upper())
    test(df=test_df, w2v_model=w2v_model, ae_dang=dang_ae, ae_safe=safe_ae, mlp=mlp, mod_rel=mod_rel,
         mod_spat=mod_spat, rel_net_path=path_rel, spat_net_path=path_spat, field_name_text=field_text,
         field_name_id=field_id, field_name_label=field_label, consider_content=consider_content,
         consider_rel=consider_rel, consider_spat=consider_spat)


if __name__ == "__main__":
    main_test()
