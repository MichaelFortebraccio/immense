from modelling.sairus import test
from os.path import join
from utils import load_from_pickle, get_model
import pandas as pd
import yaml


def main_test(args=None):
    with open("parameters.yaml", 'r') as params_file:
        params = yaml.safe_load(params_file)
        dataset_general_params = params["dataset_general_params"]
        test_dataset_params = params["test_dataset_params"]
        model_params = params["model_params"]

    train_df = dataset_general_params["train_df"]
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
    ne_technique_rel = model_params["ne_technique_rel"]
    ne_technique_spat = model_params["ne_technique_spat"]
    w2v_path = model_params["w2v_path"]
    word_emb_size = int(model_params["word_emb_size"])

    adj_mat_spat_path = adj_mat_rel_path = None
    id2idx_rel_path = join(models_dir, "id2idx_rel.pkl")
    id2idx_spat_path = join(models_dir, "id2idx_spat.pkl")
    mod_dir_rel = join(models_dir, "rel")
    mod_dir_spat = join(models_dir, "spat")

    train_df = pd.read_csv(train_df)
    test_df = pd.read_csv(test_df)
    w2v_model = load_from_pickle(w2v_path)

    dang_ae = load_from_pickle(join(models_dir, "autoencoderdang_{}.pkl".format(word_emb_size)))
    safe_ae = load_from_pickle(join(models_dir, "autoencodersafe_{}.pkl".format(word_emb_size)))

    competitor = False
    mod_rel = pca_rel = ae_rel = adj_mat_rel = id2idx_rel = mod_spat = pca_spat = ae_spat = adj_mat_spat = id2idx_spat = forest_rel = forest_spat = None
    mod_rel, pca_rel, ae_rel, adj_mat_rel, id2idx_rel = get_model(technique=ne_technique_rel, mod_dir=mod_dir_rel,
                                                                  lab="rel", adj_mat_path=adj_mat_rel_path,
                                                                  id2idx_path=id2idx_rel_path, ne_dim=ne_dim_rel,
                                                                  we_dim=word_emb_size)
    mod_spat, pca_spat, ae_spat, adj_mat_spat, id2idx_spat = get_model(technique=ne_technique_spat,
                                                                       mod_dir=mod_dir_spat,
                                                                       lab="spat", adj_mat_path=adj_mat_spat_path,
                                                                       id2idx_path=id2idx_spat_path,
                                                                       ne_dim=ne_dim_spat, we_dim=word_emb_size)
    if not competitor:
        forest_rel = load_from_pickle(join(mod_dir_rel, "forest_{}_{}.h5".format(ne_dim_rel, word_emb_size)))
        forest_spat = load_from_pickle(join(mod_dir_spat, "forest_{}_{}.h5".format(ne_dim_spat, word_emb_size)))
        name = "mlp_{}".format(word_emb_size)
        if consider_rel:
            name += "_rel_{}".format(ne_dim_rel)
        if consider_spat:
            name += "_spat_{}".format(ne_dim_spat)
        mlp = load_from_pickle(join(models_dir, name + ".pkl"))
        print(name.upper())
        test(df_train=train_df, df=test_df, w2v_model=w2v_model, ae_dang=dang_ae, ae_safe=safe_ae, tree_rel=forest_rel,
             tree_spat=forest_spat, mlp=mlp, ne_technique_rel=ne_technique_rel, ne_technique_spat=ne_technique_spat,
             id2idx_rel=id2idx_rel, id2idx_spat=id2idx_spat, mod_rel=mod_rel, mod_spat=mod_spat, rel_net_path=path_rel,
             spat_net_path=path_spat, field_text=field_text, field_id=field_id, field_label=field_label,
             consider_rel=consider_rel, consider_spat=consider_spat, cls_competitor=None)
    else:
        mlp = None
        while not (consider_rel and consider_spat):
            consider_rel = not consider_rel
            if not consider_rel:
                consider_spat = not consider_spat
                if not consider_spat:
                    consider_content = not consider_content
            cls_competitor = None
            if competitor:
                name = "forest"
                if consider_content:
                    name += "_content_{}".format(word_emb_size)
                if consider_rel:
                    name += "_rel_{}".format(ne_dim_rel)
                if consider_spat:
                    name += "_spat_{}".format(ne_dim_spat)
                cls_competitor = load_from_pickle(join(models_dir, "competitors", name + ".pkl"))
                print(name.upper())
            test(df_train=train_df, df=test_df, w2v_model=w2v_model, ae_dang=dang_ae, ae_safe=safe_ae,
                 tree_rel=forest_rel, tree_spat=forest_spat, ne_technique_rel=ne_technique_rel,
                 ne_technique_spat=ne_technique_spat, id2idx_rel=id2idx_rel, id2idx_spat=id2idx_spat, mod_rel=mod_rel,
                 mod_spat=mod_spat, rel_net_path=path_rel, spat_net_path=path_spat, field_text=field_text,
                 field_id=field_id, field_label=field_label, consider_rel=consider_rel, consider_spat=consider_spat,
                 cls_competitor=cls_competitor, mlp=mlp)
            print("\n\n")
            break


if __name__ == "__main__":
    """parser = argparse.ArgumentParser()
    parser.add_argument("--spat_technique", type=str, choices=['node2vec', 'none', 'autoencoder', 'pca'], required=True, help="Technique adopted for learning spatial node embeddings")
    parser.add_argument("--rel_technique", type=str, choices=['node2vec', 'none', 'autoencoder', 'pca'], required=True, help="Technique adopted for learning relational node embeddings")
    parser.add_argument("--dataset_dir", type=str, default="", required=True, help="Directory containing the train and test set")
    parser.add_argument("--spat_ne_size", type=int, default=128, required=True, help="Dimension of spatial node embeddings to use")
    parser.add_argument("--rel_ne_size", type=int, default=128, required=True, help="Dimension of relational node embeddings to use")
    parser.add_argument("--models_dir", type=str, default="", required=True, help="Directory where the models are saved")
    parser.add_argument("--spat_adj_mat_path", type=str, required=False, help="Link to the file containing the spatial adjacency matrix. Ignore this parameter if you used n2v for learning spatial node embeddings")
    parser.add_argument("--rel_adj_mat_path", type=str, required=False, help="Link to the file containing the relational adjacency matrix. Ignore this parameter if you used n2v for learning relational node embeddings")
    parser.add_argument("--word_embedding_size", type=int, required=True, default=128, help="Dimension of the word embeddings learned during the training process")
    parser.add_argument("--id2idx_spat_path", type=str, required=False, help="Link to the .pkl file with the matchings between node IDs and their index in the spatial adj matrix. Ignore this parameter if you used n2v for spatial node embeddings")
    parser.add_argument("--id2idx_rel_path", type=str, required=False, help="Link to the .pkl file with the matchings between node IDs and their index in the relational adj matrix. Ignore this parameter if you used n2v for relational node embeddings")
    parser.add_argument("--user_id", type=int, required=False, help="ID of the user that you want to predict. If you set this field, only the prediction for the user will be returned."
                                                                    "Ignore this field if you want to measure the performance of the system on the test set")

    args = parser.parse_args()
    main_test(args)"""
    main_test()
