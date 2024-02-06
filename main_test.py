import argparse
from modelling.sairus import test
from os.path import join
from utils import load_from_pickle, get_model
import pandas as pd


def main_test(args=None):
    """spat_technique = args.spat_technique
    rel_technique = args.rel_technique
    dataset_dir = args.dataset_dir
    models_dir = args.models_dir
    adj_mat_rel_path = args.rel_adj_mat_path
    adj_mat_spat_path = args.spat_adj_mat_path
    id2idx_spat_path = args.id2idx_spat_path
    id2idx_rel_path = args.id2idx_rel_path
    we_size = args.word_embedding_size
    spat_ne_dim = args.spat_ne_size
    rel_ne_dim = args.rel_ne_size"""

    # For testing purposes
    dataset_dir = join("dataset", "big_dataset")
    graph_dir = join(dataset_dir, "graph")
    models_dir = join("dataset", "anthony", "models")   # , "only_spatial"
    id_field = "id"
    text_field = "text_cleaned"

    train_df = pd.read_csv(join("dataset", "anthony", "train.csv"))
    test_df = pd.read_csv(join(dataset_dir, "tweets.csv"))     # tweets_only_withpos.csv

    mod_dir_rel = join(models_dir, "node_embeddings", "rel")
    mod_dir_spat = join(models_dir, "node_embeddings", "spat")
    technique_spat = technique_rel = "graphsage"

    adj_mat_spat_path = adj_mat_rel_path = None
    id2idx_rel_path = join(models_dir, "id2idx_rel.pkl")
    id2idx_spat_path = join(models_dir, "id2idx_spat.pkl")
    rel_net_path = join(graph_dir, "social_network.edg")   # "social_network_both_spatial.edg"
    spat_net_path = join(graph_dir, "spatial_network.edg")      # spatial_network_nonzero.edg

    word_embedding_size = 512
    ne_dim_spat = ne_dim_rel = 256

    w2v_model = load_from_pickle(join(models_dir, "w2v_{}.pkl".format(word_embedding_size)))

    dang_ae = load_from_pickle(join(models_dir, "autoencoderdang_{}.pkl".format(word_embedding_size)))
    safe_ae = load_from_pickle(join(models_dir, "autoencodersafe_{}.pkl".format(word_embedding_size)))

    competitor = False
    consider_content = False
    consider_rel = False
    consider_spat = False

    mod_rel = pca_rel = ae_rel = adj_mat_rel = id2idx_rel = mod_spat = pca_spat = ae_spat = adj_mat_spat = id2idx_spat = forest_rel = forest_spat = None

    mod_rel, pca_rel, ae_rel, adj_mat_rel, id2idx_rel = get_model(technique=technique_rel, mod_dir=mod_dir_rel,
                                                                  lab="rel", adj_mat_path=adj_mat_rel_path,
                                                                  id2idx_path=id2idx_rel_path, ne_dim=ne_dim_rel,
                                                                  we_dim=word_embedding_size)
    mod_spat, pca_spat, ae_spat, adj_mat_spat, id2idx_spat = get_model(technique=technique_spat,
                                                                       mod_dir=mod_dir_spat,
                                                                       lab="spat", adj_mat_path=adj_mat_spat_path,
                                                                       id2idx_path=id2idx_spat_path,
                                                                       ne_dim=ne_dim_spat, we_dim=word_embedding_size)
    if not competitor:
        forest_rel = load_from_pickle(join(mod_dir_rel, "forest_{}_{}.h5".format(ne_dim_rel, word_embedding_size)))
        forest_spat = load_from_pickle(join(mod_dir_spat, "forest_{}_{}.h5".format(ne_dim_spat, word_embedding_size)))
        stop = False
        while not stop:   # not (consider_rel and consider_spat)
            name = "mlp"
            if consider_rel:
                name += "_rel"
            if consider_spat:
                name += "_spat"
            mlp = load_from_pickle(join(models_dir, name + ".pkl"))
            print(name.upper())
            test(df_train=train_df, df=test_df, w2v_model=w2v_model, ae_dang=dang_ae, ae_safe=safe_ae, tree_rel=forest_rel,
                 tree_spat=forest_spat, mlp=mlp, ne_technique_rel=technique_rel, ne_technique_spat=technique_spat,
                 id2idx_rel=id2idx_rel, id2idx_spat=id2idx_spat, mod_rel=mod_rel, mod_spat=mod_spat,
                 rel_net_path=rel_net_path, spat_net_path=spat_net_path, field_text=text_field, field_id=id_field,
                 consider_rel=consider_rel, consider_spat=consider_spat, cls_competitor=None)
            if consider_rel and consider_spat:
                stop = True
            else:
                consider_spat = not consider_spat
                if not consider_spat:
                    consider_rel = not consider_rel

    else:
        mlp = None
        while not (consider_content and consider_rel and consider_spat):
            consider_spat = not consider_spat
            if not consider_spat:
                consider_rel = not consider_rel
                if not consider_rel:
                    consider_content = not consider_content
            cls_competitor = None
            if competitor:
                name = "forest"
                if consider_content:
                    name += "_content_{}".format(word_embedding_size)
                if consider_rel:
                    name += "_rel_{}".format(ne_dim_rel)
                if consider_spat:
                    name += "_spat_{}".format(ne_dim_spat)
                cls_competitor = load_from_pickle(join(models_dir, "competitors", name+".pkl"))
                print(name.upper())
            test(df_train=train_df, df=test_df, w2v_model=w2v_model, ae_dang=dang_ae, ae_safe=safe_ae, tree_rel=forest_rel,
                 tree_spat=forest_spat, mlp=mlp, ne_technique_rel=technique_rel, ne_technique_spat=technique_spat,
                 id2idx_rel=id2idx_rel, id2idx_spat=id2idx_spat, mod_rel=mod_rel, mod_spat=mod_spat,
                 rel_net_path=rel_net_path, spat_net_path=spat_net_path, field_text=text_field, field_id=id_field,
                 consider_rel=consider_rel, consider_spat=consider_spat, cls_competitor=cls_competitor)
            print("\n\n")


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

#TODO PRENDERE IL DATASET MIO, ISOLARE GLI UTENTI CHE HANNO LA POSIZIONE E TESTARE SU QUESTO IL MODELLO APPRESO SUL DATASET ANTHONY. RIFARE I TEST RIADDESTRANDO IL MODELLO E CONSIDERANDO I PESI ALLE CLASSI