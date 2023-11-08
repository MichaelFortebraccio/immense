import argparse
from modelling.sairus import test, predict_user
from keras.models import load_model
from node_classification.decision_tree import load_decision_tree
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
    spat_technique = rel_technique = "graphsage"
    dataset_dir = "dataset/big_dataset"
    models_dir = "dataset/big_dataset/models"
    graph_dir = join(dataset_dir, "graph")

    adj_mat_spat_path = adj_mat_rel_path = None
    id2idx_rel_path = join(models_dir, "id2idx_rel.pkl")
    id2idx_spat_path = join(models_dir, "id2idx_spat.pkl")
    word_embedding_size = 512
    spat_ne_dim = rel_ne_dim = 256
    rel_net_path = join(graph_dir, "social_network_test.edg")
    spat_net_path = join(graph_dir, "spatial_network_test.edg")
    """
    train_df = pd.read_csv(join(dataset_dir, "train.csv"))
    test_df = pd.read_csv(join(dataset_dir, "test.csv"))
    w2v_model = load_from_pickle(join(models_dir, "w2v_{}.pkl".format(word_embedding_size)))
    
    train_df = pd.read_csv(join(dataset_dir, "train_089_{}.csv".format(val)))
    test_df = pd.read_csv(join(dataset_dir, "test_089_{}.csv".format(val)))
    w2v_model = load_from_pickle(join(models_dir, "w2v_{}_089_{}.pkl".format(word_embedding_size, val)))
    """
    train_df = pd.read_csv(join(dataset_dir, "train.csv"))
    test_df = pd.read_csv(join(dataset_dir, "test.csv"))
    w2v_model = load_from_pickle(join(models_dir, "w2v_{}.pkl".format(word_embedding_size)))

    dang_ae = load_model(join(models_dir, "autoencoderdang_{}.h5".format(word_embedding_size)))
    safe_ae = load_model(join(models_dir, "autoencodersafe_{}.h5".format(word_embedding_size)))
    mlp = load_from_pickle(join(models_dir, "mlp.pkl"))

    mod_dir_rel = join(models_dir, "node_embeddings", "rel")
    mod_dir_spat = join(models_dir, "node_embeddings", "spat")
    """tree_rel = load_decision_tree(join(mod_dir_rel, "dtree_{}_{}.h5".format(rel_technique, rel_ne_dim)))
    tree_spat = load_decision_tree(join(mod_dir_spat, "dtree_{}_{}.h5".format(spat_technique, spat_ne_dim)))

    mod_rel, pca_rel, ae_rel, adj_mat_rel, id2idx_rel = get_model(technique=rel_technique, mod_dir=mod_dir_rel,
                                                                  lab="rel", adj_mat_path=adj_mat_rel_path,
                                                                  id2idx_path=id2idx_rel_path, ne_dim=rel_ne_dim)

    mod_spat, pca_spat, ae_spat, adj_mat_spat, id2idx_spat = get_model(technique=spat_technique, mod_dir=mod_dir_spat,
                                                                       lab="spat", adj_mat_path=adj_mat_spat_path,
                                                                       id2idx_path=id2idx_spat_path, ne_dim=spat_ne_dim)"""
    tree_rel = tree_spat = ae_rel = ae_spat = id2idx_rel = id2idx_spat = adj_mat_rel = adj_mat_spat = mod_rel = mod_spat = pca_rel = pca_spat = None

    """if args.user_id:
        df = train_df.append(test_df)
        user = df.loc[df.id==args.user_id]
        pred = predict_user(user=user, w2v_model=w2v_model, dang_ae=dang_ae, safe_ae=safe_ae, df=train_df, tree_rel=tree_rel,
                            tree_spat=tree_spat, mlp=mlp, rel_node_emb_technique=rel_technique, spat_node_emb_technique=spat_technique, id2idx_rel=id2idx_rel,
                            id2idx_spat=id2idx_spat, n2v_rel=n2v_rel, n2v_spat=n2v_spat, pca_rel=pca_rel, pca_spat=pca_spat, ae_rel=ae_rel, ae_spat=ae_spat,
                            adj_matrix_rel=adj_mat_rel, adj_matrix_spat=adj_mat_spat)
        print("The user is: {}".format("risky" if pred == 1 else "safe"))
    else:"""
    test(train_df=train_df, df=test_df, w2v_model=w2v_model, dang_ae=dang_ae, safe_ae=safe_ae, tree_rel=tree_rel,
         tree_spat=tree_spat, mlp=mlp, ae_rel=ae_rel, ae_spat=ae_spat, rel_ne_technique=rel_technique,
         spat_ne_technique=spat_technique, id2idx_rel=id2idx_rel, id2idx_spat=id2idx_spat, adj_matrix_rel=adj_mat_rel,
         adj_matrix_spat=adj_mat_spat, mod_rel=mod_rel, mod_spat=mod_spat, pca_rel=pca_rel, pca_spat=pca_spat,
         rel_net_path=rel_net_path, spat_net_path=spat_net_path, text_field_name="text_cleaned", id_field_name="id")


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
