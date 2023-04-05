import pandas as pd
import numpy as np
from os.path import exists
from os import makedirs
from modelling.sairus import train, test
import gdown
seed = 123
np.random.seed(seed)


def main(textual_content_link, social_net_url, spatial_net_url, word_embedding_size=512, window=5, w2v_epochs=10,
         spat_node_embedding_size=128, rel_node_embedding_size=128, n_of_walks_spat=10, n_of_walks_rel=10,
         walk_length_spat=10, walk_length_rel=10, p_spat=1, p_rel=1, q_spat=4, q_rel=4, n2v_epochs_spat=100, n2v_epochs_rel=100, technique="node2vec",
         rel_adj_mat_path=None, spat_adj_mat=None):
    dataset_dir = "dataset"
    models_dir = "models"
    if not exists(dataset_dir):
        makedirs(dataset_dir)
    if not exists(models_dir):
        makedirs(models_dir)
    posts_path = "{}/posts_labeled.csv".format(dataset_dir)
    social_path = "{}/social_network.edg".format(dataset_dir)
    closeness_path = "{}/spatial_network.edg".format(dataset_dir)
    if not exists(posts_path):
        gdown.download(url=textual_content_link, output=posts_path, quiet=False, fuzzy=True)
    if not exists(social_path):
        gdown.download(url=social_net_url, output=social_path, quiet=False, fuzzy=True)
    if not exists(closeness_path):
        gdown.download(url=spatial_net_url, output=closeness_path, quiet=False, fuzzy=True)
    train_path = "{}/train.csv".format(dataset_dir)
    test_path = "{}/test.csv".format(dataset_dir)
    df = pd.read_csv(posts_path, sep=',')
    cols = ['index', 'label', 'id', 'text_cleaned']
    if not exists(train_path) or not exists(test_path):
        shuffled_df = df.sample(frac=1, random_state=1).reset_index()    # Shuffle the dataframe
        shuffled_df = shuffled_df.drop(columns=[col for col in shuffled_df.columns if col not in cols])
        shuffled_df.to_csv("dataset/shuffled_content.csv")
        idx = round(len(shuffled_df)*0.8)
        train_df = shuffled_df[:idx]
        test_df = shuffled_df[idx:]
        train_df = train_df.reset_index()
        test_df = test_df.reset_index()
        train_df = train_df.drop(columns=[col for col in train_df.columns if col not in cols])
        test_df = test_df.drop(columns=[col for col in test_df.columns if col not in cols])
        train_df.to_csv(train_path)
        test_df.to_csv(test_path)
    else:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
    
    dang_ae, safe_ae, w2v_model, tree_rel, tree_spat, mlp = train(
        train_df=train_df, full_df=df, dataset_dir=dataset_dir, model_dir=models_dir, rel_path=social_path, spatial_path=closeness_path,
        word_embedding_size=word_embedding_size, window=window, w2v_epochs=w2v_epochs, n_of_walks_spat=n_of_walks_spat,
        n_of_walks_rel=n_of_walks_rel, walk_length_spat=walk_length_spat, walk_length_rel=walk_length_rel,
        spat_node_embedding_size=spat_node_embedding_size, rel_node_embedding_size=rel_node_embedding_size, p_spat=p_spat,
        p_rel=p_rel, q_spat=q_spat, q_rel=q_rel, n2v_epochs_spat=n2v_epochs_spat, n2v_epochs_rel=n2v_epochs_rel,
        node_emb_technique=technique, adj_matrix_spat_path=spat_adj_mat, adj_matrix_rel_path=rel_adj_mat_path)

    test(test_df=test_df, train_df=train_df, w2v_model=w2v_model, dang_ae=dang_ae, safe_ae=safe_ae, tree_rel=tree_rel,
         tree_spat=tree_spat, mlp=mlp)


if __name__ == "__main__":
    main("textual_content_link", social_net_url="https://drive.google.com/file/d/1MhSo9tMDkfnlvZXPKgxv-HBLP4fSwvmN/view?usp=sharing",
         spatial_net_url="https://drive.google.com/file/d/1fVipJMfIoqVqnImc9l79tLqzTWlhhXCq/view?usp=sharing", word_embedding_size=512, window=5, w2v_epochs=1,
         spat_node_embedding_size=128, rel_node_embedding_size=128, n_of_walks_spat=10, n_of_walks_rel=10,
         walk_length_spat=10, walk_length_rel=10, p_spat=1, p_rel=1, q_spat=4, q_rel=4, n2v_epochs_spat=100, n2v_epochs_rel=100,
         technique="pca", spat_adj_mat="node_classification/graph_embeddings/stuff/spat_adj_net.csv",
         rel_adj_mat_path="node_classification/graph_embeddings/stuff/adj_net.csv")

