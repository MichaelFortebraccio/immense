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
         walk_length_spat=10, walk_length_rel=10, p_spat=1, p_rel=1, q_spat=4, q_rel=4, n2v_epochs_spat=100, n2v_epochs_rel=100):
    dataset_dir = "dataset"
    models_dir = "models"
    if not exists(dataset_dir):
        makedirs(dataset_dir)
    if not exists(models_dir):
        makedirs(models_dir)
    posts_path = "{}/posts_labeled.csv".format(dataset_dir)
    social_path = "{}/social_network.edg".format(dataset_dir)
    closeness_path = "{}/closeness_network.edg".format(dataset_dir)
    if not exists(posts_path):
        gdown.download(url=textual_content_link, output=posts_path, quiet=False, fuzzy=True)
    if not exists(social_path):
        gdown.download(url=social_net_url, output=social_path, quiet=False, fuzzy=True)
    if not exists(closeness_path):
        gdown.download(url=spatial_net_url, output=closeness_path, quiet=False, fuzzy=True)

    train_path = "{}/train.csv".format(dataset_dir)
    test_path = "{}/test.csv".format(dataset_dir)
    df = pd.read_csv(posts_path, sep=',')
    if not exists(train_path) or not exists(test_path):
        df = df.sample(frac=1, random_state=1).reset_index()    # Shuffle the dataframe
        idx = round(len(df)*0.8)
        train_df = df[:idx]
        test_df = df[idx:]
        train_df = train_df.reset_index()
        test_df = test_df.reset_index()
        train_df.to_csv(train_path)
        test_df.to_csv(test_path)
    else:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
    
    dang_ae, safe_ae, w2v_model, n2v_rel, n2v_spat, tree_rel, tree_clos, mlp = train(train_df=train_df, dataset_dir=dataset_dir,
          model_dir=models_dir, social_path=social_path, spatial_path=closeness_path,
          word_embedding_size=word_embedding_size, window=window, w2v_epochs=w2v_epochs, n_of_walks_spat=n_of_walks_spat,
          n_of_walks_rel=n_of_walks_rel, walk_length_spat=walk_length_spat, walk_length_rel=walk_length_rel,
          spat_node_embedding_size=spat_node_embedding_size, rel_node_embedding_size=rel_node_embedding_size, p_spat=p_spat,
          p_rel=p_rel, q_spat=q_spat, q_rel=q_rel, n2v_epochs_spat=n2v_epochs_spat, n2v_epochs_rel=n2v_epochs_rel)

    test(test_df=test_df, train_df=train_df, w2v_model=w2v_model, dang_ae=dang_ae, safe_ae=safe_ae, tree_rel=tree_rel,
         tree_spat=tree_clos, n2v_rel=n2v_rel, n2v_spat=n2v_spat, mlp=mlp)


if __name__ == "__main__":
    main("textual_content_link", social_net_url="https://drive.google.com/file/d/1MhSo9tMDkfnlvZXPKgxv-HBLP4fSwvmN/view?usp=sharing",
         spatial_net_url="https://drive.google.com/file/d/1fVipJMfIoqVqnImc9l79tLqzTWlhhXCq/view?usp=sharing", word_embedding_size=512, window=5, w2v_epochs=1,
         spat_node_embedding_size=128, rel_node_embedding_size=128, n_of_walks_spat=10, n_of_walks_rel=10,
         walk_length_spat=10, walk_length_rel=10, p_spat=1, p_rel=1, q_spat=4, q_rel=4, n2v_epochs_spat=100, n2v_epochs_rel=100)

