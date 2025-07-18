import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from utils import save_to_pickle
from sklearn.manifold import TSNE

from utils import load_from_pickle


def embeddings_tsne(matrix, dst_dir, mode, size):
    dst_path = os.path.join(dst_dir, f"{mode}_{size}.pkl")
    tsne = TSNE(n_components=2, random_state=42)
    if os.path.exists(dst_path):
        tsne_embs = load_from_pickle(dst_path)
    else:
        tsne_embs = tsne.fit_transform(matrix)
        save_to_pickle(os.path.join(dst_dir, f"{mode}_{size}.pkl"), tsne_embs)
    return tsne_embs


def get_textual_embeddings(emb_size, users_embs_dict):
    emb_matrix = np.array(list(users_embs_dict.values()))
    tsne_embs = embeddings_tsne(emb_matrix, "tsne", "text", emb_size)
    plot_df = pd.DataFrame(tsne_embs, columns=['x', 'y'])
    plot_df['label'] = train_df['label'].values
    return plot_df


def get_rel_preds(node_emb_dim, loss, typerel, graph_model, graph):
    graph_model = graph_model.to(torch.device("cpu"))
    embs = graph_model(graph, inference_for_embedding=True).detach().numpy()
    tsne_embs = embeddings_tsne(embs, "tsne", typerel, str(node_emb_dim) + f"_{loss}")
    plot_df = pd.DataFrame(tsne_embs, columns=['x', 'y'])
    plot_df['label'] = graph.y
    return plot_df


def plot_image(df, savepath):
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df, x='x', y='y', hue='label', palette='tab10', s=60)
    # plt.title(f"{typerel} Embeddings, ks={size}")
    plt.xlabel("")
    plt.ylabel("")
    plt.tick_params(axis='both', labelsize=28)
    plt.yticks()
    plt.legend().remove()
    plt.tight_layout()
    plt.savefig(savepath)
    plt.show()


if __name__ == "__main__":
    emb_sizes = [128, 256, 512]
    models_dir = "../dataset/big_dataset/models"
    train_df = pd.read_csv("../dataset/big_dataset/train.csv")
    for n in ["plots", "tsne"]:
        if not os.path.exists(n):
            os.makedirs(n)

    for size in emb_sizes:
        users_embs_dict = load_from_pickle(f"users_emb_dict_{size}.pkl")
        df = get_textual_embeddings(size, users_embs_dict)
        plot_image(df, os.path.join("plots", f"textual_{size}"))

    for typerel in ["spat", "rel"]:
        for size in emb_sizes:
            for loss in ["weighted", "focal"]:
                graph_model = load_from_pickle(
                    f"../dataset/big_dataset/models/node_embeddings/{typerel}/graphsage_{size}_{size}_{loss}.pkl")
                graph = load_from_pickle(f"../graph_{typerel}_{size}.pkl")
                df = get_rel_preds(size, loss, typerel, graph_model, graph)
                plt.figure(figsize=(10, 8))
                plot_image(df, os.path.join("plots", f"{typerel}_{size}_{loss}"))

    for size in emb_sizes:
        for loss in ["weighted", "focal"]:
            X_train = load_from_pickle("../X_train_{}_{}.pkl".format(size, loss))
            tsne_embs = embeddings_tsne(X_train, "tsne", "mlp", str(size) + f"_{loss}")
            plot_df = pd.DataFrame(tsne_embs, columns=['x', 'y'])
            plot_df['label'] = train_df['label'].values
            plot_image(plot_df, os.path.join("plots", f"mlp_{size}_{loss}"))
