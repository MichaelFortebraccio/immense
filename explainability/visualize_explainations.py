import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from lime.lime_tabular import LimeTabularExplainer
from PIL import Image
from utils import load_from_pickle


def images_to_pdf(image_dir):
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    # Open all images and convert to RGB (PDFs need RGB mode)
    images = [Image.open(os.path.join(image_dir, f)).convert('RGB') for f in image_files]
    # Save as a single PDF
    output_path = os.path.join(image_dir, "output.pdf")
    images[0].save(output_path, save_all=True, append_images=images[1:])
    print(f"PDF saved to {output_path}")


def explain_single_instance(idx, xtest, mlp, explainer, loss, dim, actual_label, image_dir, label):
    test = xtest[idx].detach().cpu().numpy()
    explanation = explainer.explain_instance(test, mlp.predict_proba, labels=(0, 1), actual_label=label, idx=idx)

    # IMPORTANT! In the next instruction, the implementation of the as_pyplot_figure function was modified to accept additional parameters
    fig = explanation.as_pyplot_figure(label=actual_label, dim=dim, loss=loss)

    dst_dir = os.path.join(image_dir, f"user_{idx}")
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    fig.savefig(os.path.join(dst_dir, f"user_{idx}_{loss}_{dim}.png"))


def explain_dataset(aggregate_by_modality, mlp, x_test,explainer):
    print(dim, loss)
    modality_importance_list = []
    for i in tqdm(range(x_test.shape[0])):
        instance = x_test[i]
        instance_np = instance.detach().cpu().numpy()

        explanation = explainer.explain_instance(instance_np, mlp.predict_proba)
        weights = explanation.as_list()
        importance_weights = {}
        # Aggregate per modality
        for feature_name, weight in weights:
            if not aggregate_by_modality:
                if "safe_loss_ae" in feature_name:
                    importance_weights["safe_ae_loss"] += weight
                elif "risky_loss_ae" in feature_name:
                    importance_weights["risky_ae_loss"] += weight
                elif "label_ae" in feature_name:
                    importance_weights["label_ae"] += weight
                elif "p_safe_rel" in feature_name:
                    importance_weights["p_safe_rel"] += weight
                elif "p_risky_rel" in feature_name:
                    importance_weights["p_risky_rel"] += weight
                elif "p_safe_spat" in feature_name:
                    importance_weights["p_safe_spat"] += weight
                elif "p_risky_sapt" in feature_name:
                    importance_weights["p_risky_sapt"] += weight
            else:
                if "text_" in feature_name:
                    importance_weights["text"] += weight
                elif "rel_" in feature_name:
                    importance_weights["relation"] += weight
                elif "spatial_" in feature_name:
                    importance_weights["spatial"] += weight

        modality_importance_list.append(importance_weights)
        df = pd.DataFrame(modality_importance_list)
        return df


if __name__ == "__main__":
    aggregate_by_modality = False
    image_dir = "LIME/plots/pngs/"
    path_to_train = "."
    path_to_test = "."
    df_test = pd.read_csv("../dataset/big_dataset/test.csv")

    if aggregate_by_modality:
        feature_names = [f"text_{i}" for i in range(3)] + [f"rel_{i}" for i in range(3, 5)] + [
            f"spatial_{i}" for i in range(5, 7)]
    else:
        feature_names = [r'$R_s(u)$', r"$R_r(u)$", r"$L^{con}(u)$", r"$P^{rel}_s(u)$", r"$P^{rel}_r(u)$", r"$P^{sp}_s(u)$",
                         r"$P^{sp}_r(u)$"]
    """
    for loss in ["focal", "weighted"]:
        for dim in [128, 256, 512]:
            X_train = load_from_pickle(os.path.join(path_to_train, "X_train_{}_{}.pkl".format(dim, loss)))
            x_test = load_from_pickle(os.path.join(path_to_test, "x_test_{}_{}.pkl".format(dim, loss)))
            mlp = load_from_pickle(
                "../../dataset/big_dataset/models/mlp/mlp_content_{}_rel_{}_spat_{}_{}.pkl".format(dim, dim, dim, loss))
            explainer = LimeTabularExplainer(training_data=np.array(X_train), feature_names=feature_names,
                                             mode="classification")
            df = explain_dataset(aggregate_by_modality, mlp=mlp, x_test=x_test, explainer=explainer)
            df.to_csv(f"boxplot_df_{loss}_{dim}_{aggregate_by_modality}.csv")

            df = pd.read_csv(f"boxplot_df_{loss}_{dim}_{aggregate_by_modality}.csv")
            df = df.drop(columns=["Unnamed: 0"])
            fig, axes = plt.subplots(figsize=(15, 6))
            sns.boxplot(data=df, showfliers=False)
            if dim in [256, 512]:
                ran = np.arange(-0.4, 0.4, 0.1)
            else:
                ran = np.arange(-0.7, 0.7, 0.1)
            plt.title("{}_{}".format(loss, dim))
            plt.ylabel("LIME Importance")
            # plt.yticks(ran)
            plt.xlabel("Modality")
            plt.axhline(0, color='gray', linestyle='--')
            plt.tight_layout()
            plt.savefig(os.path.join(f"{loss}_{dim}.png"))
            plt.show()"""

    for node_user in [(689, 1)]:   # node_user = (user_idx, user_label)
        for loss in ["focal", 'weighted']:  #"weighted"
            #for idx, row in df_test.iterrows():
            for dim in [128, 256, 512]:
                X_train = load_from_pickle(os.path.join(path_to_train, "X_train_{}_{}.pkl".format(dim, loss)))
                x_test = load_from_pickle(os.path.join(path_to_test, "x_test_{}_{}.pkl".format(dim, loss)))
                mlp = load_from_pickle(
                    "../../dataset/big_dataset/models/mlp/mlp_content_{}_rel_{}_spat_{}_{}.pkl".format(dim, dim, dim, loss))
                explainer = LimeTabularExplainer(training_data=np.array(X_train), feature_names=feature_names, mode="classification")

                """label = row["label"]
                node_user = (idx, label)"""
                explain_single_instance(idx=node_user[0], xtest=x_test, mlp=mlp, explainer=explainer, loss=loss,
                                        dim=dim, actual_label=node_user[1], image_dir=image_dir, label=node_user[1])
