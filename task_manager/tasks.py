#sys.path.append('../')
import os
import pandas as pd
import hdfs
from gensim.models import Word2Vec
from modelling.sairus import train_w2v_model, learn_mlp
from modelling.ae import AE
from node_classification.decision_tree import train_decision_tree, load_decision_tree
from node_classification.reduce_dimension import dimensionality_reduction
from task_manager.worker import celery
from utils import load_from_pickle, save_to_pickle
from dataset_scripts.dataset_utils import concatenate_posts, clean_dataframe


CONTENT_FILENAME = "content_labeled.csv"
REL_EDGES_FILENAME = "social_network.edg"
SPAT_EDGES_FILENAME = "spatial_network.edg"
ID2IDX_REL_FILENAME = "id2idx_rel.pkl"
ID2IDX_SPAT_FILENAME = "id2idx_spat.pkl"
REL_ADJ_MAT_FILENAME = "rel_adj_net.csv"
SPAT_ADJ_MAT_FILENAME = "spat_adj_net.csv"
JOBS_DIR = "jobs"
MODEL_DIR = JOBS_DIR+"/{}/models"
DATASET_DIR = JOBS_DIR+"/{}/dataset"
WORD_EMB_SIZE = 0

HDFS_HOST = "http://" + os.getenv("HDFS_HOST") + ":" + os.getenv("HDFS_PORT")

client = hdfs.InsecureClient(HDFS_HOST, timeout=60, user="root")
@celery.task(bind=True)
def train_task(self, content_url, word_embedding_size, window, w2v_epochs, rel_node_emb_technique: str, spat_node_emb_technique: str,
               rel_node_embedding_size, spat_node_embedding_size, social_network_url=None, spatial_network_url=None, n_of_walks_rel=None, n_of_walks_spat=None,
               walk_length_rel=None, walk_length_spat=None, p_rel=None, p_spat=None, q_rel=None, q_spat=None, n2v_epochs_rel=None, n2v_epochs_spat=None,
               spat_ae_epochs=None, rel_ae_epochs=None, adj_matrix_rel_url=None, adj_matrix_spat_url=None, id2idx_rel_url=None, id2idx_spat_url=None):
    job_id = self.request.id
    dataset_dir = DATASET_DIR.format(job_id)
    model_dir = MODEL_DIR.format(job_id)

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(dataset_dir, exist_ok=True)

    content_path = os.path.join(dataset_dir, CONTENT_FILENAME)

    with open(os.path.join(JOBS_DIR, job_id, "techniques.txt"), "w") as f:
        f.write(rel_node_emb_technique+"\n")
        f.write(spat_node_emb_technique)
    ############### DOWNLOAD FILES ###############
    if not os.path.exists(content_path):
        self.update_state(state="PROGRESS", meta={"status": "Downloading content..."})
        if not client.content(content_path):
            raise FileNotFoundError("The URL to the content file does not exist")
        client.download(hdfs_path=content_url, local_path=content_path)

    rel_edges_path = None
    spat_edges_path = None
    rel_adj_mat_path = None
    spat_adj_mat_path = None
    id2idx_rel_path = None
    id2idx_spat_path = None

    if rel_node_emb_technique == "node2vec":
        if not social_network_url:
            raise Exception("You need to provide a URL to the relational edge list")
        rel_edges_path = os.path.join(dataset_dir, REL_EDGES_FILENAME)
        if not os.path.exists(rel_edges_path):
            self.update_state(state="PROGRESS", meta={"status": "Downloading social edge list..."})
            client.download(hdfs_path=social_network_url, local_path=rel_edges_path)
    elif rel_node_emb_technique in ["pca", "autoencoder", "none"]:
        if not adj_matrix_rel_url:
            raise Exception("You need to provide the URL to the relational adjacency matrix")
        if not id2idx_rel_url:
            raise Exception("You need to provide the URL to the relational id2idx file")
        rel_adj_mat_path = os.path.join(dataset_dir, REL_ADJ_MAT_FILENAME)
        id2idx_rel_path = os.path.join(dataset_dir, ID2IDX_REL_FILENAME)
        if not os.path.exists(rel_adj_mat_path):
            self.update_state(state="PROGRESS", meta={"status": "Downloading social adjacency matrix..."})
            client.download(hdfs_path=adj_matrix_rel_url, local_path=rel_adj_mat_path)
        if not os.path.exists(id2idx_rel_path):
            self.update_state(state="PROGRESS", meta={"status": "Downloading social id2idx_rel..."})
            client.download(hdfs_path=id2idx_rel_url, local_path=id2idx_rel_path)
    if spat_node_emb_technique == "node2vec":
        if not spatial_network_url:
            raise Exception("You need to provide a URL to the spatial network edge list")
        spat_edges_path = os.path.join(dataset_dir, SPAT_EDGES_FILENAME)
        if not os.path.exists(spat_edges_path):
            self.update_state(state="PROGRESS", meta={"status": "Downloading spatial edge list..."})
            client.download(hdfs_path=spatial_network_url, local_path=spat_edges_path)
    elif spat_node_emb_technique in ["pca", "autoencoder", "none"]:
        spat_adj_mat_path = os.path.join(dataset_dir, SPAT_ADJ_MAT_FILENAME)
        id2idx_spat_path = os.path.join(dataset_dir, ID2IDX_SPAT_FILENAME)
        if not adj_matrix_spat_url:
            raise Exception("You need to provide the URL to the spatial adjacency matrix")
        if not id2idx_spat_url:
            raise Exception("You need to provide the URL to the spatial id2idx file")
        if not os.path.exists(spat_adj_mat_path):
            self.update_state(state="PROGRESS", meta={"status": "Downloading spatial adjacency matrix..."})
            client.download(hdfs_path=adj_matrix_spat_url, local_path=spat_adj_mat_path)
        if not os.path.exists(id2idx_spat_path):
            self.update_state(state="PROGRESS", meta={"status": "Downloading spatial id2idx_rel..."})
            client.download(hdfs_path=id2idx_spat_url, local_path=id2idx_spat_path)
    self.update_state(state="PROGRESS", meta={"status": "Dataset successfully downloaded."})

    train_df = pd.read_csv(content_path, sep=',').reset_index()

    self.update_state(state="PROGRESS", meta={"status": "Learning w2v model."})
    list_dang_posts, list_safe_posts, list_embs = train_w2v_model(train_df, embedding_size=word_embedding_size, window=window,
                                                                  epochs=w2v_epochs, model_dir=model_dir, dataset_dir=dataset_dir)
    self.update_state(state="PROGRESS", meta={"status": "Learning dangerous autoencoder."})
    dang_ae = AE(X_train=list_dang_posts, name='autoencoderdang', model_dir=model_dir, epochs=100, batch_size=128, lr=0.05).train_autoencoder_content()
    self.update_state(state="PROGRESS", meta={"status": "Learning safe autoencoder."})
    safe_ae = AE(X_train=list_safe_posts, name='autoencodersafe', model_dir=model_dir, epochs=100, batch_size=128, lr=0.05).train_autoencoder_content()

    model_dir_rel = os.path.join(model_dir, "node_embeddings", "rel", rel_node_emb_technique)
    model_dir_spat = os.path.join(model_dir, "node_embeddings", "spat", spat_node_emb_technique)
    try:
        os.makedirs(model_dir_rel, exist_ok=False)
        os.makedirs(model_dir_spat, exist_ok=False)
    except OSError:
        pass
    rel_tree_path = os.path.join(model_dir_rel, "dtree.h5")
    spat_tree_path = os.path.join(model_dir_spat, "dtree.h5")

    ############### LEARN NODE EMBEDDINGS ###############
    self.update_state(state="PROGRESS", meta={"status": "Learning relational embeddings."})
    train_set_rel, train_set_labels_rel = dimensionality_reduction(rel_node_emb_technique, model_dir=model_dir_rel, edge_path=rel_edges_path,
                                                                   n_of_walks=n_of_walks_rel, walk_length=walk_length_rel, lab="rel", epochs=rel_ae_epochs,
                                                                   node_embedding_size=rel_node_embedding_size, p=p_rel, q=q_rel, id2idx_path=id2idx_rel_path,
                                                                   n2v_epochs=n2v_epochs_rel, train_df=train_df, adj_matrix_path=rel_adj_mat_path)

    train_set_spat, train_set_labels_spat = dimensionality_reduction(spat_node_emb_technique, model_dir=model_dir_spat, edge_path=spat_edges_path,
                                                                     n_of_walks=n_of_walks_spat, walk_length=walk_length_spat, epochs=spat_ae_epochs,
                                                                     node_embedding_size=spat_node_embedding_size, p=p_spat, q=q_spat, lab="spat",
                                                                     n2v_epochs=n2v_epochs_spat, train_df=train_df, adj_matrix_path=spat_adj_mat_path, id2idx_path=id2idx_spat_path)

    ############### LEARN DECISION TREES ###############
    self.update_state(state="PROGRESS", meta={"status": "Learning decision trees..."})
    train_decision_tree(train_set=train_set_rel, save_path=rel_tree_path, train_set_labels=train_set_labels_rel, name="relational")
    train_decision_tree(train_set=train_set_spat, save_path=spat_tree_path, train_set_labels=train_set_labels_spat, name="spatial")
    tree_rel = load_decision_tree(rel_tree_path)
    tree_spat = load_decision_tree(spat_tree_path)

    self.update_state(state="PROGRESS", meta={"status": "Learning mlp..."})
    if rel_node_emb_technique == "node2vec":
        mod = Word2Vec.load(os.path.join(model_dir_rel, "n2v_rel.h5"))
        d = mod.wv.key_to_index
        id2idx_rel = {int(k): d[k] for k in d.keys()}
    else:
        id2idx_rel = load_from_pickle(id2idx_rel_path)
    if spat_node_emb_technique == "node2vec":
        mod = Word2Vec.load(os.path.join(model_dir_spat, "n2v_spat.h5"))
        d = mod.wv.key_to_index
        id2idx_spat = {int(k): d[k] for k in d.keys()}
    else:
        id2idx_spat = load_from_pickle(id2idx_spat_path)
    mlp = learn_mlp(train_df=train_df, content_embs=list_embs, dang_ae=dang_ae, safe_ae=safe_ae, tree_rel=tree_rel, tree_spat=tree_spat, model_dir=model_dir,
                    rel_node_embs=train_set_rel, spat_node_embs=train_set_spat, id2idx_rel=id2idx_rel, id2idx_spat=id2idx_spat)
    save_to_pickle(os.path.join(model_dir, "mlp.pkl"), mlp)

def preprocess_task(content_url, id_field_name, text_field_name, dst_file_name):
    """
    Task that executes the preprocessing of a file. In order to be processable, the file must have these requisites:
    - Be in csv or csv-like format, such as xlsx
    - Have one user ID field and one text field. The names of the fields are provided by the user.
    The preprocessing consists in tokenization, punctuation removal, stopwords removal, stemming, and aggregation of the
    posts depending on the poster: in the final file we will have one row for each user, and the concatenation of the
    posts of that user as value
    Args:
        content_url: Path to the unprocessed file in the hadoop cluster
        id_field_name: Name of the field representing the user ID
        text_field_name: Name of the field representing the text
        dst_file_name: Name that the processed file will have
    """
    if content_url.__contains__("/"):
        p = "/".join(content_url.split("/")[:-1]) + "/"       # Retrieve the directory where the file is located. The processed file will be put there
    else:
        p = "./"
    client.download(hdfs_path=content_url, local_path="./df.csv", overwrite=True)
    df = pd.read_csv("./df.csv")
    df_proc = clean_dataframe(df, id_field_name, text_field_name)
    if len(set(df_proc[id_field_name].values)) != len(df_proc[id_field_name].values):
        df_proc = concatenate_posts(df_proc, aggregator_column=id_field_name, text_column=text_field_name)
    dst_file_name = dst_file_name + ".csv" if not dst_file_name.endswith(".csv") else dst_file_name
    df_proc.to_csv("./{}".format(dst_file_name))
    client.upload(hdfs_path=p+dst_file_name, local_path="./{}".format(dst_file_name))
