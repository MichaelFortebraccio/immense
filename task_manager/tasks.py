import json
import os
import pandas as pd
import hdfs
from dataset_scripts.dataset_class import Dataset, normalize_closeness
from gensim.models import Word2Vec
from modelling.sairus import train_w2v_model, learn_mlp
from modelling.ae import AE
from node_classification.random_forest import train_random_forest, load_random_forest
from node_classification.reduce_dimension import reduce_dimension
from task_manager.worker import celery
from utils import load_from_pickle, save_to_pickle


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

client = hdfs.InsecureClient(HDFS_HOST, timeout=60)

@celery.task(bind=True)
def preprocess_task(self, content_url, rel_url, id_field_name, text_field_name):
    """
    Task that executes the preprocessing of a file. In order to be processable, the file must have these requisites:
    - Be in csv or csv-like format, such as xlsx
    - Have one user ID field and one text field. The names of the fields are provided by the user.
    The preprocessing consists in tokenization, punctuation removal, stopwords removal, stemming, and aggregation of the
    posts depending on the poster: in the final file we will have one row for each user, and the concatenation of the
    posts of that user as value
    Args:
        content_url: Url to the file with the unprocessed content in the hdfs
        rel_url: Url to the file depicting the relationships
        id_field_name: Name of the field representing the user ID
        text_field_name: Name of the field representing the text
    """
    job_id = self.request.id
    dataset_dir = DATASET_DIR.format(job_id)
    os.makedirs(dataset_dir, exist_ok=True)
    content_path = os.path.join(dataset_dir, CONTENT_FILENAME)
    rel_path = os.path.join(dataset_dir, "rel.json")

    self.update_state(state="PROGRESS", meta={"status": "Downloading..."})
    if not client.content(content_url):
        raise FileNotFoundError("The URL to the content file does not exist")
    if not client.content(rel_url):
        raise FileNotFoundError("The URL to the file with the social relationships does not exist")
    client.download(hdfs_path=content_url, local_path=content_path)
    client.download(hdfs_path=rel_url, local_path=rel_path)

    with open(content_path, 'r') as f:
        d_content = json.load(f)
    with open(rel_path, 'r') as f:
        d_rel = json.load(f)
    ds = Dataset(posts_dict=d_content, rel_dict=d_rel)
    self.update_state(state="PROGRESS", meta={"status": "Preprocessing text..."})
    df_proc = ds.preprocess_content(id_field_name=id_field_name, text_field_name=text_field_name)
    df_proc.to_csv(os.path.join(dataset_dir, CONTENT_FILENAME))

    self.update_state(state="PROGRESS", meta={"status": "Creating the spatial network..."})
    ds.users_with_pos()
    for us in ds.users_with_position:
        us.position_mode()      # Set, for each user, the mode of locations as its location
    dist = ds.calculate_all_closenesses()
    normalize_closeness(dist, SPAT_EDGES_FILENAME)

    self.update_state(state="PROGRESS", meta={"status": "Creating the social network..."})
    ds.build_rel_network(os.path.join(dataset_dir, REL_EDGES_FILENAME))

    #self.update_state(state="PROGRESS", meta={"status": "Uploading the files on hdfs..."})
    #client.upload(hdfs_path=os.path.join(job_id, CONTENT_FILENAME), local_path="./{}".format(CONTENT_FILENAME))
    #client.upload(hdfs_path=dst_dir + SPAT_EDGES_FILENAME, local_path="./{}".format(SPAT_EDGES_FILENAME))
    #client.upload(hdfs_path=os.path.join(job_id, REL_EDGES_FILENAME), local_path="./{}".format(REL_EDGES_FILENAME))

@celery.task(bind=True)
def train_task(self, job_id, word_embedding_size, window, w2v_epochs, rel_node_emb_technique: str, spat_node_emb_technique: str,
               rel_node_emb_size, spat_node_emb_size, n_of_walks_rel=None, n_of_walks_spat=None, walk_length_rel=None,
               walk_length_spat=None, p_rel=None, p_spat=None, q_rel=None, q_spat=None, n2v_epochs_rel=None, n2v_epochs_spat=None,
               spat_ae_epochs=None, rel_ae_epochs=None, adj_matrix_rel_url=None, adj_matrix_spat_url=None, id2idx_rel_url=None, id2idx_spat_url=None):
    dataset_dir = DATASET_DIR.format(job_id)    # Dataset is in a directory linked to the preprocessing job
    model_dir = MODEL_DIR.format(self.request.id)   # Every training run will have its own job, and the models are saved in a directory linked to that job

    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    content_path = os.path.join(dataset_dir, CONTENT_FILENAME)
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError("The provided job ID does not correspond to any preprocessing task.")

    with open(os.path.join(JOBS_DIR, self.request.id, "techniques.txt"), "w") as f:
        f.write(rel_node_emb_technique+"\n")
        f.write(spat_node_emb_technique+"\n")
        f.write(str(word_embedding_size))

    rel_edges_path = os.path.join(DATASET_DIR.format(job_id), "social_network.edg")
    spat_edges_path = os.path.join(DATASET_DIR.format(job_id), "spatial_network.edg")
    rel_adj_mat_path = None
    spat_adj_mat_path = None
    id2idx_rel_path = None
    id2idx_spat_path = None

    train_df = pd.read_csv(content_path, sep=',').reset_index()

    self.update_state(state="PROGRESS", meta={"status": "Learning w2v model."})
    list_dang_posts, list_safe_posts, list_embs = train_w2v_model(train_df, embedding_size=word_embedding_size, window=window,
                                                                  epochs=w2v_epochs, model_dir=model_dir, dataset_dir=dataset_dir, name="w2v.pkl")
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
    train_set_rel, train_set_labels_rel = reduce_dimension(rel_node_emb_technique, model_dir=model_dir_rel, edge_path=rel_edges_path,
                                                           n_of_walks=n_of_walks_rel, walk_length=walk_length_rel, lab="rel", epochs=rel_ae_epochs,
                                                           ne_dim=rel_node_emb_size, p=p_rel, q=q_rel, id2idx_path=id2idx_rel_path,
                                                           n2v_epochs=n2v_epochs_rel, train_df=train_df, adj_matrix_path=rel_adj_mat_path)

    train_set_spat, train_set_labels_spat = reduce_dimension(spat_node_emb_technique, model_dir=model_dir_spat, edge_path=spat_edges_path,
                                                             n_of_walks=n_of_walks_spat, walk_length=walk_length_spat, epochs=spat_ae_epochs,
                                                             ne_dim=spat_node_emb_size, p=p_spat, q=q_spat, lab="spat",
                                                             n2v_epochs=n2v_epochs_spat, train_df=train_df, adj_matrix_path=spat_adj_mat_path, id2idx_path=id2idx_spat_path)

    ############### LEARN DECISION TREES ###############
    self.update_state(state="PROGRESS", meta={"status": "Learning decision trees..."})
    train_random_forest(train_set=train_set_rel, dst_dir=rel_tree_path, train_set_labels=train_set_labels_rel, name="relational")
    train_random_forest(train_set=train_set_spat, dst_dir=spat_tree_path, train_set_labels=train_set_labels_spat, name="spatial")
    tree_rel = load_random_forest(rel_tree_path)
    tree_spat = load_random_forest(spat_tree_path)

    self.update_state(state="PROGRESS", meta={"status": "Learning mlp..."})
    if rel_node_emb_technique == "node2vec":
        n2v_rel = Word2Vec.load(os.path.join(model_dir_rel, "n2v.h5"))
        id2idx_rel = None
    else:
        n2v_rel = None
        id2idx_rel = load_from_pickle(id2idx_rel_path)
    if spat_node_emb_technique == "node2vec":
        n2v_spat = Word2Vec.load(os.path.join(model_dir_spat, "n2v.h5"))
        id2idx_spat = None
    else:
        n2v_spat = None
        id2idx_spat = load_from_pickle(id2idx_spat_path)
    mlp = learn_mlp(train_df=train_df, content_embs=list_embs, ae_dang=dang_ae, ae_safe=safe_ae, tree_rel=tree_rel, tree_spat=tree_spat,
                    node_embs_rel=train_set_rel, node_embs_spat=train_set_spat, model_dir=model_dir, id2idx_rel=id2idx_rel, id2idx_spat=id2idx_spat,
                    n2v_rel=n2v_rel, n2v_spat=n2v_spat)
    save_to_pickle(os.path.join(model_dir, "mlp.pkl"), mlp)

# TODO: Bisogna vedere come gestire la questione del labeling.
