from flask import Flask, request, jsonify
from flask_restx import Api, Resource, reqparse, abort
from modelling.ae import AE
from node_classification.decision_tree import train_decision_tree, load_decision_tree
from modelling.sairus import classify_users, train_w2v_model, learn_mlp
from utils import load_from_pickle, save_to_pickle
from gensim.models import Word2Vec
import gdown
from os.path import exists, join
from os import makedirs
import pandas as pd
from celery import Celery

from node_classification.reduce_dimension import dimensionality_reduction

# celery -A api.celery worker --loglevel=info

CONTENT_FILENAME = "content_labeled.csv"
REL_EDGES_FILENAME = "social_network.edg"
SPAT_EDGES_FILENAME = "spatial_network.edg"
ID2IDX_REL_FILENAME = "id2idx_rel.pkl"
ID2IDX_SPAT_FILENAME = "id2idx_spat.pkl"
REL_ADJ_MAT_FILENAME = "rel_adj_net.csv"
SPAT_ADJ_MAT_FILENAME = "spat_adj_net.csv"
MODEL_DIR = "{}/models"
DATASET_DIR = "{}/dataset"  # .format('8931a2b4-7c25-4c52-a092-269f368d160e')
WORD_EMB_SIZE = 0

api = Api(title="SNA spatial and textual API", version="0.1", description="Social Network Analysis API with spatial and textual information")
app = Flask(__name__)
app.config['CELERY_BROKER_URL'] = 'redis://localhost:6379/0'
app.config['CELERY_RESULT_BACKEND'] = 'redis://localhost:6379/0'
celery = Celery('api', broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)
api.init_app(app)

train_parser = reqparse.RequestParser()
train_parser.add_argument('rel_ne_tec', type=str, required=True, default="node2vec", choices=("none", "autoencoder", "node2vec", "pca"),
                          help="Technique for learning relational node embeddings. Must be one of the following:"
                               "'ae' (autoencoder); 'none' (directly uses the adj matrix rows); 'node2vec'; 'pca'")
train_parser.add_argument('spat_ne_tec', type=str, required=True, default="node2vec", choices=("none", "autoencoder", "node2vec", "pca"),
                          help="Technique for learning spatial node embeddings. Must be one of the following:"
                               "'ae' (autoencoder); 'none' (directly uses the adj matrix rows); 'node2vec'; 'pca'")
train_parser.add_argument('kc', type=int, required=True, default=512, help="Dimension of the word embeddings")
train_parser.add_argument('content_url', type=str, required=True, help="Link to the file containing the labelled tweets")
train_parser.add_argument('window', type=int, default=5, required=True, help="Size of the window used to learn word embeddings")
train_parser.add_argument('w2v_epochs', type=int, default=10, required=True, help="No. epochs for training the word embedding model")
train_parser.add_argument('kr', type=int, default=128, required=True, help="Dimension of relational node embeddings to learn")
train_parser.add_argument('ks', type=int, default=128, required=True, help="Dimension of spatial node embeddings to learn")
train_parser.add_argument('social_net_url', type=str, required=False, help="Link to the .edg file containing the edges of the social network. Required if rel_ne_tec=='node2vec'")
train_parser.add_argument('spatial_net_url', type=str, required=False, help="Link to the .edg file containing the weighted edges with the closeness relationships among users")
train_parser.add_argument('n_of_walks_rel', type=int, default=10, required=False, help="Number of walks for the relational n2v embedding method. Required if rel_ne_tec=='node2vec'")
train_parser.add_argument('n_of_walks_spat', type=int, default=10, required=False, help="Number of walks for the spatial n2v embedding method. Required if spat_ne_tec=='node2vec'")
train_parser.add_argument('walk_length_rel', type=int, default=10, required=False, help="Walk length for the relational n2v embedding method. Required if rel_ne_tec=='node2vec'")
train_parser.add_argument('walk_length_spat', type=int, default=10, required=False, help="Walk length for the spatial n2v embedding method. Required if spat_ne_tec=='node2vec'")
train_parser.add_argument('p_rel', type=int, default=1, required=False, help="Node2vec's hyperparameter p for the relational embedding method. Required if rel_ne_tec=='node2vec'")
train_parser.add_argument('p_spat', type=int, default=1, required=False, help="Node2vec's hyperparameter p for the spatial embedding method. Required if spat_ne_tec=='node2vec'")
train_parser.add_argument('q_rel', type=int, default=4, required=False, help="Node2vec's hyperparameter q for the relational embedding method. Required if rel_ne_tec=='node2vec'")
train_parser.add_argument('q_spat', type=int, default=4, required=False, help="Node2vec's hyperparameter q for the spatial embedding method. Required if spat_ne_tec=='node2vec'")
train_parser.add_argument('n2v_epochs_rel', type=int, default=100, required=False, help="No. epochs for training the relational n2v embedding method. Required if rel_ne_tec=='node2vec'")
train_parser.add_argument('n2v_epochs_spat', type=int, default=100, required=False, help="No. epochs for training the spatial n2v embedding method. Required if spat_ne_tec=='node2vec'")
train_parser.add_argument('rel_autoencoder_epochs', type=int, default=100, required=False, help="No. epochs for training the autoencoder that embeds the social relationships. Required if rel_ne_tec=='ae'")
train_parser.add_argument('spat_autoencoder_epochs', type=int, default=100, required=False, help="No. epochs for training the autoencoder that embeds the social relationships. Required if spat_ne_tec=='ae'")
train_parser.add_argument('rel_adj_matrix_url', type=str, required=False, help="Link to the .csv file containing the relational adjacency matrix. Required if rel_ne_tec in ['none', 'ae', 'pca']")
train_parser.add_argument('spat_adj_matrix_url', type=str, required=False, help="Link to the .csv file containing the spatial adjacency matrix. Required if spat_ne_tec in ['none', 'ae', 'pca']")
train_parser.add_argument('id2idx_rel_url', type=str, required=False, help="Link to the .pkl file with the mapping between user IDs and the index of its row in the relational adjacency matrix. Required if rel_ne_tec in ['none', 'ae', 'pca']")
train_parser.add_argument('id2idx_spat_url', type=str, required=False, help="Link to the .pkl file with the mapping between user IDs and the index of its row in the spatial adjacency matrix. Required if spat_ne_tec in ['none', 'ae', 'pca']")

@celery.task(bind=True)
def train_task(self, content_url, word_embedding_size, window, w2v_epochs, rel_node_emb_technique: str, spat_node_emb_technique: str,
               rel_node_embedding_size, spat_node_embedding_size, social_network_url=None, spatial_network_url=None, n_of_walks_rel=None, n_of_walks_spat=None,
               walk_length_rel=None, walk_length_spat=None, p_rel=None, p_spat=None, q_rel=None, q_spat=None, n2v_epochs_rel=None, n2v_epochs_spat=None,
               spat_ae_epochs=None, rel_ae_epochs=None, adj_matrix_rel_url=None, adj_matrix_spat_url=None, id2idx_rel_url=None, id2idx_spat_url=None):
    job_id = self.request.id
    dataset_dir = DATASET_DIR.format(job_id)
    model_dir = MODEL_DIR.format(job_id)
    makedirs(dataset_dir, exist_ok=True)
    makedirs(model_dir, exist_ok=True)
    content_path = join(dataset_dir, CONTENT_FILENAME)

    ############### DOWNLOAD FILES ###############
    if not exists(content_path):
        self.update_state(state="PROGRESS", meta={"status": "Downloading content..."})
        gdown.download(url=content_url, output=content_path, quiet=False, fuzzy=True)

    rel_edges_path = None
    spat_edges_path = None
    rel_adj_mat_path = None
    spat_adj_mat_path = None
    id2idx_rel_path = None
    id2idx_spat_path = None

    if rel_node_emb_technique == "node2vec":
        if not social_network_url:
            raise Exception("You need to provide a URL to the relational edge list")
        rel_edges_path = join(dataset_dir, REL_EDGES_FILENAME)
        if not exists(rel_edges_path):
            self.update_state(state="PROGRESS", meta={"status": "Downloading social edge list..."})
            gdown.download(url=social_network_url, output=rel_edges_path, quiet=False, fuzzy=True)
    elif rel_node_emb_technique in ["pca", "autoencoder", "none"]:
        if not adj_matrix_rel_url:
            raise Exception("You need to provide the URL to the relational adjacency matrix")
        if not id2idx_rel_url:
            raise Exception("You need to provide the URL to the relational id2idx file")
        rel_adj_mat_path = join(dataset_dir, REL_ADJ_MAT_FILENAME)
        id2idx_rel_path = join(dataset_dir, ID2IDX_REL_FILENAME)
        if not exists(rel_adj_mat_path):
            self.update_state(state="PROGRESS", meta={"status": "Downloading social adjacency matrix..."})
            gdown.download(url=adj_matrix_rel_url, output=rel_adj_mat_path, quiet=False, fuzzy=True)
        if not exists(id2idx_rel_path):
            self.update_state(state="PROGRESS", meta={"status": "Downloading social id2idx_rel..."})
            gdown.download(url=id2idx_rel_url, output=id2idx_rel_path, quiet=False, fuzzy=True)
    if spat_node_emb_technique == "node2vec":
        if not spatial_network_url:
            raise Exception("You need to provide a URL to the spatial network edge list")
        spat_edges_path = join(dataset_dir, SPAT_EDGES_FILENAME)
        if not exists(spat_edges_path):
            self.update_state(state="PROGRESS", meta={"status": "Downloading spatial edge list..."})
            gdown.download(url=spatial_network_url, output=spat_edges_path, quiet=False, fuzzy=True)
    elif spat_node_emb_technique in ["pca", "autoencoder", "none"]:
        spat_adj_mat_path = join(dataset_dir, SPAT_ADJ_MAT_FILENAME)
        id2idx_spat_path = join(dataset_dir, ID2IDX_SPAT_FILENAME)
        if not adj_matrix_spat_url:
            raise Exception("You need to provide the URL to the spatial adjacency matrix")
        if not id2idx_spat_url:
            raise Exception("You need to provide the URL to the spatial id2idx file")
        if not exists(spat_adj_mat_path):
            self.update_state(state="PROGRESS", meta={"status": "Downloading spatial adjacency matrix..."})
            gdown.download(url=adj_matrix_spat_url, output=spat_adj_mat_path, quiet=False, fuzzy=True)
        if not exists(id2idx_spat_path):
            self.update_state(state="PROGRESS", meta={"status": "Downloading spatial id2idx_rel..."})
            gdown.download(url=id2idx_spat_url, output=id2idx_spat_path, quiet=False, fuzzy=True)
    self.update_state(state="PROGRESS", meta={"status": "Dataset successfully downloaded."})

    train_df = pd.read_csv(content_path, sep=',').reset_index()

    self.update_state(state="PROGRESS", meta={"status": "Learning w2v model."})
    list_dang_posts, list_safe_posts, list_embs = train_w2v_model(train_df, embedding_size=word_embedding_size, window=window,
                                                                  epochs=w2v_epochs, model_dir=model_dir, dataset_dir=dataset_dir)
    self.update_state(state="PROGRESS", meta={"status": "Learning dangerous autoencoder."})
    dang_ae = AE(X_train=list_dang_posts, name='autoencoderdang', model_dir=model_dir, epochs=100, batch_size=128, lr=0.05).train_autoencoder_content()
    self.update_state(state="PROGRESS", meta={"status": "Learning safe autoencoder."})
    safe_ae = AE(X_train=list_safe_posts, name='autoencodersafe', model_dir=model_dir, epochs=100, batch_size=128, lr=0.05).train_autoencoder_content()

    model_dir_rel = join(model_dir, "node_embeddings", "rel", rel_node_emb_technique)
    model_dir_spat = join(model_dir, "node_embeddings", "spat", spat_node_emb_technique)
    try:
        makedirs(model_dir_rel, exist_ok=False)
        makedirs(model_dir_spat, exist_ok=False)
    except OSError:
        pass
    rel_tree_path = join(model_dir_rel, "dtree.h5")
    spat_tree_path = join(model_dir_spat, "dtree.h5")

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
        mod = Word2Vec.load(join(model_dir_rel, "n2v_rel.h5"))
        d = mod.wv.key_to_index
        id2idx_rel = {int(k): d[k] for k in d.keys()}
    else:
        id2idx_rel = load_from_pickle(id2idx_rel_path)
    if spat_node_emb_technique == "node2vec":
        mod = Word2Vec.load(join(model_dir_spat, "n2v_spat.h5"))
        d = mod.wv.key_to_index
        id2idx_spat = {int(k): d[k] for k in d.keys()}
    else:
        id2idx_spat = load_from_pickle(id2idx_spat_path)
    mlp = learn_mlp(train_df=train_df, content_embs=list_embs, dang_ae=dang_ae, safe_ae=safe_ae, tree_rel=tree_rel, tree_spat=tree_spat, model_dir=model_dir,
                    rel_node_embs=train_set_rel, spat_node_embs=train_set_spat, id2idx_rel=id2idx_rel, id2idx_spat=id2idx_spat)
    save_to_pickle(join(model_dir, "mlp.pkl"), mlp)


@api.route("/node_classification/train", methods=['POST'])
class Train(Resource):
    @api.expect(train_parser)
    def post(self):
        train_params = train_parser.parse_args(request)
        content_url = train_params['content_url']
        WORD_EMB_SIZE = train_params['kc']
        window = train_params['window']
        w2v_epochs = train_params['w2v_epochs']
        rel_ne_technique = train_params['rel_ne_tec']
        spat_ne_technique = train_params['spat_ne_tec']
        rel_node_embedding_size = train_params['kr']
        spat_node_embedding_size = train_params['ks']
        n_of_walks_rel = train_params['n_of_walks_rel']
        n_of_walks_spat = train_params['n_of_walks_spat']
        walk_length_rel = train_params['walk_length_rel']
        walk_length_spat = train_params['walk_length_spat']
        p_rel = train_params['p_rel']
        p_spat = train_params['p_spat']
        q_rel = train_params['q_rel']
        q_spat = train_params['q_spat']
        social_network_url = train_params['social_net_url']
        spatial_network_url = train_params['spatial_net_url']
        n2v_epochs_rel = train_params['n2v_epochs_rel']
        n2v_epochs_spat = train_params['n2v_epochs_spat']
        rel_ae_epochs = train_params['rel_autoencoder_epochs']
        spat_ae_epochs = train_params['spat_autoencoder_epochs']
        rel_adj_matrix_url = train_params['rel_adj_matrix_url']
        spat_adj_matrix_url = train_params['spat_adj_matrix_url']
        id2idx_rel_url = train_params['id2idx_rel_url']
        id2idx_spat_url = train_params['id2idx_spat_url']

        task = train_task.apply_async(args=[content_url, WORD_EMB_SIZE, window, w2v_epochs, rel_ne_technique, spat_ne_technique, rel_node_embedding_size,
                                            spat_node_embedding_size, social_network_url, spatial_network_url, n_of_walks_rel, n_of_walks_spat, walk_length_rel,
                                            walk_length_spat, p_rel, p_spat, q_rel, q_spat, n2v_epochs_rel, n2v_epochs_spat, spat_ae_epochs, rel_ae_epochs,
                                            rel_adj_matrix_url, spat_adj_matrix_url, id2idx_rel_url, id2idx_spat_url])
        makedirs(task.id, exist_ok=True)
        with open(join(task.id, "techniques.txt"), "w") as f:
            f.write(rel_ne_technique+"\n")
            f.write(spat_ne_technique)
        return jsonify({"Job id": task.id})


predict_parser = reqparse.RequestParser()
predict_parser.add_argument("job_id", required=True, help="ID of the Job that created the model you want to use")
predict_parser.add_argument("user_ids", type=int, action='append', help="IDs of the users you want to classify")

@api.route("/node_classification/predict", methods=["POST"])
class Predict(Resource):
    @api.expect(predict_parser)
    def post(self):
        predict_params = predict_parser.parse_args(request)
        job_id = predict_params['job_id']
        user_ids = predict_params['user_ids']
        with open(join(job_id, "techniques.txt"), 'r') as f:
            tec = [l.strip() for l in f.readlines()]
        rel_technique = tec[0]
        spat_technique = tec[1]
        task = train_task.AsyncResult(job_id)
        if not exists(job_id) or task.state == "FAILURE":
            abort(400, "ERROR: the learning job id is not valid or not existent")
        elif task.state == 'PROGRESS' or task.state == 'STARTED':
            abort(204, "The training task has not completed yet. Try later. You can use the 'task_status' endpoint to check for the task state")
        elif task.state == 'SUCCESS' or (task.state == 'PENDING' and exists(job_id)):
            pred = classify_users(
                job_id, user_ids, CONTENT_FILENAME, ID2IDX_REL_FILENAME, ID2IDX_SPAT_FILENAME, REL_ADJ_MAT_FILENAME,
                SPAT_ADJ_MAT_FILENAME, rel_technique=rel_technique, spat_technique=spat_technique)
            readable_preds = {}
            for k in pred.keys():
                if pred[k] == 0:
                    readable_preds[k] = "safe"
                elif pred[k] == 1:
                    readable_preds[k] = "risky"
                else:
                    readable_preds[k] = pred[k]
            return jsonify(readable_preds)

@api.route("/node_classification/task_status/<task_id>")
class TaskStatus(Resource):
    def get(self, task_id):
        task = train_task.AsyncResult(task_id)
        response = {
            'state': task.state,
        }
        if task.info:
            response['info'] = task.info
        return jsonify(response)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
