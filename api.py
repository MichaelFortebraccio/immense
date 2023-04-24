from flask import Flask, request, jsonify
from flask_restx import Api, Resource, reqparse, abort
from modelling.ae import AE
from node_classification.decision_tree import train_decision_tree, load_decision_tree
from modelling.sairus import classify_users, train_w2v_model, learn_mlp
import gdown
from os.path import exists
from os import makedirs
import pandas as pd
from celery import Celery

from node_classification.reduce_dimension import dimensionality_reduction

# sudo service redis-server stop

CONTENT_FILENAME = "content_labeled.csv"
REL_EDGES_FILENAME = "social_network.edg"
SPAT_EDGES_FILENAME = "spatial_network.edg"
ID2IDX_REL_FILENAME = "id2idx_rel.pkl"
ID2IDX_SPAT_FILENAME = "id2idx_spat.pkl"
REL_ADJ_MAT_FILENAME = "rel_adj_net.csv"
SPAT_ADJ_MAT_FILENAME = "spat_adj_net.csv"
MODEL_DIR = "{}/models"
DATASET_DIR = "{}/dataset"

api = Api(title="SNA spatial and textual API", version="0.1", description="Social Network Analysis API with spatial and textual information")
app = Flask(__name__)
app.config['CELERY_BROKER_URL'] = 'redis://localhost:6379/0'
app.config['CELERY_RESULT_BACKEND'] = 'redis://localhost:6379/0'
celery = Celery('api', broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)
api.init_app(app)

train_parser = reqparse.RequestParser()
train_parser.add_argument('content_url', type=str, required=True, help="Link to the file containing the labelled tweets")
train_parser.add_argument('social_net_url', type=str, required=True, help="Link to the .edg file containing the edges of the social network")
train_parser.add_argument('spatial_net_url', type=str, required=True, help="Link to the .edg file containing the weighted edges with the closeness relationships among users")
train_parser.add_argument('kc', type=int, required=True, default=512, help="Dimension of the word embeddings to learn")
train_parser.add_argument('window', type=int, default=5, required=True, help="Size of the window used to learn word embeddings")
train_parser.add_argument('w2v_epochs', type=int, default=10, required=True, help="No. epochs for training the word embedding model")
train_parser.add_argument('kr', type=int, default=128, required=True, help="Dimension of relational node embeddings to learn")
train_parser.add_argument('ks', type=int, default=128, required=True, help="Dimension of spatial node embeddings to learn")
train_parser.add_argument('n_of_walks_spat', type=int, default=10, required=True, help="Number of walks for the spatial n2v embedding method")
train_parser.add_argument('n_of_walks_rel', type=int, default=10, required=True, help="Number of walks for the relational n2v embedding method")
train_parser.add_argument('walk_length_spat', type=int, default=10, required=True, help="Walk length for the spatial n2v embedding method")
train_parser.add_argument('walk_length_rel', type=int, default=10, required=True, help="Walk length for the relational n2v embedding method")
train_parser.add_argument('p_spat', type=int, default=1, required=True, help="Node2vec's hyperparameter p for the spatial embedding method")
train_parser.add_argument('p_rel', type=int, default=1, required=True, help="Node2vec's hyperparameter p for the relational embedding method")
train_parser.add_argument('q_spat', type=int, default=4, required=True, help="Node2vec's hyperparameter q for the spatial embedding method")
train_parser.add_argument('q_rel', type=int, default=4, required=True, help="Node2vec's hyperparameter q for the relational embedding method")
train_parser.add_argument('n2v_epochs_spat', type=int, default=100, required=True, help="No. epochs for training the spatial n2v embedding method")
train_parser.add_argument('n2v_epochs_rel', type=int, default=100, required=True, help="No. epochs for training the relational n2v embedding method")

@celery.task(bind=True)
def train_task(self, tweets_url, word_embedding_size, window, w2v_epochs, rel_node_emb_technique:str, spat_node_emb_technique:str,
               rel_node_embedding_size, spat_node_embedding_size, social_network_url=None, spatial_network_url=None, p_spat=None, p_rel=None,
               q_spat=None, q_rel=None, n_of_walks_spat=None, n_of_walks_rel=None, walk_length_spat=None, walk_length_rel=None, n2v_epochs_spat=None,
               n2v_epochs_rel=None, spat_ae_epochs=None, rel_ae_epochs=None, adj_matrix_spat_url=None, adj_matrix_rel_url=None, id2idx_rel_url=None, id2idx_spat_url=None):
    job_id = self.request.id
    dataset_dir = DATASET_DIR.format(job_id)
    model_dir = MODEL_DIR.format(job_id)
    makedirs(dataset_dir, exist_ok=True)
    makedirs(model_dir, exist_ok=True)
    content_path = "{}/{}".format(dataset_dir, CONTENT_FILENAME)

    ############### DOWNLOAD FILES ###############
    self.update_state(state="PROGRESS", meta={"status": "Downloading dataset..."})
    if not exists(content_path):
        gdown.download(url=tweets_url, output=content_path, quiet=False, fuzzy=True)

    rel_edges_path = None
    spat_edges_path = None
    rel_adj_mat_path = None
    spat_adj_mat_path = None
    id2idx_rel_path = None
    id2idx_spat_path = None

    if rel_node_emb_technique == "node2vec":
        rel_edges_path = "{}/{}".format(dataset_dir, REL_EDGES_FILENAME)
        if not social_network_url:
            raise Exception("You need to provide a URL to the relational edge list")
        gdown.download(url=social_network_url, output=rel_edges_path, quiet=False, fuzzy=True)
    elif rel_node_emb_technique in ["pca", "autoencoder", "none"]:
        rel_adj_mat_path = "{}/{}".format(dataset_dir, REL_ADJ_MAT_FILENAME)
        id2idx_rel_path = "{}/{}".format(dataset_dir, ID2IDX_REL_FILENAME)
        if not adj_matrix_rel_url:
            raise Exception("You need to provide the URL to the relational adjacency matrix")
        if not id2idx_rel_url:
            raise Exception("You need to provide the URL to the relational id2idx file")
        gdown.download(url=adj_matrix_rel_url, output=rel_adj_mat_path, quiet=False, fuzzy=True)
        gdown.download(url=id2idx_rel_url, output=id2idx_rel_path, quiet=False, fuzzy=True)
    if spat_node_emb_technique == "node2vec":
        spat_edges_path = "{}/{}".format(dataset_dir, SPAT_EDGES_FILENAME)
        if not spatial_network_url:
            raise Exception("You need to provide a URL to the spatial network edge list")
        gdown.download(url=spatial_network_url, output=spat_edges_path, quiet=False, fuzzy=True)
    elif spat_node_emb_technique in ["pca", "autoencoder", "none"]:
        spat_adj_mat_path = "{}/{}".format(dataset_dir, SPAT_ADJ_MAT_FILENAME)
        id2idx_spat_path = "{}/{}".format(dataset_dir, ID2IDX_SPAT_FILENAME)
        if not adj_matrix_spat_url:
            raise Exception("You need to provide the URL to the spatial adjacency matrix")
        if not id2idx_spat_url:
            raise Exception("You need to provide the URL to the spatial id2idx file")
        gdown.download(url=adj_matrix_spat_url, output=spat_adj_mat_path, quiet=False, fuzzy=True)
        gdown.download(url=id2idx_spat_url, output=id2idx_spat_path, quiet=False, fuzzy=True)
    self.update_state(state="PROGRESS", meta={"status": "Dataset successfully downloaded."})

    train_df = pd.read_csv(content_path, sep=',').reset_index()

    self.update_state(state="PROGRESS", meta={"status": "Learning w2v model."})
    list_dang_posts, list_safe_posts, list_embs = train_w2v_model(train_df, embedding_size=word_embedding_size, window=window, epochs=w2v_epochs,
                                                                  model_dir=model_dir, dataset_dir=dataset_dir)
    self.update_state(state="PROGRESS", meta={"status": "Learning dangerous autoencoder."})
    dang_ae = AE(X_train=list_dang_posts, name='autoencoderdang', model_dir=model_dir, epochs=100, batch_size=128, lr=0.05).train_autoencoder_content()
    self.update_state(state="PROGRESS", meta={"status": "Learning safe autoencoder."})
    safe_ae = AE(X_train=list_safe_posts, name='autoencodersafe', model_dir=model_dir, epochs=100, batch_size=128, lr=0.05).train_autoencoder_content()

    model_dir_rel = "{}/node_embeddings/rel/{}".format(model_dir, rel_node_emb_technique)
    model_dir_spat = "{}/node_embeddings/spat/{}".format(model_dir, spat_node_emb_technique)
    try:
        makedirs(model_dir_rel, exist_ok=False)
        makedirs(model_dir_spat, exist_ok=False)
    except OSError:
        pass
    rel_tree_path = "{}/dtree.h5".format(model_dir_rel)
    spat_tree_path = "{}/dtree.h5".format(model_dir_spat)

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
    learn_mlp(train_df=train_df, content_embs=list_embs, dang_ae=dang_ae, safe_ae=safe_ae, tree_rel=tree_rel, tree_spat=tree_spat, model_dir=model_dir)

@api.route("/node_classification/train", methods=['POST'])
class Train(Resource):
    @api.expect(train_parser)
    def post(self):
        train_params = train_parser.parse_args(request)
        tweets_url = train_params['content_url']
        social_network_url = train_params['social_net_url']
        spatial_network_url = train_params['spatial_net_url']
        word_embedding_size = train_params['kc']
        window = train_params['window']
        w2v_epochs = train_params['w2v_epochs']
        spat_node_embedding_size = train_params['ks']
        rel_node_embedding_size = train_params['kr']
        n_of_walks_spat = train_params['n_of_walks_spat']
        n_of_walks_rel = train_params['n_of_walks_rel']
        walk_length_spat = train_params['walk_length_spat']
        walk_length_rel = train_params['walk_length_rel']
        p_spat = train_params['p_spat']
        p_rel = train_params['p_rel']
        q_spat = train_params['q_spat']
        q_rel = train_params['q_rel']
        n2v_epochs_spat = train_params['n2v_epochs_spat']
        n2v_epochs_rel = train_params['n2v_epochs_rel']

        task = train_task.apply_async(args=[tweets_url, social_network_url, spatial_network_url, word_embedding_size, window,
                                 w2v_epochs, p_spat, p_rel, spat_node_embedding_size, rel_node_embedding_size,
                                 n_of_walks_spat, n_of_walks_rel, walk_length_spat, walk_length_rel, q_spat, q_rel,
                                 n2v_epochs_spat, n2v_epochs_rel])
        #test(test_df=test_df, train_df=train_df, w2v_model=w2v_model, dang_ae=dang_ae, safe_ae=safe_ae, tree_rel=tree_rel,
        #     tree_spat=tree_spat, n2v_rel=n2v_rel, n2v_spat=n2v_spat, mlp=mlp)
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
        task = train_task.AsyncResult(job_id)
        if not exists(job_id) or task.state == "FAILURE":
            abort(400, "ERROR: the learning job id is not valid or not existent")
        elif task.state == 'PROGRESS' or task.state == 'STARTED':
            abort(204, "The training task has not completed yet. Try later. You can use the 'task_status' endpoint to check for the task state")
        elif task.state == 'SUCCESS':
            model_dir = MODEL_DIR.format(job_id)
            pred = classify_users(job_id, user_ids, CONTENT_FILENAME, model_dir)
            return jsonify(pred)

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
