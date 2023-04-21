from flask import Flask, request, jsonify
from flask_restx import Api, Resource, reqparse, abort

from modelling.ae import AE
from node_classification.decision_tree import train_decision_tree, load_decision_tree
from node_classification.graph_embeddings.node2vec import Node2VecEmbedder
from modelling.sairus import classify_users, train_w2v_model, learn_mlp
import gdown
from os.path import exists
from os import makedirs
import pandas as pd
from celery import Celery

# sudo service redis-server stop

CONTENT_FILENAME = "content_labeled.csv"
SOCIAL_NET_FILENAME = "social_network.edg"
SPATIAL_NET_FILENAME = "spatial_network.edg"
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
def train_task(self, tweets_url, social_network_url, spatial_network_url, word_embedding_size, window, w2v_epochs,
               p_spat, p_rel, spat_node_embedding_size, rel_node_embedding_size, n_of_walks_spat, n_of_walks_rel,
               walk_length_spat, walk_length_rel, q_spat, q_rel, n2v_epochs_spat, n2v_epochs_rel):
    job_id = self.request.id
    dataset_dir = DATASET_DIR.format(job_id)
    model_dir = MODEL_DIR.format(job_id)
    makedirs(dataset_dir, exist_ok=True)
    makedirs(model_dir, exist_ok=True)
    content_path = "{}/{}".format(dataset_dir, CONTENT_FILENAME)
    social_path = "{}/{}".format(dataset_dir, SOCIAL_NET_FILENAME)
    spatial_path = "{}/{}".format(dataset_dir, SPATIAL_NET_FILENAME)
    self.update_state(state="PROGRESS", meta={"status": "Downloading dataset."})
    if not exists(content_path):
        gdown.download(url=tweets_url, output=content_path, quiet=False, fuzzy=True)
    if not exists(social_path):
        gdown.download(url=social_network_url, output=social_path, quiet=False, fuzzy=True)
    if not exists(spatial_path):
        gdown.download(url=spatial_network_url, output=spatial_path, quiet=False, fuzzy=True)
    self.update_state(state="PROGRESS", meta={"status": "Dataset successfully downloaded."})

    train_df = pd.read_csv(content_path, sep=',').reset_index()

    self.update_state(state="PROGRESS", meta={"status": "Learning w2v model."})
    list_dang_posts, list_safe_posts, list_embs = train_w2v_model(train_df, embedding_size=word_embedding_size, window=window, epochs=w2v_epochs,
                                                                  model_dir=model_dir, dataset_dir=dataset_dir)

    self.update_state(state="PROGRESS", meta={"status": "Learning dangerous autoencoder."})
    dang_ae = AE(input_len=word_embedding_size, X_train=list_dang_posts, label='dang', model_dir=model_dir).train_autoencoder_content()
    self.update_state(state="PROGRESS", meta={"status": "Learning safe autoencoder."})
    safe_ae = AE(input_len=word_embedding_size, X_train=list_safe_posts, label='safe', model_dir=model_dir).train_autoencoder_content()

    rel_tree_path = "{}/dtree_rel.h5".format(model_dir)
    spat_tree_path = "{}/dtree_spat.h5".format(model_dir)
    rel_n2v_path = "{}/n2v_rel.h5".format(model_dir)
    spat_n2v_path = "{}/n2v_spat.h5".format(model_dir)

    self.update_state(state="PROGRESS", meta={"status": "Learning relational n2v model."})
    n2v_rel = Node2VecEmbedder(path_to_edges=social_path, weighted=False, directed=True, n_of_walks=n_of_walks_rel,
                                   walk_length=walk_length_rel, embedding_size=rel_node_embedding_size, p=p_rel, q=q_rel,
                                   epochs=n2v_epochs_rel, model_path=rel_n2v_path).learn_n2v_embeddings()
    if not exists(rel_tree_path):  # IF THE DECISION TREE HAS NOT BEEN LEARNED, LOAD/TRAIN THE N2V MODEL
        train_set_ids_rel = [i for i in train_df['id'] if str(i) in n2v_rel.wv.key_to_index]
        self.update_state(state="PROGRESS", meta={"status": "Learning relational decision tree."})
        train_decision_tree(train_set_ids=train_set_ids_rel, save_path=rel_tree_path, n2v_model=n2v_rel,
                            train_set_labels=train_df[train_df['id'].isin(train_set_ids_rel)]['label'], name="relational")

    self.update_state(state="PROGRESS", meta={"status": "Learning spatial n2v model."})
    n2v_spat = Node2VecEmbedder(path_to_edges=spatial_path, weighted=True, directed=False, n_of_walks=n_of_walks_spat,
                                walk_length=walk_length_spat, embedding_size=spat_node_embedding_size, p=p_spat, q=q_spat,
                                epochs=n2v_epochs_spat, model_path=spat_n2v_path).learn_n2v_embeddings()
    if not exists(spat_tree_path):
        train_set_ids_spat = [i for i in train_df['id'] if str(i) in n2v_spat.wv.key_to_index]
        self.update_state(state="PROGRESS", meta={"status": "Learning spatial decision tree."})
        train_decision_tree(train_set_ids=train_set_ids_spat, save_path=spat_tree_path, n2v_model=n2v_spat,
                            train_set_labels=train_df[train_df['id'].isin(train_set_ids_spat)]['label'], name="spatial")
    tree_rel = load_decision_tree(rel_tree_path)
    tree_spat = load_decision_tree(spat_tree_path)

    self.update_state(state="PROGRESS", meta={"status": "Learning mlp."})
    learn_mlp(train_df=train_df, content_embs=list_embs, dang_ae=dang_ae, safe_ae=safe_ae, n2v_rel=n2v_rel,
              n2v_spat=n2v_spat, tree_rel=tree_rel, tree_spat=tree_spat, model_dir=model_dir)

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
