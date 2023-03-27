from flask import Flask, request, jsonify
from flask_restx import Api, Resource, reqparse, abort
from sairus import train, classify_users
import gdown
from os.path import exists
from os import makedirs
import pandas as pd
from celery import Celery

# sudo service redis-server stop

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
def download_dataset():

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
        job_id = ""

        train(tweets_url=tweets_url, social_network_url=social_network_url, spatial_network_url=spatial_network_url,
              word_embedding_size=word_embedding_size, window=window, w2v_epochs=w2v_epochs, p_spat=p_spat, p_rel=p_rel,
              spat_node_embedding_size=spat_node_embedding_size, rel_node_embedding_size=rel_node_embedding_size,
              n_of_walks_spat=n_of_walks_spat, n_of_walks_rel=n_of_walks_rel, walk_length_spat=walk_length_spat,
              walk_length_rel=walk_length_rel, q_spat=q_spat, q_rel=q_rel, n2v_epochs_spat=n2v_epochs_spat,
              n2v_epochs_rel=n2v_epochs_rel)

        #test(test_df=test_df, train_df=train_df, w2v_model=w2v_model, dang_ae=dang_ae, safe_ae=safe_ae, tree_rel=tree_rel,
        #     tree_clos=tree_clos, n2v_rel=n2v_rel, n2v_clos=n2v_clos, mlp=mlp)
        return jsonify({"Job id": job_id})


predict_parser = reqparse.RequestParser()
predict_parser.add_argument("job_id", required=True, help="ID of the Job that created the model you want to use")
predict_parser.add_argument("user_ids", type=int, action='append', help="IDs of the users you want to classify")

@api.route("/node_classification/predict", methods=["POST"])
class Predict(Resource):
    def post(self):
        predict_params = predict_parser.parse_args(request)
        job_id = predict_params['job_id']
        user_ids = predict_params['user_ids']
        pred = classify_users(job_id, user_ids)
        if pred == 400:
            abort(400, "ERROR: the learning job id is not valid or not existent")
        else:
            return jsonify(pred)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
