from flask import Flask, request, jsonify
from flask_restx import Api, Resource, reqparse
from sairus import train, classify_users
import gdown
from os.path import exists
from os import makedirs
import pandas as pd
import os
import redis
from rq import Worker, Queue, Connection

api = Api(title="SNA spatial and textual API", version="0.1", description="Social Network Analysis API with spatial and textual information")
app = Flask(__name__)
api.init_app(app)

train_parser = reqparse.RequestParser()
train_parser.add_argument('tweets_url', type=str, required=True, help="Link to the file containing the labelled tweets")
train_parser.add_argument('social_net_url', type=str, required=True, help="Link to the .edg file containing the edges of the social network")
train_parser.add_argument('spatial_net_url', type=str, required=True, help="Link to the .edg file containing the weighted edges with the closeness relationships among users")
train_parser.add_argument('kc', type=int, required=True, default=512, help="Dimension of the word embeddings to learn")
train_parser.add_argument('window', type=int, default=5, required=True, help="Size of the window used to learn word embeddings")
train_parser.add_argument('w2v_epochs', type=int, default=10, required=True, help="No. epochs for training the word embedding model")
train_parser.add_argument('kr', type=int, default=128, required=True, help="Dimension of relational node embeddings to learn")
train_parser.add_argument('ks', type=int, default=128, required=True, help="Dimension of spatial node embeddings to learn")
train_parser.add_argument('n_of_walks', type=int, default=10, required=True, help="Number of n2v walks")
train_parser.add_argument('walk_length', type=int, default=10, required=True, help="Walk length")
train_parser.add_argument('p', type=int, default=1, required=True, help="Node2vec's hyperparameter p")
train_parser.add_argument('q', type=int, default=4, required=True, help="Node2vec's hyperparameter q")
train_parser.add_argument('n2v_epochs', type=int, default=100, required=True, help="No. epochs for training n2v models")

@api.route("/node_classification/train", methods=['POST'])
class Train(Resource):
    @api.expect(train_parser)
    def post(self):
        train_params = train_parser.parse_args(request)
        tweets_url = train_params['tweets_url']
        social_network_url = train_params['social_net_url']
        spatial_network_url = train_params['spatial_net_url']
        word_embedding_size = train_params['kc']
        window = train_params['window']
        w2v_epochs = train_params['w2v_epochs']
        rel_node_embedding_size = train_params['kr']
        space_node_embedding_size = train_params['ks']
        n_of_walks = train_params['n_of_walks']
        walk_length = train_params['walk_length']
        p = train_params['p']
        q = train_params['q']
        n2v_epochs = train_params['n2v_epochs']
        job_id = ""
        dataset_dir = "{}/dataset".format(job_id)
        model_dir = "{}/model".format(job_id)
        makedirs(dataset_dir, exist_ok=True)
        makedirs(model_dir, exist_ok=True)
        tweets_path = "{}/tweet_labeled.csv".format(dataset_dir)
        social_path = "{}/social_network.edg".format(dataset_dir)
        closeness_path = "{}/closeness_network.edg".format(dataset_dir)
        if not exists(tweets_path):
            gdown.download(url=tweets_url, output=tweets_path, quiet=False, fuzzy=True)
        if not exists(social_path):
            gdown.download(url=social_network_url, output=social_path, quiet=False, fuzzy=True)
        if not exists(closeness_path):
            gdown.download(url=spatial_network_url, output=closeness_path, quiet=False, fuzzy=True)

        df = pd.read_csv('{}/tweet_labeled_full.csv'.format(dataset_dir), sep=',')

        train(dataset_dir=dataset_dir, model_dir=model_dir, train_df=df, path_to_edges_rel=social_path,
              path_to_edges_clos=closeness_path, word_embedding_size=word_embedding_size, window=window,
              w2v_epochs=w2v_epochs, rel_node_embedding_size=rel_node_embedding_size,
              spatial_node_embedding_size=space_node_embedding_size, n_of_walks=n_of_walks,
              walk_length=walk_length, p=p, q=q, n2v_epochs=n2v_epochs)

        #test(test_df=test_df, train_df=train_df, w2v_model=w2v_model, dang_ae=dang_ae, safe_ae=safe_ae, tree_rel=tree_rel,
        #     tree_clos=tree_clos, n2v_rel=n2v_rel, n2v_clos=n2v_clos, mlp=mlp)
        return jsonify({"Job id": job_id})


predict_parser = reqparse.RequestParser()
predict_parser.add_argument("job_id", required=True, help="ID of the Job that created the model you want to use")
predict_parser.add_argument("user_id", type=int, help="ID of the user you want to classify")

@api.route("/node_classification/predict", methods=["GET"])
class Predict(Resource):
    @api.expect(predict_parser)
    def post(self):
        predict_params = predict_parser.parse_args(request)
        job_id = request.json['learning_job_id']
        user_id = request.json['user_ids']
        pred = classify_users(job_id, user_id)

    def get(self):
        predict_params = predict_parser.parse_args(request)
        job_id = request.json['job_id']
        user_id = request.json['user_id']
        print(user_id)


if __name__ == '__main__':
    '''listen = ['default']
    redis_url = os.getenv('REDISTOGO_URL', 'redis://localhost:5000')
    conn = redis.from_url(redis_url)
    with Connection(conn):
        worker = Worker(list(map(Queue, listen)))
        worker.work()'''
    app.run(host='0.0.0.0', port=5000)
