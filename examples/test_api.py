import requests

def train(address="http://172.19.0.4:5000"):
    """ This function sends a POST request that trains the model with some predefined hyperparameters. The ID of the training job will be returned """
    req = requests.post(
        "{}/node_classification/train".format(address),
        json={"content_url": 'user/dataset/content_labeled.csv',
              "spat_ne_tec": 'pca', "rel_ne_tec": 'none', "kr": 128, "ks": 128,
              "social_net_url": 'user/dataset/social_network.edg',
              "spatial_net_url": 'user/dataset/spatial_network.edg',
              "id2idx_rel_url": 'user/dataset/id2idx_rel.pkl',
              "id2idx_spat_url": 'user/dataset/id2idx_spat.pkl',
              "rel_adj_matrix_url": 'user/dataset/rel_adj_net.csv',
              "spat_adj_matrix_url": 'user/dataset/spat_adj_net.csv',
              "kc": 64, "window": 10, "w2v_epochs": 1})
    id = req.json()['Job id']
    print("ID: {}".format(id))
    return id

def task_status(job_id, address="http://172.19.0.4:5000"):
    """Get info about the status of the task with given ID"""
    req = requests.get("{}/node_classification/task_status/{}".format(address, job_id))
    print(req.json())
    return req.json()['state']

def predict(job_id, user_ids:list, address="http://172.19.0.4:5000"):
    response = requests.post(
        "{}/node_classification/predict".format(address),
        json={"job_id": job_id, "user_ids": user_ids})
    return response.json()


if __name__ == "__main__":
    id = train()
    # predictions = predict(id, ["121212121", "2052", "1077", "2814"])
