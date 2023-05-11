import requests

def train(address="http://172.19.0.4:5000"):
    """ This function sends a POST request that trains the model with some predefined hyperparameters. The ID of the training job will be returned """
    req = requests.post(
        "{}/node_classification/train".format(address),
        json={"content_url": 'https://drive.google.com/file/d/1o3kD4jg71IWv5RH9pxrYKMXw-J3CeDPP/view?usp=sharing',
              "spat_ne_tec": 'pca', "rel_ne_tec": 'none', "kr": 128, "ks": 128,
              "social_net_url": 'https://drive.google.com/file/d/1MhSo9tMDkfnlvZXPKgxv-HBLP4fSwvmN/view?usp=sharing',
              "spatial_net_url": 'https://drive.google.com/file/d/1fVipJMfIoqVqnImc9l79tLqzTWlhhXCq/view?usp=sharing',
              "id2idx_rel_url": 'https://drive.google.com/file/d/1nEdlSo2anF8Hs4rmage1FyRExuXrtQ-s/view?usp=sharing',
              "id2idx_spat_url": 'https://drive.google.com/file/d/16o4botYT2gOTQPVXkHnPKxSXJ1fH1875/view?usp=sharing',
              "rel_adj_matrix_url": 'https://drive.google.com/file/d/1fONnjUxrem-dsuPuFy4moSzS_JlBAbC6/view?usp=sharing',
              "spat_adj_matrix_url": 'https://drive.google.com/file/d/1CE9NVTMp6lVQaVUdGya2_A0VbzFCa617/view?usp=sharing',
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
