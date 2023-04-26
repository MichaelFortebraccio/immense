import requests

response = requests.post("http://127.0.0.1:5000/node_classification/train",
                         json={"content_url": 'https://drive.google.com/file/d/1o3kD4jg71IWv5RH9pxrYKMXw-J3CeDPP/view?usp=sharing',
                               "spat_ne_tec": 'node2vec', "rel_ne_tec": 'pca', "kr": 128, "ks": 128,
                               "social_net_url": 'https://drive.google.com/file/d/1MhSo9tMDkfnlvZXPKgxv-HBLP4fSwvmN/view?usp=sharing',
                               "spatial_net_url": 'https://drive.google.com/file/d/1fVipJMfIoqVqnImc9l79tLqzTWlhhXCq/view?usp=sharing',
                               "id2idx_rel_url": 'https://drive.google.com/file/d/1nEdlSo2anF8Hs4rmage1FyRExuXrtQ-s/view?usp=sharing',
                               "id2idx_spat_url": 'https://drive.google.com/file/d/16o4botYT2gOTQPVXkHnPKxSXJ1fH1875/view?usp=sharing',
                               "rel_adj_matrix_url": 'https://drive.google.com/file/d/1fONnjUxrem-dsuPuFy4moSzS_JlBAbC6/view?usp=sharing',
                               "spat_adj_matrix_url": 'https://drive.google.com/file/d/1CE9NVTMp6lVQaVUdGya2_A0VbzFCa617/view?usp=sharing',
                               "kc": 64, "window": 1, "w2v_epochs": 1, "n_of_walks_spat": 5, "walk_length_spat": 10, "p_spat": 1,
                               "q_spat": 4, "n2v_epochs_spat": 100})
#resp = requests.get("http://127.0.0.1:5000/node_classification/task_status/1f755460-cca5-4a93-b2c5-9425bb2a7479")
"""
i = [str(j) for j in range(2052, 2058)]
response = requests.post("http://127.0.0.1:5000/node_classification/predict", json={"job_id": "ce4031fe-d382-4a5a-a97a-feb1f58844dc", "user_ids": ["2052","1077","2814"]+i})
"""
print(response)
print(response.text)


