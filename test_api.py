import requests

response = requests.post("http://172.18.0.3:5000/node_classification/train",
                         json={"content_url": 'https://drive.google.com/file/d/1o3kD4jg71IWv5RH9pxrYKMXw-J3CeDPP/view?usp=sharing',
                               "spat_ne_tec": 'pca', "rel_ne_tec": 'none', "kr": 128, "ks": 128,
                               "social_net_url": 'https://drive.google.com/file/d/1MhSo9tMDkfnlvZXPKgxv-HBLP4fSwvmN/view?usp=sharing',
                               "spatial_net_url": 'https://drive.google.com/file/d/1fVipJMfIoqVqnImc9l79tLqzTWlhhXCq/view?usp=sharing',
                               "id2idx_rel_url": 'https://drive.google.com/file/d/1nEdlSo2anF8Hs4rmage1FyRExuXrtQ-s/view?usp=sharing',
                               "id2idx_spat_url": 'https://drive.google.com/file/d/16o4botYT2gOTQPVXkHnPKxSXJ1fH1875/view?usp=sharing',
                               "rel_adj_matrix_url": 'https://drive.google.com/file/d/1fONnjUxrem-dsuPuFy4moSzS_JlBAbC6/view?usp=sharing',
                               "spat_adj_matrix_url": 'https://drive.google.com/file/d/1CE9NVTMp6lVQaVUdGya2_A0VbzFCa617/view?usp=sharing',
                               "kc": 64, "window": 10, "w2v_epochs": 1})
"""
i = [str(j) for j in range(2052, 2058)]
response = requests.post("http://127.0.0.1:5000/node_classification/predict", json={"job_id": "912fa58f-500a-4b5e-ab70-fa4c615b0e93", "user_ids": ["121212121","2052","1077","2814"]})       # ,"2052","1077","2814"]+i
"""
print(response)
print(response.text)


