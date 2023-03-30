import requests

"""response = requests.post("http://127.0.0.1:5000/node_classification/train",
                         json={"content_url": 'https://drive.google.com/file/d/1o3kD4jg71IWv5RH9pxrYKMXw-J3CeDPP/view?usp=sharing',
                               "social_net_url": 'https://drive.google.com/file/d/1MhSo9tMDkfnlvZXPKgxv-HBLP4fSwvmN/view?usp=sharing',
                               'spatial_net_url': 'https://drive.google.com/file/d/1fVipJMfIoqVqnImc9l79tLqzTWlhhXCq/view?usp=sharing',
                               "kc": 512, "window": 5, "w2v_epochs": 1, "kr": 128, "ks": 128, "n_of_walks_spat": 10,
                               "n_of_walks_rel": 10, "walk_length_spat": 10, "walk_length_rel": 10, "p_spat": 1,
                               "p_rel": 1, "q_spat": 4, "q_rel": 4, "n2v_epochs_spat": 100, "n2v_epochs_rel": 100})
"""
i = [str(j) for j in range(2052, 2058)]
response = requests.post("http://127.0.0.1:5000/node_classification/predict", json={"job_id": "ce4031fe-d382-4a5a-a97a-feb1f58844dc", "user_ids": ["2052","1077","2814"]+i})

print(response)
print(response.text)


