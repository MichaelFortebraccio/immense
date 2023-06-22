import json
import pandas as pd
import numpy as np
from collections import Counter
from dataset_scripts.tweet import Tweet
from tqdm import tqdm
from utils import load_from_pickle


class Dataset:
    def __init__(self, path_to_posts_json, ids_to_use):
        self.path_to_posts_json = path_to_posts_json
        self.ids_to_use = ids_to_use
        with open(self.path_to_posts_json, 'r') as f:
            d = json.load(f)
        self.d_effective = {str(k): d[str(k)] for k in ids_to_use}
        self.positions = {}
        self.users = []     # List of users who shared their position in at least one tweet

    def users_with_pos(self):
        for k in tqdm(self.d_effective.keys()):
            us = self.d_effective[k]
            for p_k in us.keys():
                post = us[p_k]
                user = User(k)
                if "geo" in post.keys():
                    lat = post['lat']
                    lon = post['lon']
                    country = post['country']
                    user.add_position((lat, lon, country))
                if user.positions:
                    self.users.append(user)


class User:
    def __init__(self, id):
        self.id = id
        self.positions = []

    def add_position(self, pos):
        self.positions.append(pos)

    def position_mode(self):
        """find the most frequent position for the user"""
        if self.positions:
            count = Counter(self.positions)
            p = count.most_common()[0][0]
            self.lat = p[0]
            self.lon = p[1]
            self.country = p[2]

    def calculate_closeness(self, u):
        radius = 6371000
        a = np.sin((self.lat - u.lat)/2)**2 + np.cos(self.lat) * np.cos(u.lat) * np.cos((self.lon - u.lon)/2)**2
        d = 2*radius * np.arctan(np.sqrt(a)/np.sqrt(1-a))
        return d


if __name__ == "__main__":
    l = load_from_pickle("ids_to_use.pkl")
    dat = Dataset(path_to_posts_json="../Twitter/tweets_per_user.json", ids_to_use=l)
    dat.users_with_pos()
    for us in dat.users:
        us.position_mode()      # Set, for each user, the mode of locations as its location
    distances = {}
    for i in tqdm(range(len(dat.users))):
        us1 = dat.users[i]
        for j in range(i+1, len(dat.users)):
            us2 = dat.users[j]
            distances[us1.id+"_"+us2.id] = us1.calculate_closeness(us2)

