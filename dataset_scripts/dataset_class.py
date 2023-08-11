import json
import pandas as pd
import numpy as np
from collections import Counter
from tqdm import tqdm
from utils import load_from_pickle
from scipy.stats import zscore


def normalize_closeness(d, file_name):
    """
    Take the unnormalized values and normalize them
    :param d: dictionary having as keys the couples of user ids and as values the distance between those users
    :param file_name: the couples will be exported to a file. Set this param with the path of the file.
    :return:
    """
    ar = np.fromiter(d.values(), dtype=float)
    z_ar = zscore(ar)
    z_ar[z_ar > 0] = 0  # Couples with z_scored_distance > 0 are farther than average therefore the value is zero-ed
    min = np.min(z_ar)
    for idx in np.argwhere(z_ar < 0):
        z_ar[idx] = z_ar[idx] / min  # Couples with z_scored_distance < 0 are closer than average therefore the normalized distance value is divided by the minimum

    keys = list(d.keys())
    l = []
    for i in range(len(keys)):
        k = keys[i]
        k = k.split("_")
        l.append((k[0], k[1], z_ar[i]))

    with open(file_name, "w") as f:
        for t in l:
            f.write(str(t[0]) + "\t" + str(t[1]) + "\t" + str(t[2]))
            f.write("\n")


class Dataset:
    def __init__(self, path_to_posts_json, ids_to_use=None):
        self.path_to_posts_json = path_to_posts_json
        self.ids_to_use = ids_to_use
        with open(self.path_to_posts_json, 'r') as f:
            d = json.load(f)
        if not self.ids_to_use:
            self.d_effective = d
        else:
            self.d_effective = {str(k): d[str(k)] for k in ids_to_use}
        self.positions = {}
        self.users = []     # List of users who shared their position in at least one tweet

    def users_with_pos(self):
        for k in tqdm(self.d_effective.keys()):
            us = self.d_effective[k]
            user = User(k)
            for p_k in us.keys():
                post = us[p_k]
                if "geo" in post.keys():
                    lat = post['lat']
                    lon = post['lon']
                    country = post['country']
                    user.add_position((lat, lon, country))
            if user.positions:
                self.users.append(user)

    def calculate_all_closenesses(self):
        """
        Calculates the unnormalized closenesses for all the possible couples of users. Since the distances are symmetric,
        the closeness between two points is calculated only once because closeness(A, B) = closeness(B, A)
        :return: Dictionary having as keys the couples of user ids and as values the distance between those users
        """
        dist = {}
        for i in tqdm(range(len(self.users))):
            us1 = self.users[i]
            for j in range(i + 1, len(self.users)):
                us2 = self.users[j]
                dist[us1.id + "_" + us2.id] = us1.calculate_closeness(us2)
        return dist


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
    path_to_ids = "ids_to_use.pkl"
    path_to_posts_json = "../Twitter/tweets_per_user.json"
    path_to_edg = "../dataset/graph/closeness_network_all_users.edg"

    l = load_from_pickle(path_to_ids)
    dat = Dataset(path_to_posts_json=path_to_posts_json, ids_to_use=None)
    dat.users_with_pos()
    for us in dat.users:
        us.position_mode()      # Set, for each user, the mode of locations as its location
    dist = dat.calculate_all_closenesses()
    normalize_closeness(dist, path_to_edg)








