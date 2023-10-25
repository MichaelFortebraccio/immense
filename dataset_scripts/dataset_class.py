import json
import pandas as pd
import numpy as np
from collections import Counter
from dataset_scripts.dataset_utils import clean_dataframe, concatenate_posts
from tqdm import tqdm
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
            if str(t[2]) != 0.0:
                f.write(str(t[0]) + "\t" + str(t[1]) + "\t" + str(t[2]))
                f.write("\n")


class Dataset:
    def __init__(self, posts_dict, rel_dict, ids_to_use=None):
        self.posts_dict = posts_dict
        self.ids_to_use = ids_to_use
        self.rel_dict = rel_dict
        self.positions = {}
        self.users_with_position = []     # List of users who shared their position in at least one tweet

    def preprocess_content(self, id_field_name, text_field_name):
        users = self.posts_dict
        l = []
        for k in tqdm(users.keys()):
            if users[k]:
                for tk in users[k].keys():
                    l.append({id_field_name: k, "tweet_id": tk, text_field_name: users[k][tk]['text'].replace("\n", " ")})
        df = pd.DataFrame.from_dict(l)
        cleaned_df = clean_dataframe(df, text_column=text_field_name)
        if len(set(cleaned_df[id_field_name].values)) != len(cleaned_df[id_field_name].values):
            cleaned_df = concatenate_posts(cleaned_df, aggregator_column=id_field_name, text_column=text_field_name)
        return cleaned_df

    def users_with_pos(self):
        for k in tqdm(self.posts_dict.keys()):
            us = self.posts_dict[k]
            user = User(k)
            for p_k in us.keys():
                post = us[p_k]
                if "geo" in post.keys():
                    lat = post['lat']
                    lon = post['lon']
                    country = post['country']
                    user.add_position((lat, lon, country))
            if user.positions:
                self.users_with_position.append(user)

    def calculate_all_closenesses(self):
        """
        Calculates the unnormalized closenesses for all the possible couples of users. Since the distances are symmetric,
        the closeness between two points is calculated only once because closeness(A, B) = closeness(B, A)
        :return: Dictionary having as keys the couples of user ids and as values the distance between those users
        """
        dist = {}
        for i in tqdm(range(len(self.users_with_position))):
            us1 = self.users_with_position[i]
            for j in range(i + 1, len(self.users_with_position)):
                us2 = self.users_with_position[j]
                dist[us1.id + "_" + us2.id] = us1.calculate_closeness(us2)
        return dist

    def build_rel_network(self, dst_fname):
        d_users = self.rel_dict
        l = []
        for user in d_users.keys():
            for followed in d_users[user]:
                l.append("{}\t{}\n".format(user, followed))
        with open(dst_fname, 'w') as f2:
            for e in l:
                f2.write(e)


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
