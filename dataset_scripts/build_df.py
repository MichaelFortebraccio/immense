import pandas as pd
import json
from tqdm import tqdm

path = "../Twitter/tweets_per_user.json"

d = pd.DataFrame(columns=["tweet_id", "user_id", "text"])
with open(path, 'rb') as f:
    users = json.load(f)

l = []
for k in tqdm(users.keys()):
    if users[k]:
        for tk in users[k].keys():
            l.append({"user_id": k, "tweet_id": tk, "text": users[k][tk]['text'].replace("\n", " ")})
d = pd.DataFrame.from_dict(l)
d.to_csv("tweets_users.csv")