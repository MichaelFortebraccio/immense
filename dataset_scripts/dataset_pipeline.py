import json
import os
from dataset_class import Dataset, normalize_closeness

content_path = os.path.join("..", "Twitter", "tweets_per_user.json")
rel_path = os.path.join("..", "Twitter", "followers.json")
dataset_dir = os.path.join("..", "dataset", "big_dataset")
graph_dir = os.path.join(dataset_dir, "graph")
spat_edges_fname = os.path.join("graph", "spatial_network.edg")
rel_edges_fname = os.path.join("graph", "social_network.edg")

id_field = "id"
text_field = "text_cleaned"

with open(content_path, 'r') as f:
    d_content = json.load(f)
print("Loaded content json")
with open(rel_path, 'r') as f:
    d_rel = json.load(f)
print("Loaded rel json")

ds = Dataset(posts_dict=d_content, rel_dict=d_rel)
print("Preprocessing...")
df_proc = ds.preprocess_content(id_field_name=id_field, text_field_name=text_field)
print("Preprocessing completed. Saving files...")
df_proc = df_proc.dropna()
df_proc.to_csv(os.path.join(dataset_dir, "unlabelled_dataset.csv"))
print("Positions...")
ds.users_with_pos()
for us in ds.users_with_position:
    us.position_mode()      # Set, for each user, the mode of the locations as its location
dist = ds.calculate_all_closenesses()
normalize_closeness(dist, os.path.join(dataset_dir, spat_edges_fname))
ds.build_rel_network(os.path.join(dataset_dir, rel_edges_fname))
