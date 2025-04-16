import json
import os
import yaml
from dataset_class import Dataset, normalize_closeness


with open("../dataset_preprocessing.yaml", 'r') as params_file:
    params = yaml.safe_load(params_file)

content_path = params["content_path"]
rel_path = params["rel_path"]
field_name_id = params["field_name_id"]
field_name_text = params["field_name_text"]
spat_edges_fname = params["spat_edges_fname"]
rel_edges_fname = params["rel_edges_fname"]
dataset_dir = params["output_dataset_dir"]
graph_dir = params["output_graph_dir"]
id_field = params["field_name_id"]
text_field = params["field_name_text"]

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
