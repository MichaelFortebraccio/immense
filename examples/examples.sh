# The following trains sairus using node2vec as technique for learning both spatial and relational node embeddings. Spatial node embeddings have size 64,
# relational node embeddings have size 128. The other parameters are left to their default values
python main_train.py --textual_content_path dataset/posts_labeled.csv --social_net_path dataset/graph/rel_network.edg --spatial_net_path dataset/graph/closeness_network_all_users.edf --word_embedding_size 300 --w2v_epochs 3 --spat_technique node2vec --rel_technique node2vec --spat_node_embedding_size 128 --rel_node_embedding_size 128 --models_dir dataset/models

# Now we train a new version of sairus, using node2vec as spatial node embedding technique, and PCA as relational node embedding technique
python main_train.py --textual_content_path dataset/posts_labeled.csv --spatial_net_path node_classification/graph_embeddings/stuff/spatial_network.edg --word_embedding_size 256 --w2v_epochs 3 --spat_technique node2vec --rel_technique pca --spat_node_embedding_size 128 --rel_node_embedding_size 64 --rel_adj_mat_path node_classification/graph_embeddings/stuff/rel_adj_net.csv

# Test the first learned model
python main_test.py --spat_technique node2vec --rel_technique node2vec --models_dir models --dataset_dir dataset --word_embedding_size 256
 #Test the second learned model
python main_test.py --spat_technique node2vec --rel_technique pca --models_dir models --dataset_dir dataset --word_embedding_size 256 --rel_adj_mat_path node_classification/graph_embeddings/stuff/rel_adj_net.csv --id2idx_rel_path node_classification/graph_embeddings/stuff/id2idx_rel.pkl


# Use the first learned model for getting the predictio for the user with id== 56
python main_test.py --spat_technique node2vec --rel_technique node2vec --models_dir models --dataset_dir dataset --word_embedding_size 256 --user_id 56
