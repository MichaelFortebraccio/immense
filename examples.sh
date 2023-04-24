# The following trains sairus using node2vec as technique for learning both spatial and relational node embeddings. Spatial node embeddings have size 64,
# relational node embeddings have size 128. The other parameters are left to their default values
python main_train.py --textual_content_path dataset/posts_labeled.csv --social_net_url node_classification/graph_embeddings/stuff/social_network.edg --spatial_net_url node_classification/graph_embeddings/stuff/spatial_network.edg --word_embedding_size 256 --w2v_epochs 3 --spat_technique node2vec --rel_technique node2vec --spat_node_embedding_size 128 --rel_node_embedding_size 64

# Test the previously learned model
python main_test.py --spat_technique node2vec --rel_technique node2vec --models_dir models --dataset_dir dataset --word_embedding_size 256

# Use the previously learned model for getting the predictio for the user with id== 56
python main_test.py --spat_technique node2vec --rel_technique node2vec --models_dir models --dataset_dir dataset --word_embedding_size 256 --user_id 56
