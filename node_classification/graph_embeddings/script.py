from node_classification.graph_embeddings.graphsage import GraphSAGEEmbedder
import random
from stellargraph.mapper import GraphSAGENodeGenerator
from tensorflow import keras
import tensorflow as tf
import numpy as np

random.seed(42)
tf.random.set_seed(42)
np.random.seed(42)

num_samples = [10, 5]
number_of_walks = 3
walk_length = 4
epochs = 1
batch_size = 512
t = 'ip'
mode = 'random'
layer_sizes = [32, 64]

g = GraphSAGEEmbedder(path_to_edges="stuff/sn.edg", number_of_walks=number_of_walks, walk_length=walk_length, num_samples=num_samples,
                      layer_sizes=layer_sizes, directed=True, weighted=False, embedding_size=32, feature_attribute_name="boh")
g.instanciate_graph("random")
graphsage, generator = g.create_graphsage_model(batch_size=batch_size)
x_in, x_out = g.learn_embeddings(graphsage=graphsage, generator=generator, epochs=epochs, t=t)

g2 = GraphSAGEEmbedder(path_to_edges="stuff/sn2.edg", number_of_walks=number_of_walks, walk_length=walk_length, num_samples=num_samples,
                     layer_sizes=layer_sizes, directed=True, weighted=False, embedding_size=32, feature_attribute_name="boh")
g2.instanciate_graph("random")
x_inp_src = x_in[0::2]
x_out_src = x_out[0]
embedding_model = keras.Model(inputs=x_inp_src, outputs=x_out_src)
node_ids = ["968225558"]
node_gen = GraphSAGENodeGenerator(g.sgraph, batch_size, num_samples, seed=42).flow(node_ids)
print(embedding_model.predict(node_gen)[0])
print(embedding_model.predict(node_gen)[0])