from stellargraph.mapper import GraphSAGELinkGenerator
from stellargraph.layer import GraphSAGE, link_classification
from stellargraph.data import UniformRandomWalk
from stellargraph.data import UnsupervisedSampler
from tensorflow import keras
from sklearn import preprocessing, feature_extraction, model_selection
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from stellargraph.mapper import GraphSAGENodeGenerator
from sklearn.metrics import accuracy_score
from node_classification.graph_embeddings.graph import Graph
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import binary_crossentropy
from keras.models import save_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.metrics import binary_accuracy
import pickle


class GraphSAGEEmbedder(Graph):
    def __init__(self, path_to_edges, number_of_walks, walk_length, num_samples, layer_sizes, directed, embedding_size, weighted, feature_attribute_name):
        super().__init__(path_to_edges, number_of_walks, walk_length, directed, weighted, embedding_size, feature_attribute_name)
        self.num_samples = num_samples
        self.layer_sizes = layer_sizes

    def create_graphsage_model(self, batch_size, dropout=0.0, start_nodes=None):
        """
        Sample couple of positive and negative edges and use them to train a link classification model
        :param batch_size:
        :param dropout:
        :param epochs:
        :param lr:
        :return:
        """
        if not start_nodes:
            start_nodes = self.sgraph.nodes()
        unsupervised_samples = UnsupervisedSampler(self.sgraph, nodes=start_nodes, length=self.walk_length, number_of_walks=self.number_of_walks,seed=42)
        # Generates training data for the encoder model
        generator = GraphSAGELinkGenerator(self.sgraph, batch_size, self.num_samples)
        train_gen = generator.flow(unsupervised_samples)
        graphsage = GraphSAGE(layer_sizes=self.layer_sizes, generator=generator, bias=True, dropout=dropout, normalize="l2")     # encoder, produces embeddings
        return graphsage, train_gen

    def learn_embeddings(self, graphsage, generator, epochs, t="ip", lr=1e-3):
        x_inp, x_out = graphsage.in_out_tensors()
        # get the outputs of the encoder and performs link prediction to state if two nodes can be linked
        model_checkpoint_callback = ModelCheckpoint(
            filepath='models/{}_{}/'.format(self.layer_sizes[-1], t),
            save_weights_only=False,
            monitor='loss',
            mode='min',
            verbose=0,
            save_best_only=True)
        prediction = link_classification(output_dim=1, output_act="sigmoid", edge_embedding_method=t)(x_out)
        model = keras.Model(x_inp, prediction)
        model.compile(optimizer=Adam(lr), loss=binary_crossentropy, metrics=[binary_accuracy])
        history = model.fit(generator, epochs=epochs, verbose=0, workers=4, shuffle=True, callbacks=[model_checkpoint_callback, EarlyStopping(monitor='loss', patience=20)])
        self.model = model
        return x_inp, x_out


if __name__ == "__main__":
    num_samples = [10, 5]
    number_of_walks = 3
    walk_length = 4
    epochs = 10
    batch_size = 512
    d = {}
    mode = 'random'
    t = 'ip'
    layer_sizes = [32, 32]
    g = GraphSAGEEmbedder(path_to_edges="stuff/sn.edg", number_of_walks=number_of_walks, walk_length=walk_length, num_samples=num_samples,
                          layer_sizes=layer_sizes, directed=True, weighted=False, embedding_size=32, feature_attribute_name="boh")
    #print(g.sgraph.nodes()[1])
    g.instanciate_graph(mode)
    graphsage, generator = g.create_graphsage_model(batch_size=batch_size)
    x_inp, x_out = g.learn_embeddings(graphsage=graphsage, generator=generator, epochs=epochs, t=t)
    #d[mode+"_"+str(ls)+"_"+t] = (history.history['loss'][-1], history.history['binary_accuracy'][-1])
    x_inp_src = x_inp[0::2]
    x_out_src = x_out[0]
    embedding_model = keras.Model(inputs=x_inp_src, outputs=x_out_src)
    node_ids = g.sgraph.nodes()
    batch_size = 512
    node_gen = GraphSAGENodeGenerator(g.sgraph, batch_size, num_samples, seed=42).flow(node_ids)
    node_embeddings0 = embedding_model.predict(node_gen, verbose=1)
    node_embeddings1 = embedding_model.predict(node_gen)
    print(node_embeddings.shape)