from gensim.models import Word2Vec
from pecanpy import node2vec
from os.path import exists


class Node2VecEmbedder():
    def __init__(self, path_to_edges, weighted, directed, n_of_walks, walk_length, embedding_size, p, q, epochs, model_path, name):
        """
        Args:
            path_to_edges:
            number_of_walks:
            walk_length:
            embedding_size:
            p: Defines probability, 1/p, of returning to source node
            q: Defines probability, 1/q, for moving to a node away from the source node. An high value of q will bias
            the model in learning similar representations for nodes that share structure similarity. A low value will
            produce similar representations for nodes in the same neighborhoods
            model_path: Path where the n2v model will be saved
            name: Name of the model (can be either 'closeness' or 'relationshiips'. This parameter is used only for logging purposes
        """
        self.path_to_edges = path_to_edges
        self.weighted = weighted
        self.directed = directed
        self.number_of_walks = n_of_walks
        self.walk_length = walk_length
        self.embedding_size = embedding_size
        self.p = p
        self.q = q
        self.epochs = epochs
        self._model_path = model_path
        self._name = name

    def learn_n2v_embeddings(self, workers=0):
        """
        Args:
            workers: How many threads to use. Set this to 0 for using all the available threads
        """
        g = node2vec.SparseOTF(p=self.p, q=self.q, workers=workers, verbose=True)
        g.read_edg(self.path_to_edges, weighted=self.weighted, directed=self.directed)
        g.preprocess_transition_probs()
        walks = g.simulate_walks(num_walks=self.number_of_walks, walk_length=self.walk_length, n_ckpts=10, pb_len=25)
        model = Word2Vec(sentences=walks, vector_size=self.embedding_size, min_count=0, sg=1, epochs=self.epochs)
        model.save(self._model_path)

    def load_model(self):
        if not exists(self._model_path):
            print("Learning {} n2v model".format(self._name))
            self.learn_n2v_embeddings()
        print("Loading {} n2v model".format(self._name))
        return Word2Vec.load(self._model_path)

