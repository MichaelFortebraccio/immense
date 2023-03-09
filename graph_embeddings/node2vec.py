from gensim.models import Word2Vec
from pecanpy import node2vec
import numpy as np
import os


class Node2VecEmbedder():
    def __init__(self, path_to_edges, number_of_walks, walk_length, embedding_size, p, q, epochs, save_path):
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
            save_path: Path to the directory where the model will be saved
        """
        self.path_to_edges = path_to_edges
        self.number_of_walks = number_of_walks
        self.walk_length = walk_length
        self.embedding_size = embedding_size
        self.p = p
        self.q = q
        self.epochs = epochs
        self.save_path = os.path.join(save_path, "node2vec_{}.model".format(self.embedding_size))

    def learn_n2v_embeddings(self, workers=0):
        """
        Args:
            workers: How many threads to use. Set this to 0 for using all the available threads
        """
        g = node2vec.SparseOTF(p=self.p, q=self.q, workers=workers, verbose=True)
        g.read_edg(self.path_to_edges, weighted=False, directed=True)
        g.preprocess_transition_probs()
        walks = g.simulate_walks(num_walks=self.number_of_walks, walk_length=self.walk_length, n_ckpts=10, pb_len=25)
        model = Word2Vec(sentences=walks, vector_size=self.embedding_size, min_count=0, sg=1, epochs=self.epochs)
        model.save(self.save_path)

    def load_model(self):
        if os.path.exists(self.save_path):
            return Word2Vec.load(self.save_path)
        else:
            return None


if __name__ == "__main__":
    embedder = Node2VecEmbedder(path_to_edges="stuff/network_right_ids.edg", number_of_walks=10, walk_length=10,
                                embedding_size=128, p=4, q=1, epochs=10, save_dir="stuff")
    embedder.learn_n2v_embeddings()

