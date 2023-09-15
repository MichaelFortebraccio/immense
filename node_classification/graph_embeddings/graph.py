import networkx as nx
import stellargraph as sg
import numpy as np


class Graph:
    def __init__(self, path_to_edges, number_of_walks, walk_length, directed, weighted, embedding_size, feature_attribute_name):
        self.path_to_edges = path_to_edges
        self.sgraph = sg.StellarGraph()
        self.nxgraph = nx.DiGraph()
        self.number_of_walks = number_of_walks
        self.walk_length = walk_length
        self.directed = directed
        self.weighted = weighted
        self.embedding_size = embedding_size
        print("loading matrix")
        #self.adj_matrix = np.genfromtxt(adj_matrix, dtype='int', delimiter=",")[1:, 1:]
        print("Matrix loaded")
        self.feature_attribute_name = feature_attribute_name
        #self.node_index_mapping = {}

    def _create_unweighted_graph(self):
        """
        method that creates an unweighted graph starting from the edges
        :return:
        """
        edges = open(self.path_to_edges)
        for e in edges.readlines():
            nodes = e.split()
            self.nxgraph.add_edge(nodes[0], nodes[1])

        '''self.graph = nx.convert_node_labels_to_integers(self.graph, first_label=0, ordering='default', label_attribute="dictionary")
        for n in self.graph.nodes:
            self.node_index_mapping[n] = self.graph.nodes[n]['dictionary']'''

    def instanciate_graph(self, features):
        if not self.weighted:
            self._create_unweighted_graph()
        if features == "random":
            for node_id, data in self.nxgraph.nodes(data=True):
                data[self.feature_attribute_name] = np.random.rand(100)
        else:
            for node_id, data in self.nxgraph.nodes(data=True):
                data[self.feature_attribute_name] = features[node_id]
        self.sgraph = sg.StellarGraph.from_networkx(self.nxgraph, node_features=self.feature_attribute_name)


