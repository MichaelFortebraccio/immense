import networkx as nx
import stellargraph as sg
import numpy as np


class Graph:
    def __init__(self, path_to_edges, number_of_walks, walk_length):
        self.path_to_edges = path_to_edges
        self.sgraph = sg.StellarGraph()
        self.nxgraph = nx.DiGraph()
        print("loading matrix")
        self.adj_matrix = np.genfromtxt(adj_matrix, dtype='int', delimiter=",")[1:, 1:]
        print("Matrix loaded")
        self.feature_attribute_name = feature_attribute_name
        #self.node_index_mapping = {}

    def create_unweighted_graph(self):
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

    def add_features(self, type, len):
        """
        :param type: can be "random" for generating a random feature vector, "const" for generating a feature vector
        filled with constant values (1s), "row" for having, for each node, a feature vector made of the corresponding
        row in the adjacency matrix
        :param len: Length of the feature vector. It is ignored if choosing "row" as type
        :return:
        """
        assert self.feature_attribute_name
        for node_id, data in self.nxgraph.nodes(data=True):
            if type == "random":
                data[self.feature_attribute_name] = np.random.rand(len)
            elif type == "const":
                data[self.feature_attribute_name] = np.array([1]*10)
            elif type == "row":
                data[self.feature_attribute_name] = self.adj_matrix[int(node_id)]
            elif type == "degree":
                data[self.feature_attribute_name] = [self.nxgraph.degree[node_id]]

    def instanciate_graph(self):
        self.sgraph = sg.StellarGraph.from_networkx(self.nxgraph, node_features=self.feature_attribute_name)


