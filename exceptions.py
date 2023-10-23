class AdjMatException(Exception):
    """
    Exception raised when you don't provide path to the adjacency matrix
    """
    def __init__(self, lab):
        self.message = "You need to provide the path to the {} adjacency matrix".format(lab)
        super().__init__(self.message)

class Id2IdxException(Exception):
    """
    Exception raised when you don't provide path to the id2idx file
    """
    def __init__(self, lab):
        self.message = ("You need to provide the path to the file with the matchings between node IDs and the index of "
                        "their row in the {} adjacency matrix").format(lab)
        super().__init__(self.message)