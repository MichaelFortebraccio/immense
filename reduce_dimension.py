from os.path import join, exists
import torch
import torch_geometric.transforms as T
from modelling.sage import SAGE, create_mappers, create_graph
from torch_geometric.loader import NeighborLoader
from utils import save_to_pickle


def reduce_dimension(lab, model_dir, ne_dim, train_df, we_dim, batch_size, edge_path, epochs, features_dict, sizes,
                     training_weights, separator, field_name_id, field_name_label, retrain=False):
    """
    This function applies one of the node dimensionality reduction techniques and generate the feature vectors for
    training the decision tree.
    Args:
        :param lab: Label, can be either "spat" or "rel".
        :param model_dir: Directory where the models will be saved.
        :param ne_dim: Dimension of the embeddings to create.
        :param train_df: Dataframe with the training data. The IDs will be used.
        :param batch_size: Batch size to use during training.
        :param edge_path: Path to file containing the list of edges. The file shall contain one row per each edge,
        and the row shall contain the ids of the nodes being connected and, in case of the spatial network, their
        distance. Example 1:
        12\t15
        13\t17
        means that user 12 follows user 15 and user 13 follows user 17. Example 2:
        12\t5\t0.52
        means that the users with id 12 and 5 have spatial distance equal to 0.52
        :param epochs: Epochs for training the node embedding model.
        :param features_dict: (graphsage) Dictionary having as keys the IDs of the users and as values the sum of the
        embeddings of their posts.
        :param sizes: Array containing the number of neighbors to sample for each node.
        :param training_weights: tensor of shape (1, num_classes) containing the weights to give to each class while
        training the graphsage model. If None, no weights will be used
    Returns:
        predictions (n, num_classes): Predictions made by the node embedding model for the nodes. For each node, its
        prediction is the computed probability of the node to belong to each of the class
    """
    weights_path = join(model_dir, "graphsage_{}_{}.h5".format(ne_dim, we_dim))
    model_path = join(model_dir, "graphsage_{}_{}.pkl".format(ne_dim, we_dim))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    first_key = list(features_dict.keys())[0]
    in_channels = len(features_dict[first_key])
    weighted = False
    directed = True
    if lab == "spat":
        weighted = True
        directed = False

    mapper_train, inv_map_train = create_mappers(features_dict)
    graph = create_graph(inv_map=inv_map_train, weighted=weighted, features=features_dict, edg_dir=edge_path,
                         df=train_df, separator=separator, field_name_id=field_name_id, field_name_label=field_name_label)
    graph = graph.to(device)
    split = T.RandomLinkSplit(num_val=0.1, num_test=0.0, is_undirected=not directed,
                              add_negative_train_samples=False, neg_sampling_ratio=1.0)
    train_data, valid_data, _ = split(graph)
    sage = SAGE(in_dim=in_channels, hidden_dim=ne_dim, num_layers=len(sizes), weighted=weighted, directed=directed)
    sage = sage.to(device)
    if training_weights is not None:
        training_weights = training_weights.to(device)

    train_loader = NeighborLoader(train_data, num_neighbors=sizes, batch_size=batch_size)
    if not exists(model_path) or retrain:
        print("Training {} node embedding model\n".format(lab))
        optimizer = torch.optim.Adam(lr=.01, params=sage.parameters(), weight_decay=1e-4)
        best_loss = 9999
        for i in range(epochs):
            loss = sage.train_sage(train_loader, optimizer=optimizer, weights=training_weights)
            val_loss = sage.test(valid_data)
            if loss < best_loss:
                best_loss = loss
                print("New best model found at epoch {}. Loss: {}, val_loss: {}".format(i, loss, val_loss))
                torch.save(sage.state_dict(), weights_path)
            if i % 5 == 0:
                print("Epoch {}: train loss {}, val loss: {}".format(i, loss, val_loss))
        sage.load_state_dict(torch.load(weights_path))
        save_to_pickle(model_path, sage)
    else:
        sage.load_state_dict(torch.load(weights_path))
    predictions = sage(graph, inference=True).cpu()
    predictions = predictions.detach().numpy()

    return predictions
