from sklearn.metrics.pairwise import rbf_kernel
import numpy as np

import numpy as np


def gaussian_rbf_kernel(x_i, x_j, sigma):
    """Calculate Gaussian RBF kernel."""
    return np.exp(-np.square(x_i - x_j) / (2 * sigma ** 2))


def kronecker_delta(x_i, x_j):
    """Calculate Kronecker delta."""
    return 1 if x_i == x_j else 0


def node_similarity(node_1, node_2, sigma=1):
    """
    Calculate similarity between two nodes based on their features.

    Args:
    node_1, node_2: dict
        Feature dictionaries of the two nodes.
    sigma: float
        Hyperparameter for Gaussian RBF kernel.

    Returns:
    similarity: float
        Similarity score between the two nodes.
    """
    # Ensure the nodes are of the same type
    if node_1['Type'] != node_2['Type']:
        return 0

    similarity = 0
    node_type = node_1['Type']

    # Define the features for each type of node
    continuous_features = {
        'Chip unit': ['power'],
        'CPU': ['Frequency'],
        'RAM': ['Size'],
        'Flash Mem': ['Size'],
        'ADC': ['Sensitivity_on_range', 'Frequency'],
        'DAC': ['Res', 'Lin'],
        'Battery': ['Output', 'Capacity'],
        'RDL': ['Speed'],
        'Transceiver': ['Frequency', 'Power'],
        'Material': ['Footprint'],
        'Server': ['Footprint'],
    }

    categorical_features = {
        'CPU': ['Subtype', 'Generation'],
        'RAM': ['subtype'],
        'Flash Mem': ['subtype'],
        'ADC': ['Subtype'],
        'DAC': ['Subtype'],
        'Battery': ['Subtype'],
        'Transceiver': [],
        'Material': [],
        'Server': [],
        'RDL': ['Num', 'Bits'],
        'Chip unit': [],
    }

    # Calculate similarity for continuous features
    for feature in continuous_features[node_type]:
        similarity += gaussian_rbf_kernel(node_1.get(feature, 0), node_2.get(feature, 0), sigma)

    # Calculate similarity for categorical features
    for feature in categorical_features[node_type]:
        similarity += kronecker_delta(node_1.get(feature), node_2.get(feature))

    # Normalize similarity
    total_features = len(continuous_features[node_type]) + len(categorical_features[node_type])
    if total_features > 0:
        similarity /= total_features

    return similarity


def edge_similarity(edge_1, edge_2):
    """
    Calculate similarity between two edges based on their types.

    Args:
    edge_1, edge_2: tuple
        Each tuple contains the edge attributes.

    Returns:
    similarity: float
        Similarity score between the two edges.
    """
    # Extract edge types
    edge_type_1 = edge_1[2]['type'][0] if 'type' in edge_1[2] else ''
    edge_type_2 = edge_2[2]['type'][0] if 'type' in edge_2[2] else ''

    # Calculate similarity using Kronecker delta
    similarity = kronecker_delta(edge_type_1, edge_type_2)
    return similarity


def graph_kernel(graph_1, graph_2, sigma=1):
    """
    Calculate graph kernel (similarity) between two graphs.

    Args:
    graph_1, graph_2: networkx.Graph
        The two graphs to compare.
    sigma: float
        Hyperparameter for Gaussian RBF kernel in node similarity calculation.

    Returns:
    similarity: float
        Similarity score between the two graphs.
    """
    # Calculate node similarities
    node_similarities = []
    for node_1 in graph_1.nodes(data=True):
        for node_2 in graph_2.nodes(data=True):
            sim = node_similarity(node_1[1], node_2[1], sigma)
            node_similarities.append(sim)

    # Calculate edge similarities
    edge_similarities = []
    for edge_1 in graph_1.edges(data=True):
        for edge_2 in graph_2.edges(data=True):
            sim = edge_similarity(edge_1, edge_2)
            edge_similarities.append(sim)

    # Aggregate similarities
    if node_similarities:
        avg_node_similarity = np.mean(node_similarities)
    else:
        avg_node_similarity = 0

    if edge_similarities:
        avg_edge_similarity = np.mean(edge_similarities)
    else:
        avg_edge_similarity = 0

    # Final similarity score (average of node and edge similarities)
    similarity = (avg_node_similarity + avg_edge_similarity) / 2
    return similarity