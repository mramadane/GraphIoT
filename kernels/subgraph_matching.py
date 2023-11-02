from collections import defaultdict
import numpy as np
from itertools import combinations


def enumerate_subgraphs(graph, subgraph_size):
    """
    Enumerate all subgraphs of a given size in the graph.
    """
    subgraphs = []
    for sub_nodes in combinations(graph.nodes(), subgraph_size):
        subgraph = graph.subgraph(sub_nodes)
        if subgraph.size() > 0:  # Include only subgraphs with at least one edge
            # Convert subgraph to canonical form (sorted node labels and sorted edge labels)
            sorted_node_labels = sorted([graph.nodes[node]['label'] for node in subgraph.nodes()])
            sorted_edge_labels = sorted([graph[node][neighbor].get('label', 'default') for node, neighbor in subgraph.edges()])
            canonical_form = (tuple(sorted_node_labels), tuple(sorted_edge_labels))
            subgraphs.append(canonical_form)
    return subgraphs


def subgraph_matching_kernel(graph1, graph2, subgraph_size=2):
    """
    Compute the subgraph matching kernel between two graphs.
    """
    # Enumerate subgraphs of the given size in both graphs
    subgraphs_1 = enumerate_subgraphs(graph1, subgraph_size)
    subgraphs_2 = enumerate_subgraphs(graph2, subgraph_size)

    # Create feature vectors for both graphs
    feature_vector_1 = defaultdict(int)
    feature_vector_2 = defaultdict(int)
    for subgraph in subgraphs_1:
        feature_vector_1[subgraph] += 1
    for subgraph in subgraphs_2:
        feature_vector_2[subgraph] += 1

    # Compute the dot product of the feature vectors
    kernel_value = sum(feature_vector_1[subgraph] * feature_vector_2[subgraph] for subgraph in set(feature_vector_1)
                       & set(feature_vector_2))

    return kernel_value
