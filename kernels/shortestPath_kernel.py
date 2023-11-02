import networkx as nx
from kernels.RandomWalk_kernel import extract_graphlets_from_walk
from kernels.graphlet_kernel import count_graphlets,graphlet_kernel_similarity
import itertools

def extract_graphlets_from_shortest_paths(graph, sizes=[2, 3, 4]):
    """
    Extract graphlets of specified sizes from shortest paths between all pairs of nodes in a graph.

    :param graph: A NetworkX graph
    :param sizes: Sizes of graphlets to extract
    :return: A list of graphlets (each graphlet is a NetworkX graph)
    """
    graphlets = []
    for source, target in itertools.combinations(graph.nodes(), 2):
        try:
            # Find the shortest path between source and target
            shortest_path = nx.shortest_path(graph, source=source, target=target)
            # Extract graphlets from the shortest path
            path_graphlets = extract_graphlets_from_walk(graph, shortest_path, sizes)
            graphlets.extend(path_graphlets)
        except nx.NetworkXNoPath:
            continue  # No path between source and target
    return graphlets


import numpy as np


def apply_shortest_path_graphlet_kernel(graph1, graph2, sizes=[2, 3, 4]):
    """
    Apply the Shortest Path Graphlet Kernel to compute similarity between two graphs.

    :param graph1: A NetworkX graph
    :param graph2: A NetworkX graph
    :param sizes: Sizes of graphlets to consider
    :return: Similarity between the two graphs
    """
    # Extract graphlets from shortest paths and count them for graph1
    graphlets_1 = extract_graphlets_from_shortest_paths(graph1, sizes)
    graphlet_counts_1 = count_graphlets(graphlets_1)

    # Extract graphlets from shortest paths and count them for graph2
    graphlets_2 = extract_graphlets_from_shortest_paths(graph2, sizes)
    graphlet_counts_2 = count_graphlets(graphlets_2)

    # Compute similarity between the two graphs
    similarity = graphlet_kernel_similarity(graphlet_counts_1, graphlet_counts_2)

    return similarity
