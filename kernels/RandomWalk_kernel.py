import itertools
from collections import defaultdict
import numpy as np
import random
from kernels.graphlet_kernel import count_graphlets,graphlet_kernel_similarity


def perform_random_walk(graph, length):
    """
    Perform a random walk of a specified length on the graph.

    :param graph: A NetworkX graph
    :param length: The length of the random walk
    :return: A list of nodes visited during the random walk
    """
    # Start from a random node
    current_node = random.choice(list(graph.nodes()))
    walk = [current_node]
    for _ in range(length - 1):
        # Get neighbors of the current node
        neighbors = list(graph.neighbors(current_node))
        if not neighbors:
            break  # If the node has no neighbors, stop the walk
        # Choose a random neighbor to visit next
        current_node = random.choice(neighbors)
        walk.append(current_node)

    return walk


def extract_graphlets_from_walk(graph, walk, sizes=[2, 3, 4]):
    """
    Extract graphlets of specified sizes from a random walk on a graph.

    :param graph: A NetworkX graph
    :param walk: A list of nodes visited during the random walk
    :param sizes: Sizes of graphlets to extract
    :return: A list of graphlets (each graphlet is a NetworkX graph)
    """
    graphlets = []
    for size in sizes:
        for i in range(len(walk) - size + 1):
            sub_nodes = walk[i:i + size]
            subgraph = graph.subgraph(sub_nodes)
            if subgraph.number_of_edges() > 0:
                graphlets.append(subgraph)
    return graphlets

def apply_random_walk_graphlet_kernel(graph1, graph2, walk_length=10, num_walks=100, sizes=[2, 3, 4]):
    """
    Apply the Random Walk Graphlet Kernel to compute similarity between two graphs.

    :param graph1: A NetworkX graph
    :param graph2: A NetworkX graph
    :param walk_length: The length of the random walks
    :param num_walks: The number of random walks to perform on each graph
    :param sizes: Sizes of graphlets to consider
    :return: Similarity between the two graphs
    """
    # Perform random walks, extract and count graphlets for graph1
    counts_1 = defaultdict(int)
    for _ in range(num_walks):
        walk = perform_random_walk(graph1, walk_length)
        graphlets = extract_graphlets_from_walk(graph1, walk, sizes)
        graphlet_counts = count_graphlets(graphlets)
        for key, value in graphlet_counts.items():
            counts_1[key] += value

    # Perform random walks, extract and count graphlets for graph2
    counts_2 = defaultdict(int)
    for _ in range(num_walks):
        walk = perform_random_walk(graph2, walk_length)
        graphlets = extract_graphlets_from_walk(graph2, walk, sizes)
        graphlet_counts = count_graphlets(graphlets)
        for key, value in graphlet_counts.items():
            counts_2[key] += value

    # Compute similarity between the two graphs
    similarity = graphlet_kernel_similarity(counts_1, counts_2)

    return similarity
