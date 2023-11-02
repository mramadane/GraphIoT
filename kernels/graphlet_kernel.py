import itertools
from collections import defaultdict
import numpy as np


def extract_graphlets(graph, size):
    """
    Extract all graphlets of a given size from the graph.

    :param graph: A NetworkX graph
    :param size: Size of the graphlets to extract
    :return: A list of graphlets (each graphlet is represented as a set of nodes)
    """
    graphlets = []
    for sub_nodes in itertools.combinations(graph.nodes(), size):
        subgraph = graph.subgraph(sub_nodes)
        if subgraph.number_of_edges() > 0:
            graphlets.append(subgraph)
    return graphlets


def count_graphlets(graphlets):
    """
    Count the occurrences of each type of graphlet based on node "Type" attributes.

    :param graphlets: A list of graphlets (each graphlet is a NetworkX graph)
    :return: A dictionary with graphlet types as keys and their counts as values
    """
    counts = defaultdict(int)
    for graphlet in graphlets:
        # Get the "Type" attributes of the nodes in the graphlet
        types = tuple(sorted([graphlet.nodes[node]['Type'] for node in graphlet.nodes()]))
        # Update the count of this type of graphlet
        counts[types] += 1
    return counts


def graphlet_kernel_similarity(counts1, counts2):
    """
    Compute the similarity between two graphs based on graphlet counts.

    :param counts1: A dictionary of graphlet counts for the first graph
    :param counts2: A dictionary of graphlet counts for the second graph
    :return: The similarity between the two graphs
    """
    # Combine the keys from both dictionaries
    all_keys = set(counts1.keys()) | set(counts2.keys())

    # Compute the inner product of the count vectors
    inner_product = sum(counts1.get(key, 0) * counts2.get(key, 0) for key in all_keys)

    # Compute the norms of the count vectors
    norm1 = np.sqrt(sum(count ** 2 for count in counts1.values()))
    norm2 = np.sqrt(sum(count ** 2 for count in counts2.values()))

    # Compute the similarity
    similarity = inner_product / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0
    return similarity


def apply_graphlet_kernel(graph1, graph2, sizes=[2, 3, 4]):
    """
    Apply the graphlet kernel to compute similarity between two graphs.

    :param graph1: A NetworkX graph
    :param graph2: A NetworkX graph
    :param sizes: Sizes of graphlets to consider
    :return: Similarity between the two graphs
    """
    # Extract and count graphlets for each graph
    counts_1 = defaultdict(int)
    counts_2 = defaultdict(int)

    for size in sizes:
        graphlets_1 = extract_graphlets(graph1, size)
        graphlet_counts_1 = count_graphlets(graphlets_1)
        for key, value in graphlet_counts_1.items():
            counts_1[key] += value

        graphlets_2 = extract_graphlets(graph2, size)
        graphlet_counts_2 = count_graphlets(graphlets_2)
        for key, value in graphlet_counts_2.items():
            counts_2[key] += value

    # Compute similarity between the two graphs
    similarity = graphlet_kernel_similarity(counts_1, counts_2)

    return similarity
