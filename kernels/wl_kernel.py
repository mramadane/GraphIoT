from collections import defaultdict
import numpy as np


def wl_iteration(graph):
    """
    Perform one iteration of the Weisfeiler-Lehman algorithm on the given graph.
    The graph is modified in-place, with new labels assigned to each node.
    """
    # Create a mapping from multiset strings to new labels
    new_label_dict = defaultdict(lambda: len(new_label_dict))

    # Store the new labels for each node here (to avoid changing the graph while iterating over it)
    new_labels = {}

    # For each node, create a multiset of labels and assign a new label
    for node in graph.nodes:
        # Get the current label of the node
        current_label = graph.nodes[node]['label']

        # Create a multiset of labels that includes the node's own label and the labels of its neighbors
        neighbor_labels = [graph.nodes[neighbor]['label'] for neighbor in graph.neighbors(node)]
        multiset = sorted([str(current_label)] + [str(label) for label in neighbor_labels])

        # Convert the multiset to a string and assign a new label based on this string
        multiset_str = ','.join(multiset)
        new_labels[node] = new_label_dict[multiset_str]

    # Update the labels in the graph
    for node, new_label in new_labels.items():
        graph.nodes[node]['label'] = new_label


def wl_subtree_kernel(graph1, graph2, num_iterations=3):
    """
    Compute the Weisfeiler-Lehman subtree kernel between two graphs.
    """
    # Make a deep copy of the graphs to avoid modifying the original graphs
    g1 = graph1.copy()
    g2 = graph2.copy()

    # Perform WL iterations
    for _ in range(num_iterations):
        wl_iteration(g1)
        wl_iteration(g2)

    # Compute label histograms for both graphs
    label_histogram_1 = defaultdict(int)
    label_histogram_2 = defaultdict(int)
    for node in g1.nodes:
        label = g1.nodes[node]['label']
        label_histogram_1[label] += 1
    for node in g2.nodes:
        label = g2.nodes[node]['label']
        label_histogram_2[label] += 1

    # Compute the dot product of the label histograms
    kernel_value = sum(label_histogram_1[label] * label_histogram_2[label] for label in
                       set(label_histogram_1) & set(label_histogram_2))

    return kernel_value

