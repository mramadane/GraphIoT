from collections import defaultdict
import numpy as np
def wlek_iteration(graph):
    """
    Perform one iteration of the Weisfeiler-Lehman Edge Kernel algorithm on the given graph.
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

        # Create a multiset of labels that includes the node's own label, the labels of its neighbors,
        # and the labels of the edges connecting to those neighbors
        neighbor_labels = []
        for neighbor in graph.neighbors(node):
            edge_label = graph[node][neighbor]['label'] if 'label' in graph[node][neighbor] else ''
            neighbor_labels.append((graph.nodes[neighbor]['label'], edge_label))
        multiset = sorted(
            [(str(current_label), '')] + [(str(label), str(edge_label)) for label, edge_label in neighbor_labels])

        # Convert the multiset to a string and assign a new label based on this string
        multiset_str = ','.join([f'{label}-{edge_label}' for label, edge_label in multiset])
        new_labels[node] = new_label_dict[multiset_str]

    # Update the labels in the graph
    for node, new_label in new_labels.items():
        graph.nodes[node]['label'] = new_label


def wlek_subtree_kernel(graph1, graph2, num_iterations=10):
    """
    Compute the Weisfeiler-Lehman Edge Kernel between two graphs.
    """
    # Make a deep copy of the graphs to avoid modifying the original graphs
    g1 = graph1.copy()
    g2 = graph2.copy()

    # Initialization: Assign initial labels to nodes based on their types, and to edges
    for graph in [g1, g2]:
        for node, data in graph.nodes(data=True):
            graph.nodes[node]['label'] = data['Type']
        for edge in graph.edges:
            if 'label' not in graph.edges[edge]:
                graph.edges[edge]['label'] = ''

    # Perform WLEK iterations
    for _ in range(num_iterations):
        wlek_iteration(g1)
        wlek_iteration(g2)

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

