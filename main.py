
import pickle

import networkx as nx
import numpy as np
from utils.viz import plot_interactive_graph
from kernels.shortestPath_kernel import apply_shortest_path_graphlet_kernel


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    fnames = []
    graphs=[]
    for i in range(1,10):
        fnames.append(f'C:/Users/thami/Desktop/Graphs_IOT/pgraph{i}.pkl')
    for name in fnames:
        with open(name, 'rb') as f:
            graphs.append(pickle.load(f))
    print(graphs)
    num_graphs = len(graphs)
    kernel_matrix = np.zeros((num_graphs, num_graphs))
    for graph in graphs:
        for node, data in graph.nodes(data=True):
            graph.nodes[node]['label'] = data['Type']
    for i in range(num_graphs):
        for j in range(i, num_graphs):
            kernel_value = apply_shortest_path_graphlet_kernel(graphs[i], graphs[j])
            kernel_matrix[i, j] = kernel_value
            kernel_matrix[j, i] = kernel_value  # The matrix is symmetric

    print(kernel_matrix)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
