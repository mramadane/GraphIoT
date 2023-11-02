import plotly.graph_objects as go
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
def plot_interactive_graph(G, layout_method='spring'):
    # Determine the layout based on the given method
    if layout_method == 'spring':
        pos = nx.spring_layout(G)
    elif layout_method == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(G)
    elif layout_method == 'circular':
        pos = nx.circular_layout(G)
    else:
        raise ValueError("Invalid layout method")

    # Assign colors to nodes based on their types
    node_types = list(set(nx.get_node_attributes(G, 'Type').values()))
    node_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
                   '#17becf']
    node_color_map = dict(zip(node_types, node_colors[:len(node_types)]))

    # Node positions
    x_vals = [pos[k][0] for k in G.nodes()]
    y_vals = [pos[k][1] for k in G.nodes()]

    # Node descriptions for hover functionality
    node_hover = ['<br>'.join([f'{key}: {value}' for key, value in data.items() if key != 'Type'])
                  for node, data in G.nodes(data=True)]

    # Create edge traces
    edge_x = []
    edge_y = []
    edge_hover_texts = []

    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

        edge_data = edge[2]
        edge_text = '<br>'.join([f'{key}: {value}' for key, value in edge_data.items()])
        edge_hover_texts.append(edge_text)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        mode='lines',
        line=dict(width=0.5, color='grey'),
        hoverinfo='text',
        hovertext=edge_hover_texts
    )

    # Create node trace
    node_trace = go.Scatter(
        x=x_vals, y=y_vals,
        mode='markers+text',
        hoverinfo='text',
        text=[data['Type'] for _, data in G.nodes(data=True)],
        textposition="top center",
        hovertext=node_hover,
        marker=dict(
            color=[node_color_map[data['Type']] for _, data in G.nodes(data=True)],
            size=10,
            line_width=2),
        showlegend=True
    )

    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='Network Graph made with Python',
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=0, l=0, r=0, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )

    fig.show()


def plot_mds(kernel_matrix: np.ndarray, is_similarity: bool = True):
    """
    Apply MDS and plot the results based on a kernel (similarity/dissimilarity) matrix.

    :param kernel_matrix: A numpy array representing the kernel matrix.
    :param is_similarity: A boolean indicating whether the kernel matrix represents similarities or dissimilarities.
    :return: MDS coordinates
    """
    num_graphs = kernel_matrix.shape[0]

    # Convert similarity to dissimilarity if necessary
    dissimilarities = np.max(kernel_matrix) - kernel_matrix if is_similarity else kernel_matrix

    # Apply MDS
    mds = MDS(n_components=2, dissimilarity='precomputed')
    mds_coordinates = mds.fit_transform(dissimilarities)

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.scatter(mds_coordinates[:, 0], mds_coordinates[:, 1], c='blue')
    for i in range(num_graphs):
        plt.text(mds_coordinates[i, 0], mds_coordinates[i, 1], f'Graph {i + 1}', fontsize=12, ha='right', va='bottom')
    plt.title('MDS Plot')
    plt.xlabel('MDS Dimension 1')
    plt.ylabel('MDS Dimension 2')
    plt.grid(True)
    plt.show()

    return mds_coordinates