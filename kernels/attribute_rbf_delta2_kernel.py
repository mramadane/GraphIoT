import numpy as np


def rbf_kernel(x: float, y: float, sigma: float = 1.0) -> float:
    """
    Compute the RBF kernel between two scalars.

    :param x: First scalar
    :param y: Second scalar
    :param sigma: Parameter for the RBF kernel
    :return: RBF kernel value
    """
    return np.exp(-((x - y) ** 2) / (2 * sigma ** 2))


def categorical_kernel(x: str, y: str) -> int:
    """
    Compute the kernel between two categorical values.

    :param x: First categorical value
    :param y: Second categorical value
    :return: 1 if x and y are the same, 0 otherwise
    """
    return 1 if x == y else 0


def compute_node_kernel(node1: Dict[str, Union[float, str]], node2: Dict[str, Union[float, str]]) -> float:
    """
    Compute the kernel between two nodes based on their attributes.

    :param node1: Attributes of the first node
    :param node2: Attributes of the second node
    :return: Kernel value between the two nodes
    """
    kernel_value = 1.0

    # Continuous features and their corresponding sigma values for RBF kernel
    continuous_features = {
        'Power': 1.0,
        'Frequency': 1.0,
        'Size': 1.0,
        'SensR': 1.0,
        'Res': 1.0,
        'Lin': 1.0,
        'Capacity': 1.0,
        'Voltage': 1.0,
        'Speed': 1.0,
        'CO2 Footprint': 1.0,
        'Bit Width': 1.0,
    }

    # Compute kernel for continuous features
    for feature, sigma in continuous_features.items():
        if feature in node1 and feature in node2:
            kernel_value *= rbf_kernel(node1[feature], node2[feature], sigma)

    # Categorical features
    categorical_features = ['Subtype', 'Generation']

    # Compute kernel for categorical features
    for feature in categorical_features:
        if feature in node1 and feature in node2:
            kernel_value *= categorical_kernel(node1[feature], node2[feature])

    return kernel_value


def compute_graph_kernel(graph1: nx.Graph, graph2: nx.Graph) -> float:
    """
    Compute the graph kernel between two graphs.

    :param graph1: First graph
    :param graph2: Second graph
    :return: Graph kernel value
    """
    kernel_value = 0.0

    for node1_id, node1_attributes in graph1.nodes(data=True):
        for node2_id, node2_attributes in graph2.nodes(data=True):
            # Compare only nodes of the same type and subtype
            if node1_attributes['Type'] == node2_attributes['Type']:
                kernel_value += compute_node_kernel(node1_attributes, node2_attributes)

    return kernel_value


