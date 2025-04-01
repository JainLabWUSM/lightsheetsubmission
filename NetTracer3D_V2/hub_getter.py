import networkx as nx
import pandas as pd
import numpy as np
import tifffile
from scipy import ndimage
from . import network_analysis
from . import node_draw

def upsample_with_padding(data, factor, original_shape):
    # Upsample the input binary array while adding padding to match the original shape

    # Get the dimensions of the original and upsampled arrays
    original_shape = np.array(original_shape)
    binary_array = zoom(data, factor, order=0)
    upsampled_shape = np.array(binary_array.shape)

    # Calculate the positive differences in dimensions
    difference_dims = original_shape - upsampled_shape

    # Calculate the padding amounts for each dimension
    padding_dims = np.maximum(difference_dims, 0)
    padding_before = padding_dims // 2
    padding_after = padding_dims - padding_before

    # Pad the binary array along each dimension
    padded_array = np.pad(binary_array, [(padding_before[0], padding_after[0]),
                                         (padding_before[1], padding_after[1]),
                                         (padding_before[2], padding_after[2])], mode='constant', constant_values=0)

    # Calculate the subtraction amounts for each dimension
    sub_dims = np.maximum(-difference_dims, 0)
    sub_before = sub_dims // 2
    sub_after = sub_dims - sub_before

    # Remove planes from the beginning and end
    if sub_dims[0] == 0:
        trimmed_planes = padded_array
    else:
        trimmed_planes = padded_array[sub_before[0]:-sub_after[0], :, :]

    # Remove rows from the beginning and end
    if sub_dims[1] == 0:
        trimmed_rows = trimmed_planes
    else:
        trimmed_rows = trimmed_planes[:, sub_before[1]:-sub_after[1], :]

    # Remove columns from the beginning and end
    if sub_dims[2] == 0:
        trimmed_array = trimmed_rows
    else:
        trimmed_array = trimmed_rows[:, :, sub_before[2]:-sub_after[2]]

    return trimmed_array


def weighted_network(excel_file_path):
    """creates a network where the edges have weights proportional to the number of connections they make between the same structure"""
    # Read the Excel file into a pandas DataFrame
    master_list = read_excel_to_lists(excel_file_path)

    # Create a graph
    G = nx.Graph()

    # Create a dictionary to store edge weights based on node pairs
    edge_weights = {}

    nodes_a = master_list[0]
    nodes_b = master_list[1]

    # Iterate over the DataFrame rows and update edge weights
    for i in range(len(nodes_a)):
        node1, node2 = nodes_a[i], nodes_b[i]
        edge = (node1, node2) if node1 < node2 else (node2, node1)  # Ensure consistent order
        edge_weights[edge] = edge_weights.get(edge, 0) + 1

    # Add edges to the graph with weights
    for edge, weight in edge_weights.items():
        G.add_edge(edge[0], edge[1], weight=weight)

    return G, edge_weights

def read_excel_to_lists(file_path, sheet_name=0):
    """Convert a pd dataframe to lists"""
    # Read the Excel file into a DataFrame without headers
    df = pd.read_excel(file_path, header=None, sheet_name=sheet_name)

    df = df.drop(0)

    # Initialize an empty list to store the lists of values
    data_lists = []

    # Iterate over each column in the DataFrame
    for column_name, column_data in df.items():
        # Convert the column values to a list and append to the data_lists
        data_lists.append(column_data.tolist())

    master_list = [[], [], []]


    for i in range(0, len(data_lists), 3):

        master_list[0].extend(data_lists[i])
        master_list[1].extend(data_lists[i+1])

        try:
            master_list[2].extend(data_lists[i+2])
        except IndexError:
            pass

    return master_list

def labels_to_boolean(label_array, labels_list):
    # Use np.isin to create a boolean array with a single operation
    boolean_array = np.isin(label_array, labels_list)
    
    return boolean_array

def get_hubs(nodepath, network, proportion = None, directory = None, centroids = None):

    if type(nodepath) == str:
        nodepath = tifffile.imread(nodepath)

        if len(np.unique(nodepath)) < 3:
        
            structure_3d = np.ones((3, 3, 3), dtype=int)
            nodepath, num_nodes = ndimage.label(nodepath, structure=structure_3d)

    if type(network) == str:
        G, weights = weighted_network(network)
    else:
        G = network

    if proportion is None:
        proportion = 0.10
        print("Isolating top 0.10 high degree nodes by default. Specify 'proportion = 0.x' for custom node isolation.")
    else:
        print(f"Isolating top {proportion} high degree nodes")


    node_list = list(G.nodes)
    node_dict = {}

    for node in node_list:
        node_dict[node] = (G.degree(node))

    # Calculate the number of top proportion% entries
    num_items = len(node_dict)
    num_top_10_percent = max(1, int(num_items * proportion))  # Ensure at least one item

    # Sort the dictionary by values in descending order and get the top 10%
    sorted_items = sorted(node_dict.items(), key=lambda item: item[1], reverse=True)
    top_10_percent_items = sorted_items[:num_top_10_percent]

    # Extract the keys from the top proportion% items
    top_10_percent_keys = [key for key, value in top_10_percent_items]

    masks = labels_to_boolean(nodepath, top_10_percent_keys)

    masks = masks * nodepath #Makes it save with labels

    # Convert boolean values to 0 and 255
    #masks = masks.astype(np.uint8) * 255

    if directory is None:

        tifffile.imwrite("isolated_hubs.tif", masks)
        print(f"Isolated hubs saved to isolated_hubs.tif")


    else:

        tifffile.imwrite(f"{directory}/isolated_hubs.tif", masks)
        print(f"Isolated hubs saved to {directory}/isolated_hubs.tif")

    if centroids is None:
        for item in nodepath.shape:
            if item < 5:
                down_factor = 1
                break
            else:
                down_factor = 5

        centroids = network_analysis._find_centroids(masks, top_10_percent_keys, down_factor = down_factor)

    degree_dict = {}

    for node in top_10_percent_keys:
        degree_dict[node] = G.degree(node)

    labels = node_draw.degree_draw(degree_dict, centroids, masks)

    if directory is None:

        tifffile.imwrite("hub_degree_labels.tif", labels)
        print("Node hub labels saved to hub_degree_labels.tif")

    else:
        tifffile.imwrite(f"{directory}/hub_degree_labels.tif", labels)
        print(f"Node hub labels saved to {directory}/hub_degree_labels.tif")

    masks = node_draw.degree_infect(degree_dict, masks)

    if directory is None:

        tifffile.imwrite("hub_degree_labels_grayscale.tif", masks)
        print(f"Node hub grayscale labels saved to hub_degree_labels_grayscale.tif")


    else:
        tifffile.imwrite(f"{directory}/hub_degree_labels_grayscale.tif", masks)
        print(f"Node hub grayscale labels saved to {directory}/hub_degree_labels_grayscale.tif")

    return top_10_percent_keys



if __name__ == "__main__":

    masks = input("Labelled nodes?: ")
    outer_net = input('outer edges?: ')

    masks = tifffile.imread(masks)
    outer_G, weights = weighted_network(outer_net)


    node_list = list(outer_G.nodes)
    node_dict = {}

    for node in node_list:
        node_dict[node] = (outer_G.degree(node))

    # Calculate the number of top 10% entries
    num_items = len(node_dict)
    num_top_10_percent = max(1, int(num_items * 0.10))  # Ensure at least one item

    # Sort the dictionary by values in descending order and get the top 10%
    sorted_items = sorted(node_dict.items(), key=lambda item: item[1], reverse=True)
    top_10_percent_items = sorted_items[:num_top_10_percent]

    # Extract the keys from the top 10% items
    top_10_percent_keys = [key for key, value in top_10_percent_items]

    mask2 = labels_to_boolean(masks, top_10_percent_keys)

    # Convert boolean values to 0 and 255
    mask2 = mask2.astype(np.uint8) * 255

    tifffile.imwrite("isolated_vertices.tif", mask2)