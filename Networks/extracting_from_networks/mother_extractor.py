import pandas as pd
import networkx as nx
import tifffile
import numpy as np
from networkx.algorithms import community
from community import community_louvain
import glom_draw
from scipy.ndimage import zoom

def labels_to_boolean(label_array, labels_list):
    # Use np.isin to create a boolean array with a single operation
    boolean_array = np.isin(label_array, labels_list)
    
    return boolean_array

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

def compute_centroid(binary_stack, label):
    """
    Finds centroid of labelled object in array
    """
    indices = np.argwhere(binary_stack == label)
    centroid = np.round(np.mean(indices, axis=0)).astype(int)

    return centroid


def get_border_nodes(partition):
# Find nodes that border nodes in other communities
    border_nodes = set()
    for edge in G.edges():
        if partition[edge[0]] != partition[edge[1]]:
            border_nodes.add(edge[0])
            border_nodes.add(edge[1])
    print("Border nodes:", border_nodes)

    return border_nodes

def downsample(data, factor):
    # Downsample the input data by a specified factor
    return zoom(data, 1/factor, order=0)

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


if __name__ == '__main__':

    # Read the Excel file into a pandas DataFrame
    excel_file_path = input("Excel with network?: ")
    gloms = input("Labelled gloms tiff (corresponding to network?: ")

    gloms = tifffile.imread(gloms)

    G, edge_weights = weighted_network(excel_file_path)
    # Apply the Louvain algorithm for community detection
    partition = community_louvain.best_partition(G)


    my_nodes = get_border_nodes(partition)

    mother_nodes = list(my_nodes)

    smalls = downsample(gloms, 5)

    mother_dict = {}
    centroid_dic = {}

    for item in mother_nodes:
        centroid = compute_centroid(smalls, item)
        centroid_dic[item] = centroid

    print(centroid_dic)

    for node in mother_nodes:
        mother_dict[node] = G.degree(node)

    print(mother_dict)

    print("Mother nodes (prediction):", mother_nodes)

    mask2 = labels_to_boolean(gloms, mother_nodes)

    smalls = labels_to_boolean(smalls, mother_nodes)

    # Convert boolean values to 0 and 255
    mask2 = mask2.astype(np.uint8) * 255

    small_labels = glom_draw.mother_draw(mother_dict, centroid_dic, smalls)


    tifffile.imwrite("small_mother_labels.tif", small_labels)



    tifffile.imwrite("isolated_community.tif", mask2)

    large_labels = upsample_with_padding(small_labels, 5, gloms.shape)

    tifffile.imwrite("large_mother_labels.tif", large_labels)