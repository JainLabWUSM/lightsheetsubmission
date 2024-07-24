import pandas as pd
import networkx as nx
import tifffile
import numpy as np
from networkx.algorithms import community

def labels_to_boolean(label_array, labels_list):
    # Use np.isin to create a boolean array with a single operation
    boolean_array = np.isin(label_array, labels_list)
    return boolean_array

def open_network(excel_file_path):
    """opens an unweighted network from the network excel file"""

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

    # Read the Excel file into a pandas DataFrame
    master_list = read_excel_to_lists(excel_file_path)

    # Create a graph
    G = nx.Graph()

    nodes_a = master_list[0]
    nodes_b = master_list[1]

    # Add edges to the graph
    for i in range(len(nodes_a)):
        G.add_edge(nodes_a[i], nodes_b[i])

    return G



# Read the Excel file into a pandas DataFrame
excel_file_path = input("Excel file?: ")
masks = input("watershedded, dilated glom mask?: ")
specific_node_label = int(input("node key?: "))
masks = tifffile.imread(masks)
masks = masks.astype(np.uint16)

G = open_network(excel_file_path)

# Get the connected component containing the specific node label
connected_component = nx.node_connected_component(G, specific_node_label)

# Convert the set of nodes to a list
nodes_in_component = list(connected_component)

print("Nodes in the connected component containing the specific node label:", nodes_in_component)

mask2 = labels_to_boolean(masks, nodes_in_component)

# Convert boolean values to 0 and 255
mask2 = mask2.astype(np.uint8) * 255

tifffile.imsave("connected_component_containing_specific_node.tif", mask2)