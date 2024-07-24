import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms import community
import matplotlib.colors as mcolors
import numpy as np
from scipy.ndimage import zoom
import tifffile
import radial
import os
import power
import modularity
from scipy import ndimage

def open_network(excel_file_path):
    """opens an unweighted network from the network excel file"""

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


def create_directory(directory_name):
    try:
        os.mkdir(directory_name)
        print(f"Directory '{directory_name}' created successfully.")
    except FileExistsError:
        pass

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

def get_res_bool():
    res_bool = input("Is this a 5x network? (Y/N): ")
    if res_bool == 'Y':
        res_bool = True
    elif res_bool == 'N':
        res_bool = False
    else:
        get_res_bool()

    return res_bool




gloms = input("glom file?:" )
excel_file_path = input("Excel with network?: ")
xy_scale = float(input("xy scale?: "))
z_scale = float(input("z scale?: "))
identifier = input("Output name? (will be used to label graphs and name files): ")
res_bool = get_res_bool()
gloms = tifffile.imread(gloms)
G, edge_weights = weighted_network(excel_file_path)
G0 = open_network(excel_file_path)
directory_name = f"Network Stats and Graphs {identifier}"
create_directory(directory_name)

number_of_nodes = G.number_of_nodes()
edge_node_avg = G.number_of_edges()/number_of_nodes
print(f"There are {edge_node_avg} edges per node in the network")
with open(f'{directory_name}/{identifier} network stats.txt', 'w') as f:
    f.write(f'Number of nodes: {number_of_nodes}\nAverage edge/node = {edge_node_avg}')
f.close()


if not res_bool:
    print("labelling gloms...")
    gloms, num_gloms = ndimage.label(gloms) #For testing volumes that aren't prelabelled

power.power_dist(G0, identifier, directory_name)
radial.radial_analysis(G0, gloms, xy_scale, z_scale, identifier, directory_name)


if res_bool:
    modularity.louvain_analysis_5x(G, excel_file_path, identifier, directory_name)
else:
    modularity.louvain_analysis_20x(G, excel_file_path, identifier, directory_name)

