from networkx.algorithms import community
from community import community_louvain
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import random
import modularity


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

def create_and_save_dataframe(pairwise_connections, excel_filename):
    """This method is just for saving output to excel"""
    # Create a DataFrame from the list of pairwise connections
    df = pd.DataFrame(pairwise_connections, columns=['Column A', 'Column B'])

    # Save the DataFrame to an Excel file
    df.to_excel(excel_filename, index=False)

    return df

def list_trim(list1, list2, component):

    list1_copy = list1
    indices_to_delete = []
    for i in range(len(list1)):

        if list1[i] not in component and list2[i] not in component:
            indices_to_delete.append(i)

    for i in reversed(indices_to_delete):
        del list1_copy[i]

    return list1_copy

def louvain_mod_20x(G):

    connected_components = list(nx.connected_components(G))

    for i, component in enumerate(connected_components):
        # Apply the Louvain community detection on the subgraph
        partition = community_louvain.best_partition(G.subgraph(component))

        
        # Calculate modularity
        modularity = community_louvain.modularity(partition, G.subgraph(component))

    return modularity

def rand_net(num_rows, nodes, identifier, directory_path):
    running_mod = 0
    for i in range(100):
        random_network = []

        for i in range(0, num_rows):
            random_partner = random.randint(0, len(nodes)-1)
            random_partner = nodes[random_partner]
            if random_partner == nodes[i]:
                while random_partner == nodes[i]:
                    random_partner = random.randint(0, len(nodes)-1)
                    random_partner = nodes[random_partner]
            random_pair = [nodes[i], random_partner]
            random_network.append(random_pair)

        df = create_and_save_dataframe(random_network, "output.xlsx")

        

        G, edge_weights = weighted_network("output.xlsx")

        modularity = louvain_mod_20x(G)

        running_mod += modularity

    running_mod = running_mod/100

    return running_mod



if __name__ == "__main__":
    identifier = "output"
    directory_path = 'output'
    modularity.create_directory(directory_path)
    excel_name = input("excel file?: ")

    G, edge_weights = weighted_network(excel_name)

    print("Original large component modularity: ")

    modularity.louvain_mod_20x(G, identifier, directory_path)

    nodes = max(nx.connected_components(G), key=len)

    master_list = read_excel_to_lists(excel_name)

    column1_list = master_list[0]
    column2_list = master_list[1]
    nodes = list(nodes)
    iter_list = list_trim(column1_list, column2_list, nodes)

    num_rows = len(iter_list)

    while len(nodes) < num_rows:
        nodes = nodes + nodes

print(f"\nRandom equivalent largest component modularity (avged over 100 iterations): ")

running_mod = rand_net(num_rows, nodes, identifier, directory_path)

print(running_mod)
