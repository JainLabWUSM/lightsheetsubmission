import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms import community
import matplotlib.colors as mcolors

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
excel_file_path = input("Excel file?: ")

master_list = read_excel_to_lists(excel_file_path)

edges = zip(master_list[0], master_list[1])

#df = pd.read_excel(excel_file_path)

# Create a graph
G = nx.Graph()

# Add edges from the DataFrame
#edges = df.values  # Assuming the columns are named "Node1" and "Node2"
G.add_edges_from(edges)

# Print basic information about the graph
print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())

# Calculate the average degree connectivity
#average_degree_connectivity = nx.average_degree_connectivity(G)
#print("Average degree connectivity:", average_degree_connectivity)

# Calculate the average number of edges attached to a node
#average_edges_per_node = sum(k * v for k, v in average_degree_connectivity.items()) / G.number_of_nodes()
#print("Average edges per node:", average_edges_per_node)

# Visualize the graph with different edge colors for each community
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, font_color='red', font_weight='bold', node_size=10)

#connected_components = list(nx.connected_components(G))
#for i, component in enumerate(connected_components):
    #communities = community.label_propagation_communities(G.subgraph(component))
    
    # Assign a different color to each community
    #colors = [mcolors.to_hex(plt.cm.tab10(i / len(connected_components))[:3]) for _ in range(len(component))]
    
    #nx.draw_networkx_edges(G, pos, edgelist=G.subgraph(component).edges(), edge_color=colors)

    #num_nodes = len(component)
    #modularity = community.modularity(G.subgraph(component), communities)
    #print(f"Modularity for component with {num_nodes} nodes:", modularity)

plt.show()