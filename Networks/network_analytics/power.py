import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms import community
import matplotlib.colors as mcolors
import numpy as np
from scipy.ndimage import zoom
import tifffile
import powerlaw
import os



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


def neighbors(G, identifier, directory_path):
    neighbors_count_list = []

    for node in G.nodes(): #Find how many neighbors all nodes have, return it
        neighbors_count = G.degree(node)
        neighbors_count_list.append(neighbors_count)

    save_to_excel(neighbors_count_list, f"neighbor distribution for {identifier}.xlsx", directory_path)

    return neighbors_count_list


def get_histogram_lists(neighbors):
    y_len = max(neighbors)
    y_list = list(range(1, y_len + 1))
    x_list = []

    for i in range(1, y_len + 1):
        x = neighbors.count(i)
        x_list.append(x)

    return x_list, y_list

def save_to_excel(list1, excel_filename, directory_path):
    # Create a DataFrame from the lists
    data = {'num_neighbors': list1}
    df = pd.DataFrame(data)

    # Create the full path for saving
    full_path = os.path.join(directory_path, excel_filename)

    # Save the DataFrame to an Excel file
    df.to_excel(full_path, index=False)

def histogram(counts, y_vals, identifier, directory_path):
    # Calculate the bin edges based on the y_vals
    bins = np.linspace(min(y_vals), max(y_vals), len(y_vals) + 1)

    # Create a histogram
    plt.hist(x=y_vals, bins=bins, weights=counts, edgecolor='black')

    # Adding labels and title (Optional, but recommended for clarity)
    plt.title(f'Degree Distribution of Glom-Nerve Network ({identifier})')
    plt.xlabel('Node Degree')
    plt.ylabel('Glom Count')

    file_name = f"Neighbor Distribution Histogram {identifier}"
    full_path = os.path.join(directory_path, file_name)
    plt.savefig(full_path)

    # Show the plot
    plt.show()

def ks_power(data, identifier, directory_name):

    # Fit the power-law distribution to the data
    fit = powerlaw.Fit(data)

    # Perform the Kolmogorov-Smirnov test
    ks_statistic, p_value = fit.distribution_compare('power_law', 'exponential')

    # Print the test results
    print(f"KS Statistic: {ks_statistic}")
    print(f"P-Value: {p_value}")

    # Interpret the results
    if p_value > 0.05:
        response = "The neighbor distribution is likely power-law distributed."
        print("The neigbhor distribution is likely power-law distributed.")
    else:
        response = "The neighbor distribution is likely not power-law distributed."
        print("The neighbor distribution is likely not power-law distributed.")

    with open(f'{directory_name}/{identifier} network stats.txt', 'a') as f:
        f.write(f'\npower law ks stat: {ks_statistic}; P-value: {p_value}\n{response}')
    f.close()


def power_dist(G, identifier, directory_path):

    print("Performing Neighbor Distribution Analysis for Power Law...")

    neighbor_list = neighbors(G, identifier, directory_path)

    x_vals, y_vals = get_histogram_lists(neighbor_list)
    histogram(x_vals, y_vals, identifier, directory_path)
    ks_power(neighbor_list, identifier, directory_path)

    print("Done")


if __name__ == '__main__':

    # Read the Excel file into a pandas DataFrame
    excel_file_path = input("Excel with network?: ")
    G = open_network(excel_file_path)

    neighbor_list = neighbors(G)

    x_vals, y_vals = get_histogram_lists(neighbor_list)
    histogram(x_vals, y_vals, 'test', None)
    ks_power(neighbor_list)