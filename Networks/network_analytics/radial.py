import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms import community
import matplotlib.colors as mcolors
import numpy as np
from scipy.ndimage import zoom
import tifffile
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


def save_figure(directory_path, identifier):
	file_name = f"Radial Analysis Histogram {identifier}"
	full_path = os.path.join(directory_path, file_name)
	plt.savefig(full_path)


def downsample(data, factor):
    # Downsample the input data by a specified factor
    return zoom(data, 1/factor, order=0)



def get_distance_list(centroids, G, xy_scale, z_scale):
	print("Generating radial distribution...")
	length = len(centroids) #find total len to iterate through

	distance_list = [] #init empty list to contain all distance vals

	for idx, centroid in enumerate(centroids, start=1): #iterate through all gloms
		# Check if idx is in G.nodes
		if idx not in G.nodes:
			continue  # Skip to the next iteration of the outer loop
		z1, y1, x1 = centroid.astype(float)
		z1, y1, x1 = z1 * z_scale, y1 * xy_scale, x1 * xy_scale
		connected_component = list(G.neighbors(idx))

		for i in range(len(connected_component)): #find distances from current gloms to other gloms
			z2, y2, x2 = centroids[i].astype(float)
			z2, y2, x2 = z2 * z_scale, y2 * xy_scale, x2 * xy_scale
			dist = np.sqrt((z2 - z1)**2 + (y2 - y1)**2 + (x2 - x1)**2)

			if dist > 0:
				distance_list.append(dist)

	return distance_list



def get_centroids(array):
	print("Finding centroids for radial distribution analysis...")
	max_index = int(np.max(array))
	# Find centroids
	centroids = np.array([np.mean(np.argwhere(array == i), axis=0) for i in range(1, max_index+1)])

	print("Done.")
	return centroids

def save_to_excel(list1, list2, excel_filename, directory_path):
	# Create a DataFrame from the lists
	data = {'rad_dist': list2, 'num_gloms': list1}
	df = pd.DataFrame(data)

	# Create the full path for saving
	full_path = os.path.join(directory_path, excel_filename)

	# Save the DataFrame to an Excel file
	df.to_excel(full_path, index=False)


def buckets(dists, num_objects, rad_dist, identifier, directory_path):
	y_vals = []
	x_vals = []
	radius = 0
	max_dist = max(dists)

	while radius < max_dist:
		radius2 = radius + rad_dist
		radial_objs = 0
		for item in dists:
			if item >= radius and item <= radius2:
				radial_objs += 1
		radial_avg = radial_objs/num_objects
		radius = radius2
		x_vals.append(radial_avg)
		y_vals.append(radius)

	save_to_excel(x_vals, y_vals, f'radial_distances {identifier}.xlsx', directory_path)

	return x_vals, y_vals

def histogram(counts, y_vals, identifier, directory_path):
	# Calculate the bin edges based on the y_vals
	bins = np.linspace(min(y_vals), max(y_vals), len(y_vals) + 1)

	# Create a histogram
	plt.hist(x=y_vals, bins=bins, weights=counts, edgecolor='black')

	# Adding labels and title (Optional, but recommended for clarity)
	plt.title(f'Radial Distribution of Glom-Nerve Network ({identifier})')
	plt.xlabel('Distance From Glom (µm)')
	plt.ylabel('Avg Number of Neigbhoring Vertices')

	file_name = f"Radial Analysis Histogram {identifier}"
	full_path = os.path.join(directory_path, file_name)
	plt.savefig(full_path)

	# Show the plot
	plt.show()

def radial_analysis(G, gloms, xy_scale, z_scale, identifier, directory_path):

	print("Performing Radial Distribution Analysis...")

	down_factor = 5
	rad_dist = 150
	xy_small = xy_scale * 5
	z_small = z_scale * 5
	gloms = downsample(gloms, down_factor)
	num_objects = np.max(gloms)
	centroids = get_centroids(gloms)
	dist_list = get_distance_list(centroids, G, xy_small, z_small)
	x_vals, y_vals = buckets(dist_list, num_objects, rad_dist, identifier, directory_path)
	histogram(x_vals, y_vals, identifier, directory_path)


if __name__ == '__main__':

	down_factor = 5
	rad_dist = 150
	# Read the Excel file into a pandas DataFrame
	excel_file_path = input("Excel with network?: ")
	G = open_network(excel_file_path)

	gloms = input("glom file?:" )
	gloms = tifffile.imread(gloms)
	xy_scale = float(input("xy scale?: "))
	xy_small = xy_scale * 5
	z_scale = float(input("z scale?: "))
	z_small = z_scale * 5
	gloms = downsample(gloms, down_factor)
	num_objects = np.max(gloms)
	centroids = get_centroids(gloms)
	dist_list = get_distance_list(centroids, G, xy_small, z_small)
	x_vals, y_vals = buckets(dist_list, num_objects, rad_dist)
	histogram(x_vals, y_vals)