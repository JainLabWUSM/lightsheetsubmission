import tifffile
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.ndimage import zoom

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

def remove_dupes(pair1, pair2):
    # Combine pairs into a set of tuples for faster membership check
    pairwise_set = set(zip(pair1, pair2))

    # Initialize sets to store unique pairs and their reversed forms
    unique_pairs = set()
    reversed_pairs = set()

    # Iterate through the pairs, adding them to unique_pairs if not already present,
    # or to reversed_pairs if they're in reversed order
    for pair in pairwise_set:
        if pair not in unique_pairs and pair[::-1] not in reversed_pairs:
            unique_pairs.add(pair)
            reversed_pairs.add(pair[::-1])

    # Unpack the unique pairs into separate lists
    pair1_unique, pair2_unique = zip(*unique_pairs)

    return pair1_unique, pair2_unique

def compute_centroid(binary_stack, label):
    """
    Finds centroid of labelled object in array
    """
    indices = np.argwhere(binary_stack == label)
    if indices.shape[0] == 0:
        return None
    else:
        centroid = np.round(np.mean(indices, axis=0)).astype(int)
        return centroid

def draw_line_inplace(start, end, array):
    """
    Draws a white line between two points in a 3D array.
    """
    
    # Calculate the distances between start and end coordinates
    distances = end - start

    # Determine the number of steps along the line
    num_steps = int(max(np.abs(distances)) + 1)

    # Generate linearly spaced coordinates along each dimension
    x_coords = np.linspace(start[0], end[0], num_steps, endpoint=True).round().astype(int)
    y_coords = np.linspace(start[1], end[1], num_steps, endpoint=True).round().astype(int)
    z_coords = np.linspace(start[2], end[2], num_steps, endpoint=True).round().astype(int)

    # Clip coordinates to ensure they are within the valid range
    x_coords = np.clip(x_coords, 0, array.shape[0] - 1)
    y_coords = np.clip(y_coords, 0, array.shape[1] - 1)
    z_coords = np.clip(z_coords, 0, array.shape[2] - 1)

    # Set the line coordinates to 255 in the existing array
    array[x_coords, y_coords, z_coords] = 255

def downsample(data, factor):
    # Downsample the input data by a specified factor
    return zoom(data, 1/factor, order=0)

def draw_network(gloms, network):
    network = read_excel_to_lists(network)
    pair1 = network[0]
    pair2 = network[1]
    centroid_dic = {}
    print("removing duplicates")
    pair1, pair2 = remove_dupes(pair1, pair2)
    pair1 = list(pair1)
    pair2 = list(pair2)
    network = set(pair1 + pair2)
    print(network)
    print("Finding centroids")
    for item in network:
        centroid = compute_centroid(gloms, item)
        if centroid is not None:
            centroid_dic[item] = centroid
    output_stack = np.zeros(np.shape(gloms), dtype=np.uint8)

    for i, pair1_val in enumerate(pair1):
        print(f"Drawing line for pair {i}...")
        pair2_val = pair2[i]
        try:
            pair1_centroid = centroid_dic[pair1_val]
            pair2_centroid = centroid_dic[pair2_val]
            draw_line_inplace(pair1_centroid, pair2_centroid, output_stack)
        except KeyError:
            print("Missing centroid")
            pass

    tifffile.imwrite("drawn_network.tif", output_stack)
    print("done")

def draw_network_from_centroids(gloms, network, centroids, twod_bool):

    print("Drawing network")
    network = read_excel_to_lists(network)
    pair1 = network[0]
    pair2 = network[1]
    centroid_dic = {}
    print("removing duplicates")
    pair1, pair2 = remove_dupes(pair1, pair2)
    pair1 = list(pair1)
    pair2 = list(pair2)
    network = set(pair1 + pair2)
    print(network)
    print("Finding centroids")
    for item in network:
        centroid = centroids[item]
        centroid_dic[item] = centroid
    output_stack = np.zeros(np.shape(gloms), dtype=np.uint8)

    for i, pair1_val in enumerate(pair1):
        print(f"Drawing line for pair {i}...")
        pair2_val = pair2[i]
        try:
            pair1_centroid = centroid_dic[pair1_val]
            pair2_centroid = centroid_dic[pair2_val]
            draw_line_inplace(pair1_centroid, pair2_centroid, output_stack)
        except KeyError:
            print("Missing centroid")
            pass

    if twod_bool:
        output_stack = output_stack[0,:,:]

    tifffile.imwrite("drawn_network.tif", output_stack)
    print("done")

if __name__ == '__main__':

    gloms = input("glom file?: ")
    while True:
    	Q = input("Label gloms (Y/N)? Label if they are binary. Do not label if they already have grayscale labels: ")
    	if Q == 'Y' or Q == 'N':
    		break
    network = input("excel with glom network info?: ")

    gloms = tifffile.imread(gloms)

    glom_shape = gloms.shape



    if Q == 'Y':
    	print("labelling gloms...")
    	gloms, num_gloms = ndimage.label(gloms)

    #gloms = downsample(gloms, 10)


    network = read_excel_to_lists(network)
    pair1 = network[0]
    pair2 = network[1]
    centroid_dic = {}
    print("removing duplicates")
    pair1, pair2 = remove_dupes(pair1, pair2)
    pair1 = list(pair1)
    pair2 = list(pair2)
    network = set(pair1 + pair2)
    print(network)
    print("Finding centroids")
    for item in network:
        centroid = compute_centroid(gloms, item)
        if centroid is not None:
            centroid_dic[item] = centroid
    output_stack = np.zeros(np.shape(gloms), dtype=np.uint8)

    for i, pair1_val in enumerate(pair1):
        print(f"Drawing line for pair {i}...")
        pair2_val = pair2[i]
        try:
            pair1_centroid = centroid_dic[pair1_val]
            pair2_centroid = centroid_dic[pair2_val]
            draw_line_inplace(pair1_centroid, pair2_centroid, output_stack)
        except KeyError:
            print("Missing centroid")
            pass

    if len(glom_shape) == 2:
        output_stack = output_stack[0,:,:]


    tifffile.imwrite("drawn_network.tif", output_stack)
    print("done")


