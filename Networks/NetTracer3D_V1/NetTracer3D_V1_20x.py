import pandas as pd
import numpy as np
import tifffile
from scipy import ndimage
from skimage import measure
import cv2
import concurrent.futures
from scipy.ndimage import zoom
import glom_draw
from skimage.morphology import skeletonize_3d
import network_draw
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor


#THIS IS THE CURRENT MAIN SCRIPT FOR THIS ENDEAVOR

def invert_array(array):
    """Used to flip glom array indices. 0 becomes 255 and vice versa."""
    inverted_array = np.where(array == 0, 255, 0).astype(np.uint8)
    return inverted_array

def establish_edges(gloms, nerve):
    """Used to black out where nerves interact with gloms"""
    invert_gloms = invert_array(gloms)
    edges = nerve * invert_gloms
    return edges

def establish_inner_edges(gloms, nerve):
    """Returns any edges that may exist betwixt dilated gloms."""
    inner_edges = nerve * gloms
    return inner_edges

def downsample(data, factor):
    # Downsample the input data by a specified factor
    return zoom(data, 1/factor, order=0)

def trunk_remove_bool():
    """
    Establishes a boolean of whether user wants to remove the nerve trunk
    """
    user_string = input("Remove nerve trunk? (Y/N) (check if there is one in vol): ")
    if user_string == "Y":
        return True
    elif user_string == "N":
        return False
    else:
        trunk_remove_bool()

def remove_trunk(edges):
    """
    Used to remove the nerve trunk. Essentially removes the largest object from
    a binary array.
    """
     # Label connected components in the binary array
    labeled_array = measure.label(edges)

    # Get unique labels and their counts
    unique_labels, label_counts = np.unique(labeled_array, return_counts=True)

    # Find the label corresponding to the largest object
    largest_label = unique_labels[np.argmax(label_counts[1:]) + 1]

    # Set indices of the largest object to 0
    edges[labeled_array == largest_label] = 0

    return edges

def binarize(image):
    """Convert an array from numerical values to boolean mask"""
    image = image != 0 

    return image


def dilate_3D(tiff_array, dilated_x, dilated_y, dilated_z):
    """Dilate an array in 3D. Consider replacing with scipy dilation method.
    Arguments are an array,  and the desired pixel dilation amounts in X, Y, Z."""

    # Create empty arrays to store the dilated results for the XY and XZ planes
    dilated_xy = np.zeros_like(tiff_array, dtype=np.uint8)
    dilated_xz = np.zeros_like(tiff_array, dtype=np.uint8)

    # Perform 2D dilation in the XY plane
    for z in range(tiff_array.shape[0]):
        kernel_x = int(dilated_x)
        kernel_y = int(dilated_y)
        kernel = np.ones((kernel_y, kernel_x), dtype=np.uint8)

        # Convert the slice to the appropriate data type
        tiff_slice = tiff_array[z].astype(np.uint8)

        dilated_slice = cv2.dilate(tiff_slice, kernel, iterations=1)
        dilated_xy[z] = dilated_slice

    # Perform 2D dilation in the XZ plane
    for y in range(tiff_array.shape[1]):
        kernel_x = int(dilated_x)
        kernel_z = int(dilated_z)
        kernel = np.ones((kernel_z, kernel_x), dtype=np.uint8)

        # Convert the slice to the appropriate data type
        tiff_slice = tiff_array[:, y, :].astype(np.uint8)

        dilated_slice = cv2.dilate(tiff_slice, kernel, iterations=1)
        dilated_xz[:, y, :] = dilated_slice

    # Overlay the results (you can use logical OR operation or another method)
    final_result = dilated_xy | dilated_xz

    return final_result

def dilation_length_to_pixels(xy_scaling, z_scaling, micronx, micronz):
    """Find XY and Z dilation parameters based on voxel micron scaling"""
    dilate_xy = 2 * int(round(micronx/xy_scaling))

    # Ensure the dilation param is odd to have a center pixel
    dilate_xy += 1 if dilate_xy % 2 == 0 else 0

    dilate_z = 2 * int(round(micronz/z_scaling))

    # Ensure the dilation param is odd to have a center pixel
    dilate_z += 1 if dilate_z % 2 == 0 else 0

    return dilate_xy, dilate_z


def create_structuring_element(dilate_xy, dilate_z):    
    # Create a cubic structuring element
    structuring_element = np.ones((dilate_z, dilate_xy, dilate_xy), dtype=bool)
    
    return structuring_element

def listwise_dilation(labeled_array, structuring_element):
    """Dilates labelled objects as lists, preserving identities in overlapping regions"""
    unique_labels = np.unique(labeled_array)
    dilated_result = np.zeros_like(labeled_array, dtype=object)

    x = 0

    print("Initializing empty list array")
    #Initialize dilated_result array to contain all empty lists
    for i in range(dilated_result.shape[0]):
        for j in range(dilated_result.shape[1]):
            for k in range(dilated_result.shape[2]):
                dilated_result[i, j, k] = []


    for label in unique_labels:
        #Skip empty label
        if label == 0:
            continue

        #Establish boolean array for label
        mask = (labeled_array == label)
        #Dilate boolean label
        dilated_mask = ndimage.binary_dilation(mask, structure = structuring_element)

        print("Labelling glom indices...")
        # Get the indices of True values in dilated_mask
        true_indices = np.argwhere(dilated_mask)

        for index in true_indices:
            i, j, k = index
            dilated_result[i, j, k].append(label)

        print(f"Glom {label} processed")

        x += 1

    with open('num_gloms.txt', 'w') as f:
        f.write(f'There are {x} gloms in this image including partial')
    f.close()
        

    return dilated_result


def establish_connections_parallel(gloms, dilated_edges, structuring_element, edge_labels, num_edge, glom_labels, num_gloms, list_gloms):
    """Looks at dilated edges array and gloms array. Iterates through edges. 
    Each edge will see what gloms it overlaps. It will put these in a list."""
    
    all_connections = []

    glom_labels = list_gloms

    def process_edge(label):
        edge_connections = []

        # Get the indices corresponding to the current edge label
        indices = np.argwhere(edge_labels == label)

        for index in indices:
            # Retrieve Z, Y, X coordinates from the index
            z, y, x = index

            # Check if the corresponding value in glom_labels is nonzero
            if glom_labels[z, y, x]:
                for value in glom_labels[z, y, x]:
                    edge_connections.append(value)

        #the set() wrapper removes duplicates from the same sublist
        edge_connections = list(set(edge_connections))

        #Edges only interacting with one glom are not used:
        if len(edge_connections) > 1:
            print(f"Edge processed.")
            return edge_connections
        else:
            return None

    #These lines makes CPU run for loop iterations simultaneously, speeding up the program:
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(process_edge, range(1, num_edge + 1)))

    all_connections = [result for result in results if result is not None]

    return all_connections


def extract_pairwise_connections(connections):
    """This method takes the list of edge interactions from the above method(s) and breaks it down into pairs"""
    pairwise_connections = []

    # Iterate through each sublist in the connections list
    for sublist in connections:
        # Generate all unique pairs within the sublist
        pairs_within_sublist = [(sublist[i], sublist[j]) for i in range(len(sublist))
                                for j in range(i + 1, len(sublist))]

        # Add the unique pairs to the final list, ignoring duplicates. Remove the set() wrapper if you want duplicates for whatever reason.
        pairwise_connections.extend(set(map(tuple, pairs_within_sublist)))

    # Convert the final list to a list of lists
    pairwise_connections = [list(pair) for pair in pairwise_connections]

    #The commented out code below can be enabled to optionally make the returned list
    #have no duplicated paired connections. Make sure to delete the existing the return statement.
    
    #isolated_pairs = []

    #def reverse_pairs(pair):
        #flip = [pair[1], pair[0]]
        #return flip

    #for thing in pairwise_connections:
        #if thing not in isolated_pairs and reverse_pairs(thing) not in isolated_pairs:
            #isolated_pairs.append(thing)


    #return isolated_pairs


    return pairwise_connections

def create_and_save_dataframe(pairwise_connections, excel_filename):
    """This method is just for saving output to excel"""
    # Create a DataFrame from the list of pairwise connections
    df = pd.DataFrame(pairwise_connections, columns=['Column A', 'Column B'])

    # Save the DataFrame to an Excel file
    df.to_excel(excel_filename, index=False)

def remove_zeros(input_list):
    # Use boolean indexing to filter out zeros
    result_array = input_list[input_list != 0] #note - presumes your list is an np array

    return result_array



def get_reslice_indices(args):
    indices, dilate_xy, dilate_z, array_shape = args
    """Finds dilation extrem indices for purpose of reslicing array into subarray prior to individual-glom dilation to save mem"""
    max_indices = np.amax(indices, axis = 0) #Get the max/min of each index. 
    min_indices = np.amin(indices, axis = 0)

    z_max, y_max, x_max = max_indices[0], max_indices[1], max_indices[2]

    z_min, y_min, x_min = min_indices[0], min_indices[1], min_indices[2]

    y_max = y_max + ((dilate_xy-1)/2) + 1 #Establish dimensions of intended subarray, expanding the max/min indices to include
    y_min = y_min - ((dilate_xy-1)/2) - 1 #the future dilation space (by adding/subtracting half the dilation kernel for each axis)
    x_max = x_max + ((dilate_xy-1)/2) + 1 #an additional index is added in each direction to make sure nothing is discluded.
    x_min = x_min - ((dilate_xy-1)/2) - 1
    z_max = z_max + ((dilate_z-1)/2) + 1
    z_min = z_min - ((dilate_z-1)/2) - 1

    if y_max > (array_shape[1] - 1): #Some if statements to make sure the subarray will not cause an indexerror
        y_max = (array_shape[1] - 1)
    if x_max > (array_shape[2] - 1):
        x_max = (array_shape[2] - 1)
    if z_max > (array_shape[0] - 1):
        z_max = (array_shape[0] - 1)
    if y_min < 0:
        y_min = 0
    if x_min < 0:
        x_min = 0
    if z_min < 0:
        z_min = 0

    y_vals = [y_min, y_max] #Return the subarray dimensions as lists
    x_vals = [x_min, x_max]
    z_vals = [z_min, z_max]

    return z_vals, y_vals, x_vals

def reslice_3d_array(args):
    input_array, z_range, y_range, x_range = args
    """Reslice a 3D array by specified range"""
    z_start, z_end = z_range
    z_start, z_end = int(z_start), int(z_end)
    y_start, y_end = y_range
    y_start, y_end = int(y_start), int(y_end)
    x_start, x_end = x_range
    x_start, x_end = int(x_start), int(x_end)
    
    # Reslice the array
    resliced_array = input_array[z_start:z_end+1, y_start:y_end+1, x_start:x_end+1]
    
    return resliced_array



def _get_glom_edge_dict(label_array, edge_array, label, dilate_xy, dilate_z):

    
    # Create a boolean mask where elements with the specified label are True
    label_array = label_array == label
    label_array = dilate_3D(label_array, dilate_xy, dilate_xy, dilate_z) #Dilate the label to see where the dilated label overlaps
    edge_array = edge_array * label_array  # Filter the edges by the label in question
    edge_array = edge_array.flatten()  # Convert 3d array to 1d array
    edge_array = remove_zeros(edge_array)  # Remove zeros
    edge_array = set(edge_array)  # Remove duplicates
    edge_array = list(edge_array)  # Back to list
    
    return edge_array

def process_label(args):
    gloms, edges, label, dilate_xy, dilate_z, array_shape = args
    #print(f"Processing glom {label}")
    indices = np.argwhere(gloms == label)
    z_vals, y_vals, x_vals = get_reslice_indices((indices, dilate_xy, dilate_z, array_shape))
    print(f"{label}: {z_vals}")
    sub_gloms = reslice_3d_array((gloms, z_vals, y_vals, x_vals))
    sub_edges = reslice_3d_array((edges, z_vals, y_vals, x_vals))
    return label, sub_gloms, sub_edges

def create_glom_dictionary(gloms, edges, num_gloms, dilate_xy, dilate_z):
    """Create a dictionary of overlaps between the dilated labels in the array 'gloms' and those in the array 'edges'"""
    # Initialize the dictionary to be returned
    glom_dict = {}

    array_shape = gloms.shape

    # Use ThreadPoolExecutor for parallel execution
    with ThreadPoolExecutor(max_workers=mp.cpu_count()) as executor:
        # First parallel section to process labels
        # List of arguments for each parallel task
        args_list = [(gloms, edges, i, dilate_xy, dilate_z, array_shape) for i in range(1, num_gloms + 1)]

        # Execute parallel tasks to process labels
        results = executor.map(process_label, args_list)

        # Second parallel section to create dictionary entries
        for label, sub_gloms, sub_edges in results:
            executor.submit(create_dict_entry, glom_dict, label, sub_gloms, sub_edges, dilate_xy, dilate_z)

    return glom_dict

def create_dict_entry(glom_dict, label, sub_gloms, sub_edges, dilate_xy, dilate_z):
    """Create entry in the dictionary"""
    glom_dict[label] = _get_glom_edge_dict(sub_gloms, sub_edges, label, dilate_xy, dilate_z)

def find_shared_value_pairs(input_dict):
    master_list = []
    compare_dict = input_dict.copy()

    # Iterate through each key in the dictionary
    for key1, values1 in input_dict.items():
        # Iterate through each other key in the dictionary
        for key2, values2 in compare_dict.items():
            # Avoid comparing the same key to itself
            if key1 != key2:
                # Find the intersection of values between the two keys
                shared_values = set(values1) & set(values2)
                # If there are shared values, create pairs and add to master list
                if shared_values:
                    for value in shared_values:
                        master_list.append([key1, key2])
        del compare_dict[key1]

    return master_list

if __name__ == "__main__":
    # Define a 3x3x3 structuring element for diagonal, horizontal, and vertical connectivity
    #This structuring element is for user
    structure_3d = np.ones((3, 3, 3), dtype=int)

    #Obtain info from user. Namely, what files will be used and tiff voxel scales:
    glom_name = input("Seperated Glom File?: ")
    nerve_name = input("Processed Nerve File?: ")
    xy_scale = float(input("XY scaling of tiffs? (will use 10 micron dilation by default): "))
    z_scale = float(input("z scaling of tiffs?: "))
    excel_name = input("Excel File Name? (Will place in this directory): ")
    node_search = float(input("Node extra search region? (Microns): "))
    while True:
        dilate_nerves = input("Dilate nerves at all? (Y/N): ")
        if dilate_nerves == 'Y' or dilate_nerves == 'N':
            if dilate_nerves == 'Y':
                nerve_xy = float(input("Microns to dilate nerves?: "))
            break
    trunk_bool = trunk_remove_bool()

    down_xy = 5 * xy_scale
    down_z = 5 * z_scale
    down_xy, down_z = dilation_length_to_pixels(down_xy, down_z, node_search, node_search)


    #Convert voxel scaling to actual pixels to dilate (note this script presumes you will dilate by 10 um)
    dilate_xy, dilate_z = dilation_length_to_pixels(xy_scale, z_scale, node_search, node_search)


    #Convert tiffs to numpy arrays:
    gloms = tifffile.imread(glom_name)
    nerve = tifffile.imread(nerve_name)

    print("Establishing edges")

    #Dilate glom objects to use as node boundaries for edges
    dilated_gloms = dilate_3D(gloms, dilate_xy, dilate_xy, dilate_z)

    #Find edges:
    edges = establish_edges(dilated_gloms, nerve)
    tifffile.imwrite("outer_edges.tif", edges)

    if dilate_nerves == 'Y':
        nerve_xy, nerve_z = dilation_length_to_pixels(xy_scale, z_scale, nerve_xy, nerve_xy)
        edges = dilate_3D(edges, nerve_xy, nerve_xy, nerve_z)



    #Remove the nerve trunk if the user wants:
    if trunk_bool:
        print("Snipping trunk...")
        edges = remove_trunk(edges)

    inner_edges = establish_inner_edges(dilated_gloms, nerve)
    tifffile.imwrite("inner_edges.tif", inner_edges)

    edges = input("outer edges?: ")
    inner_edges = input("inner edges?: ")
    edges = tifffile.imread(edges)
    inner_edges = tifffile.imread(inner_edges)

    dilated_edges = dilate_3D(edges, 3, 3, 3)  # Dilate edges by one to force them to overlap stuff. Not sure if this is better to do before or after the labelling...
    
    print("Done")

    tifffile.imwrite("dilated_edges.tif", dilated_edges)

    # Get the labelling information before entering the parallel section.
    print("Labelling Edges...")
    edge_labels_1, num_edge_1 = ndimage.label(dilated_edges, structure=structure_3d)  # Diagonal connectivity
    edge_labels_2, num_edge_2 = ndimage.label(inner_edges, structure=structure_3d)
    max_val = np.max(edge_labels_1)
    edge_bools_1 = edge_labels_1 == 0
    edge_bools_2 = edge_labels_2 > 0
    edge_labels_2 = edge_labels_2 + max_val
    edge_labels_2 = edge_labels_2 * edge_bools_2
    edge_labels_2 = edge_labels_2 * edge_bools_1
    edge_labels = edge_labels_1 + edge_labels_2

    del edge_labels_1
    del edge_bools_1
    del edge_bools_2
    del edge_labels_2
    print("Done")





    glom_labels, num_gloms = ndimage.label(gloms)  # Does not use diagonal connectivity
    print(f"There are {num_gloms} (including partial) gloms in this image (use this number for reference for disconnected gloms)")
    glom_labels = downsample(glom_labels, 5)


    #There are a lot of args fed into the below method. Not sure if all of them are even necessary, but I wouldn't mess with it...
    print("Processing Edge Connections")

    connections_parallel = create_glom_dictionary(glom_labels, edge_labels, num_gloms, down_xy, down_z)
    connections_parallel = find_shared_value_pairs(connections_parallel)

    print("Done")
    print("Trimming lists for excel...")
    create_and_save_dataframe(connections_parallel, excel_name)
    print("Excel output saved")

    print("Drawing glom labels...")

    #This is another script I wrote. The program will not run if it is not present in the same directory. But all this does is give a tiff file with Glom numerical IDs.
    glom_draw.draw_gloms(glom_labels, num_gloms)
    print("Drawing network (Requires excel file to still be there)...")
    network_draw.draw_network(glom_labels, excel_name)