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
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import tkinter as tk
from tkinter import simpledialog
import smart_dilate


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

def upsample_with_padding(data, factor, original_shape):
    # Upsample the input binary array while adding padding to match an arbitrary shape

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

def binarize(image):
    """Convert an array from numerical values to boolean mask"""
    image = image != 0

    image = image.astype(np.uint8)

    return image

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

        print("Labelling node indices...")
        # Get the indices of True values in dilated_mask
        true_indices = np.argwhere(dilated_mask)

        for index in true_indices:
            i, j, k = index
            dilated_result[i, j, k].append(label)

        print(f"Node {label} processed")

        x += 1

    with open('num_nodes.txt', 'w') as f:
        f.write(f'There are {x} nodes in this image including partial')
    f.close()
        

    return dilated_result

def array_trim(edge_array, glom_array):
    """Efficiently and massively reduces extraneous search regions for edge-node intersections"""
    edge_list = edge_array.flatten() #Turn arrays into lists
    glom_list = glom_array.flatten()

    edge_bools = edge_list != 0 #establish where edges/gloms exist by converting to a boolean list
    glom_bools = glom_list != 0

    overlaps = edge_bools * glom_bools #Establish boolean list where edges and gloms intersect.

    edge_overlaps = overlaps * edge_list #Set all vals in the edges/gloms to 0 where intersections are not occurring
    glom_overlaps = overlaps * glom_list

    edge_overlaps = remove_zeros(edge_overlaps) #Remove all values where intersections are not present, so we don't have to iterate through them later
    glom_overlaps = remove_zeros(glom_overlaps)

    return edge_overlaps, glom_overlaps

def establish_connections_parallel(edge_labels, num_edge, glom_labels):
    """Looks at dilated edges array and gloms array. Iterates through edges. 
    Each edge will see what gloms it overlaps. It will put these in a list."""
    
    all_connections = []

    def process_edge(label):

        if label not in edge_labels:
            return None

        edge_connections = []

        # Get the indices corresponding to the current edge label
        indices = np.argwhere(edge_labels == label).flatten()

        for index in indices:

            edge_connections.append(glom_labels[index])

        #the set() wrapper removes duplicates from the same sublist
        my_connections = list(set(edge_connections))


        edge_connections = [[my_connections, label]]


        #Edges only interacting with one glom are not used:
        if len(my_connections) > 1:
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
        sublist = sublist[0]
        edge_ID = sublist[1]
        sublist = sublist[0]
        pairs_within_sublist = [(sublist[i], sublist[j], edge_ID) for i in range(len(sublist))
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
    # Determine the length of the input list
    length = len(pairwise_connections)
    
    # Initialize counters for column assignment
    col_start = 0
    
    # Initialize master list to store sublists
    master_list = []
    
    # Split the input list into sublists of maximum length 1 million
    while col_start < length:
        # Determine the end index for the current sublist
        col_end = min(col_start + 1000000, length)
        
        # Append the current sublist to the master list
        master_list.append(pairwise_connections[col_start:col_end])
        
        # Update column indices for the next sublist
        col_start = col_end
    
    # Create an empty DataFrame
    df = pd.DataFrame()
    
    # Assign trios to columns in the DataFrame
    for i, sublist in enumerate(master_list):
        # Determine column names for the current sublist
        column_names = ['Column {}A'.format(i+1), 'Column {}B'.format(i+1), 'Column {}C'.format(i+1)]
        
        # Create a DataFrame from the current sublist
        temp_df = pd.DataFrame(sublist, columns=column_names)
        
        # Concatenate the DataFrame with the master DataFrame
        df = pd.concat([df, temp_df], axis=1)
    
    # Save the DataFrame to an Excel file
    df.to_excel(excel_filename, index=False)

def remove_zeros(input_list):
    # Use boolean indexing to filter out zeros
    result_array = input_list[input_list != 0] #note - presumes your list is an np array

    return result_array



def get_reslice_indices(args):
    indices, dilate_xy, dilate_z, array_shape = args
    """Finds dilation extrem indices for purpose of reslicing array into subarray prior to individual-glom dilation to save mem"""
    try:
        max_indices = np.amax(indices, axis = 0) #Get the max/min of each index.
    except ValueError: #Return Nones if this error is encountered
        return None, None, None
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
    print(f"Processing node {label}")
    indices = np.argwhere(gloms == label)
    z_vals, y_vals, x_vals = get_reslice_indices((indices, dilate_xy, dilate_z, array_shape))
    if z_vals is None: #If get_reslice_indices ran into a ValueError, nothing is returned.
        return None, None, None
    sub_gloms = reslice_3d_array((gloms, z_vals, y_vals, x_vals))
    sub_edges = reslice_3d_array((edges, z_vals, y_vals, x_vals))
    return label, sub_gloms, sub_edges

def count_unique_values(label_array):
    # Flatten the 3D array to a 1D array
    flattened_array = label_array.flatten()

    non_zero = remove_zeros(flattened_array)

    non_zero = non_zero.tolist()

    # Find unique values
    unique_values = set(non_zero)

    return unique_values

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
    if label is None:
        pass
    else:
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
                        master_list.append([key1, key2, value])
        del compare_dict[key1]

    return master_list

class InputForm:
    def __init__(self, root):
        self.root = root
        self.root.title("User Input Form")

        # Create labels and entry widgets for user inputs
        self.create_label_entry("Separated Node File Name (.tif?):", "glom_name")
        self.create_label_entry("Processed Edge File Name (.tif?):", "nerve_name")
        self.create_label_entry("XY scaling of tiffs? (Float):", "xy_scale")
        self.create_label_entry("z scaling of tiffs? (Float):", "z_scale")
        self.create_label_entry("Intended Excel File Name? (.xlsx?):", "excel_name")
        self.create_label_entry("Node extra search region? (Microns - Float):", "node_search")
        self.create_label_entry("Use fast search (Y/N)? (Recommended for large images with thousands of nodes - much faster in those cases. Do not enable if above value = 0):", "fast_bool")
        self.create_label_entry("Amount to dilate edges? (Microns - Float):", "dilate_nerves")
        self.create_label_entry("Label nodes? (Do this if glom mask is binary but not if its labelled already) (Y/N):", "label_the_gloms")
        self.create_label_entry("Remove the trunk? (Y/N):", "trunk_bool")
        self.create_label_entry("Utilize inner edges? (Y/N):", "inner_bool")
        self.create_label_entry("Downsampling factor for drawing network? (Not used to calculate network but speeds up drawing label and connection overlays) (Int): ", "down_factor")

        self.submit_button = tk.Button(root, text="Submit", command=self.submit)
        self.submit_button.pack()

        self.result = None

    def create_label_entry(self, label_text, var_name):
        label = tk.Label(self.root, text=label_text)
        label.pack()
        entry = tk.Entry(self.root)
        entry.pack()
        setattr(self, f"entry_{var_name}", entry)

    def submit(self):
        glom_name = self.entry_glom_name.get()
        nerve_name = self.entry_nerve_name.get()
        xy_scale = float(self.entry_xy_scale.get())
        z_scale = float(self.entry_z_scale.get())
        excel_name = self.entry_excel_name.get()
        node_search = float(self.entry_node_search.get())
        fast_bool = self.entry_fast_bool.get()
        dilate_nerves = float(self.entry_dilate_nerves.get())
        label_the_gloms = self.entry_label_the_gloms.get()
        trunk_bool = self.entry_trunk_bool.get()
        inner_bool = self.entry_inner_bool.get()
        down_factor = int(self.entry_down_factor.get())

        self.result = {
            "glom_name": glom_name,
            "nerve_name": nerve_name,
            "xy_scale": xy_scale,
            "z_scale": z_scale,
            "excel_name": excel_name,
            "node_search": node_search,
            "fast_bool": fast_bool,
            "dilate_nerves": dilate_nerves,
            "label_the_gloms": label_the_gloms,
            "trunk_bool": trunk_bool,
            "inner_bool": inner_bool,
            "down_factor": down_factor
        }
        
        self.root.quit()

def get_user_inputs():
    root = tk.Tk()
    app = InputForm(root)
    root.mainloop()
    return app.result

def get_user_args():

    user_inputs = get_user_inputs()

    # Get user inputs
    glom_name = user_inputs["glom_name"]
    nerve_name = user_inputs["nerve_name"]
    xy_scale = user_inputs["xy_scale"]
    z_scale = user_inputs["z_scale"]
    excel_name = user_inputs["excel_name"]
    node_search = user_inputs["node_search"]
    fast_bool = user_inputs["fast_bool"]
    dilate_nerves = user_inputs["dilate_nerves"]
    label_the_gloms = user_inputs["label_the_gloms"]
    trunk_bool = user_inputs["trunk_bool"]
    inner_bool = user_inputs["inner_bool"]
    down_factor = user_inputs["down_factor"]

    print(f"Params: Node file: {glom_name}, Edge File: {nerve_name}, xy_scale: {xy_scale}, z_scale: {z_scale}, excel_name: {excel_name}, node_search_region: {node_search}, fast_search?: {fast_bool}, edge_dilation: {dilate_nerves}, node_labelling: {label_the_gloms}, trunk_removal: {trunk_bool}, inner_edges?: {inner_bool}, downsample_for_overlay_creation: {down_factor}")

    return glom_name, nerve_name, xy_scale, z_scale, excel_name, node_search, fast_bool, dilate_nerves, label_the_gloms, trunk_bool, inner_bool, down_factor

def combine_edges(edge_labels_1, edge_labels_2):
    """Combine the edges and 'inner edges' into a single array while preserving their IDs."""

    max_val = np.max(edge_labels_1) 
    edge_bools_1 = edge_labels_1 == 0 #Get boolean mask where edges do not exist.
    edge_bools_2 = edge_labels_2 > 0 #Get boolean mask where inner edges exist.
    edge_labels_2 = edge_labels_2 + max_val #Add the maximum edge ID to all inner edges so the two can be merged without overriding eachother
    edge_labels_2 = edge_labels_2 * edge_bools_2 #Eliminate any indices that should be 0 from inner edges.
    edge_labels_2 = edge_labels_2 * edge_bools_1 #Eliminate any indices where outer edges overlap inner edges (Outer edges are giving overlap priority)
    edge_labels = edge_labels_1 + edge_labels_2 #Combine the outer edges with the inner edges modified via the above steps

    return edge_labels

def setup_params(glom_name, nerve_name, xy_scale, z_scale, node_search, label_the_gloms, inner_bool):
    # Define a 3x3x3 structuring element for diagonal, horizontal, and vertical connectivity
    #This structuring element is for user
    structure_3d = np.ones((3, 3, 3), dtype=int)
    dilate_xy, dilate_z = dilation_length_to_pixels(xy_scale, z_scale, node_search, node_search)

    global twod_bool

    twod_bool = False


    gloms = tifffile.imread(glom_name)
    glom_shape = gloms.shape
    nerve = tifffile.imread(nerve_name)

    if len(glom_shape) == 2:
        twod_bool = True
        gloms = np.stack((gloms, gloms), axis=0)
        nerve = np.stack((nerve, nerve), axis=0)


    if label_the_gloms == 'N':
        print("binarizing nodes...")
        glom_labels = gloms
        gloms = binarize(gloms)
    else:
        glom_labels = None

    print("Establishing edges")
    #Dilate glom objects to use as node boundaries for edges
    dilated_gloms = dilate_3D(gloms, dilate_xy, dilate_xy, dilate_z)

    #Find edges:
    edges = establish_edges(dilated_gloms, nerve)
    if inner_bool == 'Y':

        inner_edges = establish_inner_edges(dilated_gloms, nerve) #The inner edges are the connections within dilated node search regions

    else:
        inner_edges = None

    return structure_3d, dilate_xy, dilate_z, gloms, edges, inner_edges, glom_labels

def process_edge_optional_params(edges, dilate_nerves, trunk_bool, xy_scale, z_scale):
    #Remove the nerve trunk if the user wants:
    if trunk_bool == 'Y':
        print("Snipping trunk...")
        edges = remove_trunk(edges)
        #tifffile.imwrite("Trunkless_edges.tif", edges) #Saving the trunkless volume will allow you to repeat the trunkless analysis without this step.

    if dilate_nerves > 0:
        nerve_xy, nerve_z = dilation_length_to_pixels(xy_scale, z_scale, dilate_nerves, dilate_nerves) #Dilate edges if user wants. Purpose is to reconstruct gaps in network at the cost of connective fidelity.
        dilated_edges = dilate_3D(edges, nerve_xy, nerve_xy, nerve_z)#Naturally, there is probably an optimal dilation point that balances reconstruction and fidelity.

    if dilate_nerves == 0:
        dilated_edges = dilate_3D(edges, 3, 3, 3)  # Dilate edges by one to force them to overlap stuff. This has to happen at least once so it will still happen if no dilation is desired.

    return dilated_edges

def label_edges(dilated_edges, inner_edges, structure_3d):

    # Get the labelling information before entering the parallel section.
    print("Labelling Edges...")
    edge_labels_1, num_edge_1 = ndimage.label(dilated_edges, structure=structure_3d)  # Label the edges with diagonal connectivity so they have unique ids

    if inner_edges is not None:

        edge_labels_2, num_edge_2 = ndimage.label(inner_edges, structure=structure_3d) # Label the 'inner edges'

        edge_labels = combine_edges(edge_labels_1, edge_labels_2)

        num_edge_1 = np.max(edge_labels)

        #num_edge_1 = count_unique_values(edge_labels)

        #tifffile.imwrite("labelled_edges.tif", edge_labels)

    else:
        edge_labels = edge_labels_1

    return edge_labels, num_edge_1

def label_gloms(gloms, label_the_gloms, glom_labels):
    if label_the_gloms == 'Y':
        structure_3d = np.ones((3, 3, 3), dtype=int)
        glom_labels, num_gloms = ndimage.label(gloms, structure=structure_3d)  # Label the glom objects. Note this presumes no overlap between node masks.
        #tifffile.imwrite("labelled_nodes.tif", glom_labels)
    else:
        num_gloms = int(np.max(glom_labels))
        #tifffile.imwrite("labelled_nodes.tif", glom_labels)

    print(f"There are {num_gloms} (including partial) nodes in this image")

    
    return glom_labels, num_gloms

def create_network():
    glom_name, nerve_name, xy_scale, z_scale, excel_name, node_search, fast_bool, dilate_nerves, label_the_gloms, trunk_bool, inner_bool, down_factor = get_user_args()

    structure_3d, dilate_xy, dilate_z, gloms, edges, inner_edges, glom_labels = setup_params(glom_name, nerve_name, xy_scale, z_scale, node_search, label_the_gloms, inner_bool)

    dilated_edges = process_edge_optional_params(edges, dilate_nerves, trunk_bool, xy_scale, z_scale)

    del edges

    edge_labels, num_edge_1 = label_edges(dilated_edges, inner_edges, structure_3d)

    del dilated_edges
    del inner_edges

    glom_labels, num_gloms = label_gloms(gloms, label_the_gloms, glom_labels)

    del gloms

    print("Processing Edge Connections")

    if fast_bool == 'Y':
        glom_labels = smart_dilate.smart_dilate(glom_labels, dilate_xy, dilate_z)
        node_search = 0

    if node_search == 0:
        edge_labels, trim_glom_labels = array_trim(edge_labels, glom_labels)
        connections_parallel = establish_connections_parallel(edge_labels, num_edge_1, trim_glom_labels)
        del edge_labels
        connections_parallel = extract_pairwise_connections(connections_parallel)

    else:
        connections_parallel = create_glom_dictionary(glom_labels, edge_labels, num_gloms, dilate_xy, dilate_z) #Find which edges connect which nodes and put them in a dictionary.
        del edge_labels
        connections_parallel = find_shared_value_pairs(connections_parallel) #Sort through the dictionary to find connected node pairs.

    print("Done")
    print("Trimming lists for excel...")
    create_and_save_dataframe(connections_parallel, excel_name) #Save to excel
    print("Excel output saved")

    return glom_labels, num_gloms, excel_name, down_factor


def draw_network(glom_labels, num_gloms, excel_name, down_factor):

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

    def get_centroids(gloms):
        centroid_dict = {}
        unique_gloms = gloms.flatten()
        unique_gloms = remove_zeros(unique_gloms)
        unique_gloms = set(list(unique_gloms))
        for glom in unique_gloms:
            centroid = compute_centroid(gloms, glom)
            if centroid is None:
                continue
            else:
                centroid_dict[glom] = centroid

        return centroid_dict

    glom_labels = downsample(glom_labels, down_factor) #Downsample labelled gloms for below processing

    centroid_dict = get_centroids(glom_labels)

    #The below scripts are optional but will run after the network is generated by default.
    glom_draw.draw_from_centroids(glom_labels, num_gloms, centroid_dict, twod_bool) #This script draws the labelled glom IDs onto a downsampled tiff image. In something like Imaris, they can be overlayed for easy identification of nodes.

    network_draw.draw_network_from_centroids(glom_labels, excel_name, centroid_dict, twod_bool) #This script draws the connections between nodes into a downsampled tiff image. Downsampling is used to speed this up.
    #Please note the method "upsample_with_padding()" can be called to return a downsampled array to an arbitrary size.

def create_and_draw_network():
    glom_labels, num_gloms, excel_name, down_factor = create_network()
    draw_network(glom_labels, num_gloms, excel_name, down_factor)



if __name__ == "__main__":
    create_and_draw_network()