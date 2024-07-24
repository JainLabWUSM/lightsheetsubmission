import pandas as pd
import numpy as np
import tifffile
from scipy import ndimage
from skimage import measure
import cv2
import concurrent.futures
from scipy.ndimage import zoom
import glom_draw
import network_draw
from skimage.morphology import skeletonize_3d

#Most updated version

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




def dilate_3D(tiff_array, dilated_x, dilated_y, dilated_z):
    """Dilate an array in 3D.
    Arguments are an array,  and the desired pixel dilation amounts in X, Y, Z."""

    # Create empty arrays to store the dilated results for the XY and XZ planes
    dilated_xy = np.zeros_like(tiff_array, dtype=np.uint8)
    dilated_xz = np.zeros_like(tiff_array, dtype=np.uint8)

    # Perform 2D dilation in the XY plane
    for z in range(tiff_array.shape[0]):
        kernel_x = int(dilated_x)
        kernel_y = int(dilated_y)
        kernel = np.ones((kernel_y, kernel_x), dtype=np.uint8)

        dilated_slice = cv2.dilate(tiff_array[z], kernel, iterations=1)
        dilated_xy[z] = dilated_slice

    # Perform 2D dilation in the XZ plane
    for y in range(tiff_array.shape[1]):
        kernel_x = int(dilated_x)
        kernel_z = int(dilated_z)
        kernel = np.ones((kernel_z, kernel_x), dtype=np.uint8)

        dilated_slice = cv2.dilate(tiff_array[:, y, :], kernel, iterations=1)
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

    print(f"dilating in x by {dilate_xy} pixels")
    print(f"dilating in z by {dilate_z} pixels")

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


def establish_connections_parallel(edge_labels, num_edge, glom_labels):
    """Looks at dilated edges array and gloms array. Iterates through edges. 
    Each edge will see what gloms it overlaps. It will put these in a list."""
    
    all_connections = []

    def process_edge(label):
        edge_connections = []

        # Get the indices corresponding to the current edge label
        indices = np.argwhere(edge_labels == label).flatten()

        for index in indices:

            edge_connections.append(glom_labels[index])

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

        # Add the pairs to the final list.
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

def count_unique_values(label_array):
    # Flatten the 3D array to a 1D array
    flattened_array = label_array.flatten()

    # Find unique values
    unique_values = np.unique(flattened_array)

    # Get the total number of unique values
    total_unique_values = len(unique_values)

    return total_unique_values

def labels_to_boolean(label_array, labels_list):
    # Use np.isin to create a boolean array with a single operation
    boolean_array = np.isin(label_array, labels_list)
    
    return boolean_array

def remove_zeros(input_list):
    # Use boolean indexing to filter out zeros
    result_array = input_list[input_list != 0] #note - presumes your list is an np array

    return result_array

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



# Define a 3x3x3 structuring element for diagonal, horizontal, and vertical connectivity
#This structuring element is for user
structure_3d = np.ones((3, 3, 3), dtype=int)

#Obtain info from user. Namely, what files will be used and tiff voxel scales:
glom_masks = input("Processed Glom File? (3D Watershed first): ")
nerve_name = input("Processed Nerve File?: ")
excel_name = input("Excel File Name? (Will place in this directory): ")
while True:
    dilate_nerves = input("Dilate nerves at all? (Y/N): ")
    if dilate_nerves == 'Y' or dilate_nerves == 'N':
        if dilate_nerves == 'Y':
            nerve_xy = float(input("Microns to dilate nerves?: "))
            xy_scale = float(input("xy scaling?: "))
            z_scale = float(input("z scaling?: "))
        break

trunk_bool = trunk_remove_bool()


#Convert tiffs to numpy arrays:

masks = tifffile.imread(glom_masks)
gloms = masks > 0
masks = masks.astype(np.uint16) #save mem cause watershed output usually 32 bit for no reason
nerve = tifffile.imread(nerve_name)

print("Establishing edges")

if dilate_nerves == 'Y':
    dilate_xy, dilate_z = dilation_length_to_pixels(xy_scale, z_scale, nerve_xy, nerve_xy)
    nerve = dilate_3D(nerve, dilate_xy, dilate_xy, dilate_z)
    nerve = skeletonize_3d(nerve)

edges = establish_edges(gloms, nerve)
del nerve
#Remove the nerve trunk if the user wants:
if trunk_bool:
    print("Snipping trunk...")
    edges = remove_trunk(edges)





dilated_edges = dilate_3D(edges, 3, 3, 3)  # Dilate edges by one to force them to overlap stuff.
print("Done")
del edges

print("labelling edges")

# Get the labelling information before entering the parallel section.
edge_labels_1, num_edge_1 = ndimage.label(dilated_edges, structure=structure_3d)  # Diagonal connectivity
del dilated_edges


num_gloms = count_unique_values(masks)
print(f"There are {num_gloms} (including partial) gloms in this image (use this number for reference for disconnected gloms)")
with open('num_gloms.txt', 'w') as f:
    f.write(f'There are {num_gloms} gloms in this image including partial')
f.close()

print("finding glom-nerve intersection points")
edge_labels_1, trim_masks = array_trim(edge_labels_1, masks) #Remove extraneous overlap. Note this command flattens the numpy arrays to 1D.

#get network connections:
print("Processing Edge Connections")
connections_parallel = establish_connections_parallel(edge_labels_1, num_edge_1, trim_masks)
print("Done")
print("Trimming lists for excel...")
pairwise_connections_parallel = extract_pairwise_connections(connections_parallel)
create_and_save_dataframe(pairwise_connections_parallel, excel_name)
print("Excel output saved")

print("Drawing glom labels")
#Create label array for gloms, for easy reference
small_labels = downsample(masks, 5)

glom_draw.draw_gloms(small_labels, num_gloms)

print("Drawing network (Requires excel file to still be there)...")
network_draw.draw_network(small_labels, excel_name)