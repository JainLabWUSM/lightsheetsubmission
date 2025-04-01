from . import nettracer
from . import network_analysis
import numpy as np
from scipy.ndimage import zoom
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
import tifffile
import pandas as pd

def get_reslice_indices(args):
    """Internal method used for the secondary algorithm that finds dimensions for subarrays around nodes"""

    indices, dilate_xy, dilate_z, array_shape = args
    try:
        max_indices = np.amax(indices, axis = 0) #Get the max/min of each index.
    except ValueError: #Return Nones if this error is encountered
        return None, None, None
    min_indices = np.amin(indices, axis = 0)

    #This parallelized calculation was used to evaluate approximate neural innervation surrounding 3D glomeruli. This block specifically picks out a small subarray around each glomeruli that contains its future dilation space, to chop up the array and dilate the small arrays in parallel to save RAM. There is one analysis where I restrict this to 'vascular poles' instead - we previously established that vascular poles in kidney are oriented away from the medulla. Therefore this calculation for vascular poles was approximated by clipping only the upper half of the glomerulus by editing the lines below specifically: For example, we can set y_max to min_indices[1] + ((max_indices[1] - min_indices[1]))/2 to avoid including the upper half of the current subarray. Note that this does correspond with the vascular pole orientation in the image. The origin in tifs and numpy arrays is in the upper corner, meaning the y axis is 0 at the 'top' and some larger n at the 'bottom'. My images were typically captured with the cortex physically oriented away from the earth, so this means in the array, the vascular poles were oriented towards the '0' on the y axis, and away from larger value.
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
    """Internal method used for the secondary algorithm to reslice subarrays around nodes."""

    input_array, z_range, y_range, x_range = args
    z_start, z_end = z_range
    z_start, z_end = int(z_start), int(z_end)
    y_start, y_end = y_range
    y_start, y_end = int(y_start), int(y_end)
    x_start, x_end = x_range
    x_start, x_end = int(x_start), int(x_end)
    
    # Reslice the array
    resliced_array = input_array[z_start:z_end+1, y_start:y_end+1, x_start:x_end+1]
    
    return resliced_array



def _get_node_edge_dict(label_array, edge_array, label, dilate_xy, dilate_z, cores = 0):
    """Internal method used for the secondary algorithm to find pixel involvement of nodes around an edge."""
    
    # Create a boolean mask where elements with the specified label are True
    label_array = label_array == label
    dil_array = nettracer.dilate_3D(label_array, dilate_xy, dilate_xy, dilate_z) #Dilate the label to see where the dilated label overlaps

    if cores == 0: #For getting the volume of objects. Cores presumes you want the 'core' included in the interaction.
        edge_array = edge_array * dil_array  # Filter the edges by the label in question
        label_array = np.count_nonzero(dil_array)
        edge_array = np.count_nonzero(edge_array) # For getting the interacting skeleton

    elif cores == 1: #Cores being 1 presumes you do not want to 'core' included in the interaction
        label_array = dil_array - label_array
        edge_array = edge_array * label_array
        label_array = np.count_nonzero(label_array)
        edge_array = np.count_nonzero(edge_array) # For getting the interacting skeleton

    elif cores == 2: #Presumes you want skeleton within the core but to only 'count' the stuff around the core for volumes... because of imaging artifacts, perhaps
        edge_array = edge_array * dil_array
        label_array = dil_array - label_array
        label_array = np.count_nonzero(label_array)
        edge_array = np.count_nonzero(edge_array) # For getting the interacting skeleton


    
    args = [edge_array, label_array]

    return args

def process_label(args):
    """Internal method used for the secondary algorithm to process a particular node."""
    nodes, edges, label, dilate_xy, dilate_z, array_shape = args
    indices = np.argwhere(nodes == label)
    z_vals, y_vals, x_vals = get_reslice_indices((indices, dilate_xy, dilate_z, array_shape))
    if z_vals is None: #If get_reslice_indices ran into a ValueError, nothing is returned.
        return None, None, None
    sub_nodes = reslice_3d_array((nodes, z_vals, y_vals, x_vals))
    sub_edges = reslice_3d_array((edges, z_vals, y_vals, x_vals))
    return label, sub_nodes, sub_edges


def create_node_dictionary(nodes, edges, num_nodes, dilate_xy, dilate_z, cores = 0):
    """Internal method used for the secondary algorithm to process nodes in parallel."""
    # Initialize the dictionary to be returned
    node_dict = {}

    array_shape = nodes.shape

    # Use ThreadPoolExecutor for parallel execution
    with ThreadPoolExecutor(max_workers=mp.cpu_count()) as executor:
        # First parallel section to process labels
        # List of arguments for each parallel task
        args_list = [(nodes, edges, i, dilate_xy, dilate_z, array_shape) for i in range(1, num_nodes + 1)]

        # Execute parallel tasks to process labels
        results = executor.map(process_label, args_list)

        # Second parallel section to create dictionary entries
        for label, sub_nodes, sub_edges in results:
            executor.submit(create_dict_entry, node_dict, label, sub_nodes, sub_edges, dilate_xy, dilate_z, cores)

    return node_dict

def create_dict_entry(node_dict, label, sub_nodes, sub_edges, dilate_xy, dilate_z, cores = 0):
    """Internal method used for the secondary algorithm to pass around args in parallel."""

    if label is None:
        pass
    else:
        node_dict[label] = _get_node_edge_dict(sub_nodes, sub_edges, label, dilate_xy, dilate_z, cores = cores)


def quantify_edge_node(nodes, edges, search = 0, xy_scale = 1, z_scale = 1, cores = 0, resize = None, directory = ''):

    def save_dubval_dict(dict, index_name, val1name, val2name, filename):

        #index name goes on the left, valname on the right
        df = pd.DataFrame.from_dict(dict, orient='index', columns=[val1name, val2name])

        # Rename the index to 'Node ID'
        df.index.name = index_name

        # Save DataFrame to Excel file
        df.to_excel(filename, engine='openpyxl')

    if type(nodes) is str:
        nodes = tifffile.imread(nodes)

    if type(edges) is str:
        edges = tifffile.imread(edges)


    edges = nettracer.skeletonize(edges)

    if len(np.unique(nodes)) == 2:
        nodes, num_nodes = nettracer.label_objects(nodes)
    else:
        num_nodes = np.max(nodes)

    if resize is not None:
        edges = zoom(edges, resize)
        nodes = zoom(nodes, resize)
        edges = nettracer.skeletonize(edges)

    if search > 0:
        dilate_xy, dilate_z = nettracer.dilation_length_to_pixels(xy_scale, z_scale, search, search)
    else:
        dilate_xy, dilate_z = 0, 0


    edge_quants = create_node_dictionary(nodes, edges, num_nodes, dilate_xy, dilate_z, cores = cores) #Find which edges connect which nodes and put them in a dictionary.
    
    save_dubval_dict(edge_quants, 'NodeID', 'Edge Skele Quantity', 'Search Region Volume', f'{directory}/edge_node_quantity.xlsx')