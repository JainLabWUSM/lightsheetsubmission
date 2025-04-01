import tifffile
import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt
from scipy.ndimage import gaussian_filter
from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2
import os
import math
import re
from . import nettracer
import multiprocessing as mp
from skimage.feature import peak_local_max
try:
    import cupy as cp
    import cupyx.scipy.ndimage as cpx
    from cupyx.scipy.ndimage import maximum_filter
except:
    pass


def dilate_3D(tiff_array, dilated_x, dilated_y, dilated_z):
    """Internal method to dilate an array in 3D.
    Arguments are an array,  and the desired pixel dilation amounts in X, Y, Z."""

    def create_circular_kernel(diameter):
        """Create a 2D circular kernel with a given radius.

        Parameters:
        radius (int or float): The radius of the circle.

        Returns:
        numpy.ndarray: A 2D numpy array representing the circular kernel.
        """
        # Determine the size of the kernel
        radius = diameter/2
        size = radius  # Diameter of the circle
        size = int(np.ceil(size))  # Ensure size is an integer
        
        # Create a grid of (x, y) coordinates
        y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
        
        # Calculate the distance from the center (0,0)
        distance = np.sqrt(x**2 + y**2)
        
        # Create the circular kernel: points within the radius are 1, others are 0
        kernel = distance <= radius
        
        # Convert the boolean array to integer (0 and 1)
        return kernel.astype(np.uint8)

    def create_ellipsoidal_kernel(long_axis, short_axis):
        """Create a 2D ellipsoidal kernel with specified axis lengths and orientation.

        Parameters:
        long_axis (int or float): The length of the long axis.
        short_axis (int or float): The length of the short axis.

        Returns:
        numpy.ndarray: A 2D numpy array representing the ellipsoidal kernel.
        """
        semi_major, semi_minor = long_axis / 2, short_axis / 2

        # Determine the size of the kernel

        size_y = int(np.ceil(semi_minor))
        size_x = int(np.ceil(semi_major))
        
        # Create a grid of (x, y) coordinates centered at (0,0)
        y, x = np.ogrid[-semi_minor:semi_minor+1, -semi_major:semi_major+1]
        
        # Ellipsoid equation: (x/a)^2 + (y/b)^2 <= 1
        ellipse = (x**2 / semi_major**2) + (y**2 / semi_minor**2) <= 1
        
        return ellipse.astype(np.uint8)


    # Function to process each slice
    def process_slice(z):
        tiff_slice = tiff_array[z].astype(np.uint8)
        dilated_slice = cv2.dilate(tiff_slice, kernel, iterations=1)
        return z, dilated_slice

    def process_slice_other(y):
        tiff_slice = tiff_array[:, y, :].astype(np.uint8)
        dilated_slice = cv2.dilate(tiff_slice, kernel, iterations=1)
        return y, dilated_slice

    # Create empty arrays to store the dilated results for the XY and XZ planes
    dilated_xy = np.zeros_like(tiff_array, dtype=np.uint8)
    dilated_xz = np.zeros_like(tiff_array, dtype=np.uint8)

    kernel_x = int(dilated_x)
    kernel = create_circular_kernel(kernel_x)

    num_cores = mp.cpu_count()

    with ThreadPoolExecutor(max_workers=num_cores) as executor:
        futures = {executor.submit(process_slice, z): z for z in range(tiff_array.shape[0])}

        for future in as_completed(futures):
            z, dilated_slice = future.result()
            dilated_xy[z] = dilated_slice

    kernel_x = int(dilated_x)
    kernel_z = int(dilated_z)

    kernel = create_ellipsoidal_kernel(kernel_x, kernel_z)

    with ThreadPoolExecutor(max_workers=num_cores) as executor:
        futures = {executor.submit(process_slice_other, y): y for y in range(tiff_array.shape[1])}
        
        for future in as_completed(futures):
            y, dilated_slice = future.result()
            dilated_xz[:, y, :] = dilated_slice

    # Overlay the results
    final_result = dilated_xy | dilated_xz

    return final_result


def dilate_3D_old(tiff_array, dilated_x, dilated_y, dilated_z):
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

def binarize(image):
    """Convert an array from numerical values to boolean mask"""
    return (image != 0).astype(np.uint8)

def invert_array(array):
    """Used to flip glom array indices. 0 becomes 1 and vice versa."""
    return np.logical_not(array).astype(np.uint8)

def process_chunk(start_idx, end_idx, nodes, ring_mask, nearest_label_indices):
    nodes_chunk = nodes[start_idx:end_idx]
    ring_mask_chunk = ring_mask[start_idx:end_idx]
    dilated_nodes_with_labels_chunk = np.copy(nodes_chunk)
    ring_indices = np.argwhere(ring_mask_chunk)

    for index in ring_indices:
        z, y, x = index
        nearest_z, nearest_y, nearest_x = nearest_label_indices[:, z + start_idx, y, x]
        dilated_nodes_with_labels_chunk[z, y, x] = nodes[nearest_z, nearest_y, nearest_x]
    
    return dilated_nodes_with_labels_chunk

def smart_dilate(nodes, dilate_xy, dilate_z, directory = None, GPU = True, fast_dil = False):

    original_shape = nodes.shape

    # Step 1: Binarize the labeled array
    binary_nodes = binarize(nodes)

    # Step 2: Dilate the binarized array
    if not fast_dil:
        dilated_binary_nodes = dilate_3D(binary_nodes, dilate_xy, dilate_xy, dilate_z)
    else:
        dilated_binary_nodes = dilate_3D_old(binary_nodes, dilate_xy, dilate_xy, dilate_z)


    # Step 3: Isolate the ring (binary dilated mask minus original binary mask)
    ring_mask = dilated_binary_nodes & invert_array(binary_nodes)

    print("Preforming distance transform for smart search... this step may take some time if computed on CPU...")

    try:

        if GPU == True and cp.cuda.runtime.getDeviceCount() > 0:
            print("GPU detected. Using CuPy for distance transform.")

            try:

                # Step 4: Find the nearest label for each voxel in the ring
                nearest_label_indices = compute_distance_transform_GPU(invert_array(nodes))

            except cp.cuda.memory.OutOfMemoryError as e:
                down_factor = catch_memory(e) #Obtain downsample amount based on memory missing

                while True:
                    downsample_needed = down_factor**(1./3.)
                    small_nodes = nettracer.downsample(nodes, downsample_needed) #Apply downsample
                    try:
                        nearest_label_indices = compute_distance_transform_GPU(invert_array(small_nodes)) #Retry dt on downsample
                        print(f"Using {down_factor} downsample ({downsample_needed} in each dim - Largest possible with this GPU)")
                        break
                    except cp.cuda.memory.OutOfMemoryError:
                        down_factor += 1
                binary_nodes = binarize(small_nodes) #Recompute variables for downsample
                dilated_mask = dilated_binary_nodes #Need this for later to stamp out the correct output
                if not fast_dil:
                    dilated_binary_nodes = dilate_3D(binary_nodes, 2 + round_to_odd(dilate_xy/downsample_needed), 2 + round_to_odd(dilate_xy/downsample_needed), 2 + round_to_odd(dilate_z/downsample_needed)) #Mod dilation to recompute variables for downsample while also over dilatiing
                else:
                    dilated_binary_nodes = dilate_3D_old(binary_nodes, 2 + round_to_odd(dilate_xy/downsample_needed), 2 + round_to_odd(dilate_xy/downsample_needed), 2 + round_to_odd(dilate_z/downsample_needed)) 
                ring_mask = dilated_binary_nodes & invert_array(binary_nodes)
                nodes = small_nodes
                del small_nodes
        else:
            goto_except = 1/0
    except Exception as e:
        print("GPU dt failed or did not detect GPU (cupy must be installed with a CUDA toolkit setup...). Computing CPU distance transform instead.")
        nearest_label_indices = compute_distance_transform(invert_array(nodes))


    # Step 5: Process in parallel chunks using ThreadPoolExecutor
    num_cores = mp.cpu_count()  # Use all available CPU cores
    chunk_size = nodes.shape[0] // num_cores  # Divide the array into chunks along the z-axis

    with ThreadPoolExecutor(max_workers=num_cores) as executor:
        args_list = [(i * chunk_size, (i + 1) * chunk_size if i != num_cores - 1 else nodes.shape[0], nodes, ring_mask, nearest_label_indices) for i in range(num_cores)]
        results = list(executor.map(lambda args: process_chunk(*args), args_list))

    # Combine results from chunks
    dilated_nodes_with_labels = np.concatenate(results, axis=0)

    if nodes.shape[1] < original_shape[1]: #If downsample was used, upsample output
        dilated_nodes_with_labels = nettracer.upsample_with_padding(dilated_nodes_with_labels, downsample_needed, original_shape)
        dilated_nodes_with_labels = dilated_nodes_with_labels * dilated_mask

    if directory is not None:
        try:
            tifffile.imwrite(f"{directory}/search_region.tif", dilated_nodes_with_labels)
        except Exception as e:
            print(f"Could not save search region file to {directory}")

    return dilated_nodes_with_labels

def round_to_odd(number):
    rounded = round(number)
    # If the rounded number is even, add or subtract 1 to make it odd
    if rounded % 2 == 0:
        if number > 0:
            rounded += 1
        else:
            rounded -= 1
    return rounded

def smart_label(binary_array, label_array, directory = None, GPU = True):

    original_shape = binary_array.shape

    if type(binary_array) == str or type(label_array) == str:
        string_bool = True
    else:
        string_bool = None
    if type(binary_array) == str:
        binary_array = tifffile.imread(binary_array)
    if type(label_array) == str:
        label_array = tifffile.imread(label_array)

    # Step 1: Binarize the labeled array
    binary_core = binarize(label_array)
    binary_array = binarize(binary_array)

    # Step 3: Isolate the ring (binary dilated mask minus original binary mask)
    ring_mask = binary_array & invert_array(binary_core)


    try:

        if GPU == True and cp.cuda.runtime.getDeviceCount() > 0:
            print("GPU detected. Using CuPy for distance transform.")

            try:

                # Step 4: Find the nearest label for each voxel in the ring
                nearest_label_indices = compute_distance_transform_GPU(invert_array(label_array))

            except cp.cuda.memory.OutOfMemoryError as e:
                down_factor = catch_memory(e) #Obtain downsample amount based on memory missing

                while True:
                    downsample_needed = down_factor**(1./3.)
                    small_array = nettracer.downsample(label_array, downsample_needed) #Apply downsample
                    try:
                        nearest_label_indices = compute_distance_transform_GPU(invert_array(small_array)) #Retry dt on downsample
                        print(f"Using {down_factor} downsample ({downsample_needed} in each dim - Largest possible with this GPU)")
                        break
                    except cp.cuda.memory.OutOfMemoryError:
                        down_factor += 1
                binary_mask = binary_array #Need this for later to stamp out the correct output
                binary_core = binarize(small_array)
                label_array = small_array
                binary_array = nettracer.downsample(binary_array, downsample_needed)
                binary_array = nettracer.dilate_3D(binary_array, 3, 3, 3)
                ring_mask = binary_array & invert_array(binary_core)

        else:
            goto_except = 1/0
    except Exception as e:
        print("GPU dt failed or did not detect GPU (cupy must be installed with a CUDA toolkit setup...). Computing CPU distance transform instead.")
        print(f"Error message: {str(e)}")
        nearest_label_indices = compute_distance_transform(invert_array(label_array))

    print("Preforming distance transform for smart label...")

    # Step 5: Process in parallel chunks using ThreadPoolExecutor
    num_cores = mp.cpu_count()  # Use all available CPU cores
    chunk_size = label_array.shape[0] // num_cores  # Divide the array into chunks along the z-axis

    with ThreadPoolExecutor(max_workers=num_cores) as executor:
        args_list = [(i * chunk_size, (i + 1) * chunk_size if i != num_cores - 1 else label_array.shape[0], label_array, ring_mask, nearest_label_indices) for i in range(num_cores)]
        results = list(executor.map(lambda args: process_chunk(*args), args_list))

    # Combine results from chunks
    dilated_nodes_with_labels = np.concatenate(results, axis=0)

    if label_array.shape[1] < original_shape[1]: #If downsample was used, upsample output
        dilated_nodes_with_labels = nettracer.upsample_with_padding(dilated_nodes_with_labels, downsample_needed, original_shape)
        dilated_nodes_with_labels = dilated_nodes_with_labels * binary_mask

    if string_bool:
        if directory is not None:
            try:
                tifffile.imwrite(f"{directory}/smart_labelled_array.tif", dilated_nodes_with_labels)
            except Exception as e:
                print(f"Could not save search region file to {directory}")
        else:
            try:
                tifffile.imwrite("smart_labelled_array.tif", dilated_nodes_with_labels)
            except Exception as e:
                print(f"Could not save search region file to active directory")

    return dilated_nodes_with_labels

def compute_distance_transform_GPU(nodes):
    # Convert numpy array to CuPy array
    nodes_cp = cp.asarray(nodes)
    
    # Compute the distance transform on the GPU
    distance, nearest_label_indices = cpx.distance_transform_edt(nodes_cp, return_indices=True)
    
    # Convert results back to numpy arrays
    nearest_label_indices_np = cp.asnumpy(nearest_label_indices)
    
    return nearest_label_indices_np


def compute_distance_transform(nodes):
    distance, nearest_label_indices = distance_transform_edt(nodes, return_indices=True)
    return nearest_label_indices



def compute_distance_transform_distance_GPU(nodes):

    # Convert numpy array to CuPy array
    nodes_cp = cp.asarray(nodes)
    
    # Compute the distance transform on the GPU
    distance, nearest_label_indices = cpx.distance_transform_edt(nodes_cp, return_indices=True)
    
    # Convert results back to numpy arrays
    distance = cp.asnumpy(distance)
    
    return distance    


def compute_distance_transform_distance(nodes):

    # Fallback to CPU if there's an issue with GPU computation
    distance, nearest_label_indices = distance_transform_edt(nodes, return_indices=True)
    return distance




def gaussian(search_region, GPU = True):
    try:
        if GPU == True and cp.cuda.runtime.getDeviceCount() > 0:
            print("GPU detected. Using CuPy for guassian blur.")

            # Convert to CuPy array
            search_region_cp = cp.asarray(search_region)

            # Apply Gaussian filter
            blurred_search_cp = cpx.gaussian_filter(search_region_cp, sigma=1)

            # Convert back to NumPy array if needed
            blurred_search = cp.asnumpy(blurred_search_cp)

            return blurred_search
        else:
            print("No GPU detected. Using CPU for guassian blur.")
            blurred_search = gaussian_filter(search_region, sigma = 1)
            return blurred_search
    except Exception as e:
        print("GPU blur failed or did not detect GPU (cupy must be installed with a CUDA toolkit setup...). Computing CPU guassian blur instead.")
        print(f"Error message: {str(e)}")
        # Fallback to CPU if there's an issue with GPU computation
        blurred_search = gaussian_filter(search_region, sigma = 1)
        return blurred_search

def get_local_maxima(distance, image):
    try:
        if cp.cuda.runtime.getDeviceCount() > 0:
            print("GPU detected. Using CuPy for local maxima.")

            distance = cp.asarray(distance)

            # Perform a maximum filter to find local maxima
            footprint = cp.ones((3, 3, 3))  # Define your footprint
            filtered = maximum_filter(distance, footprint=footprint)

            # Find local maxima by comparing with the original array
            local_max = (distance == filtered)  # Peaks are where the filtered result matches the original

            # Extract coordinates of local maxima
            coords = cp.argwhere(local_max)
            coords = cp.asnumpy(coords)

            return coords
        else:
            print("No GPU detected. Using CPU for local maxima.")
            coords = peak_local_max(distance, footprint=np.ones((3, 3, 3)), labels=image)
            return coords
    except Exception as e:
        print("GPU operation failed or did not detect GPU (cupy must be installed with a CUDA toolkit set up...). Computing CPU local maxima instead.")
        coords = peak_local_max(distance, footprint=np.ones((3, 3, 3)), labels=image)
        return coords


def catch_memory(e):


    # Get the current GPU device
    device = cp.cuda.Device()

    # Get total memory in bytes
    total_memory = device.mem_info[1]

    # Capture the error message
    error_message = str(e)
    print(f"Error encountered: {error_message}")

    # Use regex to extract the memory required from the error message
    match = re.search(r'allocating ([\d,]+) bytes', error_message)

    if match:
        memory_required = int(match.group(1).replace(',', ''))

        print(f"GPU Memory required for distance transform: {memory_required}, retrying with temporary downsample")

        downsample_needed = (memory_required/total_memory)
        return (downsample_needed)


if __name__ == "__main__":
    nodes = input("Labelled Nodes tiff?: ")
    nodes = tifffile.imread(nodes)

    # Step 1: Binarize the labeled array
    binary_nodes = binarize(nodes)

    # Step 2: Dilate the binarized array
    dilated_binary_nodes = dilate_3D(binary_nodes, 10, 10, 10)

    # Step 3: Isolate the ring (binary dilated mask minus original binary mask)
    ring_mask = dilated_binary_nodes & invert_array(binary_nodes)

    # Step 4: Find the nearest label for each voxel in the ring
    distance, nearest_label_indices = distance_transform_edt(invert_array(nodes), return_indices=True)

    # Step 5: Process in parallel chunks using ThreadPoolExecutor
    num_cores = mp.cpu_count()  # Use all available CPU cores
    chunk_size = nodes.shape[0] // num_cores  # Divide the array into chunks along the z-axis

    with ThreadPoolExecutor(max_workers=num_cores) as executor:
        args_list = [(i * chunk_size, (i + 1) * chunk_size if i != num_cores - 1 else nodes.shape[0], nodes, ring_mask, nearest_label_indices) for i in range(num_cores)]
        results = list(executor.map(lambda args: process_chunk(*args), args_list))

    # Combine results from chunks
    dilated_nodes_with_labels = np.concatenate(results, axis=0)

    # Save the result
    output_file = "dilated_nodes_with_labels.tif"
    tifffile.imwrite(output_file, dilated_nodes_with_labels)
    print(f"Result saved to {output_file}")