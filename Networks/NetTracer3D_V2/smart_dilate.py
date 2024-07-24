import tifffile
import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt
from concurrent.futures import ThreadPoolExecutor
import cv2
import os
import multiprocessing as mp

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

def smart_dilate(nodes, dilate_xy, dilate_z):

    # Step 1: Binarize the labeled array
    binary_nodes = binarize(nodes)

    # Step 2: Dilate the binarized array
    dilated_binary_nodes = dilate_3D(binary_nodes, dilate_xy, dilate_xy, dilate_z)

    # Step 3: Isolate the ring (binary dilated mask minus original binary mask)
    ring_mask = dilated_binary_nodes & invert_array(binary_nodes)

    print("Preforming distance transform for smart search... this step may take several minutes or more but nonetheless is a much faster method for thousands of nodes in a large array...")

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

    tifffile.imwrite("smart_dilate_output.tif", dilated_nodes_with_labels)
    return dilated_nodes_with_labels
    print("Smart search complete...")


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