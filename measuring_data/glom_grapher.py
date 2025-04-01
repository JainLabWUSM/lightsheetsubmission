import numpy as np
import tifffile
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import ConvexHull, Delaunay
from scipy.ndimage import zoom
from scipy import ndimage

def calculate_y_vals(data, search, y_depth):
    """
    Calculate the ratio of white pixels (value=255) to the total number of pixels
    in the Z, X plane for each Y index of a 3D numpy array.

    Parameters:
    - arr: A numpy array of shape (Z, Y, X) with binary data (0 or 255).

    Returns:
    - A list of ratios for each Y index.
    """
    vals = []  # Initialize an empty list to store the ratios
    depth = [] #Y depth
    start_y = 0
    Z, Y, X = data.shape  # Unpack the shape of the array

    # Iterate through each Y index
    for y in range(Y):
        plane = data[:, y, :]  # Isolate the Z, X plane at the current Y index
        search_region = search[:, y, :]
        white_pixels = np.count_nonzero(plane)  # Count white (255) pixels

        search_pixels = np.count_nonzero(search_region)

        if search_pixels != 0 and start_y == 0:
            start_y = y

        try:

            ratio = white_pixels / search_pixels 

        except ZeroDivisionError:
            continue

        #if ratio > 0.05:
            #continue

        if start_y != 0 and search_pixels == 0:
            break

        ratio = white_pixels / search_pixels  # Calculate the ratio
        vals.append(ratio)  # Add the ratio to the list
        depth.append((y - start_y) * y_depth)

    vals, depth = sort_outliers(vals, depth)

    return vals, depth

def calculate_y_vals_unique(data, search, y_depth):
    """
    Calculate the ratio of white pixels (value=255) to the total number of pixels
    in the Z, X plane for each Y index of a 3D numpy array.

    Parameters:
    - arr: A numpy array of shape (Z, Y, X) with binary data (0 or 255).

    Returns:
    - A list of ratios for each Y index.
    """
    vals = []  # Initialize an empty list to store the ratios
    depth = [] #Y depth
    start_y = 0
    Z, Y, X = data.shape  # Unpack the shape of the array

    # Iterate through each Y index
    for y in range(Y):
        plane = data[:, y, :]  # Isolate the Z, X plane at the current Y index
        search_region = search[:, y, :]
        unique_vals = count_unique_values(plane)  # Count unique vals

        search_pixels = np.count_nonzero(search_region)

        if search_pixels != 0 and start_y == 0:
            start_y = y

        try:

            ratio = unique_vals / search_pixels 

        except ZeroDivisionError:
            continue

        #if ratio > 0.00004:
            #continue

        if start_y != 0 and search_pixels == 0:
            break

        ratio = unique_vals / search_pixels  # Calculate the ratio
        vals.append(ratio)  # Add the ratio to the list
        depth.append((y - start_y) * y_depth)

    vals, depth = sort_outliers(vals, depth)

    return vals, depth

def plot_scatter(x_data, y_data, directory):
    """
    Plot a scatter plot of corresponding data points from two lists.

    Parameters:
    - x_data: List of x-coordinates (or any numerical sequence)
    - y_data: List of y-coordinates (or any numerical sequence) corresponding to x_data
    """
    # Ensure the lists have equal length
    if len(x_data) != len(y_data):
        print("Error: The lists must have the same length.")
        return
    
    # Creating the scatter plot
    plt.scatter(x_data, y_data)
    
    # Adding titles and labels (Optional, but recommended for clarity)
    plt.title("Glom Volumetric Density vs Cortical Depth")
    plt.xlabel("Cortical Depth (Microns)")
    plt.ylabel("Density of Glom in Tissue")

    plt.savefig(f'{directory}/density_gloms_vertical_distribution.png')

    # Showing the plot
    plt.show()

def plot_scatter_unique(x_data, y_data, directory):
    """
    Plot a scatter plot of corresponding data points from two lists.

    Parameters:
    - x_data: List of x-coordinates (or any numerical sequence)
    - y_data: List of y-coordinates (or any numerical sequence) corresponding to x_data
    """
    # Ensure the lists have equal length
    if len(x_data) != len(y_data):
        print("Error: The lists must have the same length.")
        return
    
    # Creating the scatter plot
    plt.scatter(x_data, y_data)
    
    # Adding titles and labels (Optional, but recommended for clarity)
    plt.title("Glom Numerical Density vs Cortical Depth")
    plt.xlabel("Cortical Depth (Microns)")
    plt.ylabel("Density of Glom (number) in Tissue")

    plt.savefig(f'{directory}/unique_gloms_vertical_distribution.png')

    # Showing the plot
    plt.show()

def remove_zeros(input_list):
    # Use boolean indexing to filter out zeros
    result_array = input_list[input_list != 0] #note - presumes your list is an np array

    return result_array

def count_unique_values(label_array):
    # Flatten the 3D array to a 1D array
    flattened_array = label_array.flatten()

    flattened_array = remove_zeros(flattened_array)

    # Find unique values
    unique_values = np.unique(flattened_array)

    # Get the total number of unique values
    total_unique_values = len(unique_values)

    return total_unique_values

def binarize(image):
    """Convert an array from numerical values to boolean mask"""
    image = image != 0

    image = image.astype(np.uint8)

    return image

def calculate_bounds(data):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    upper_bound = q3 + (1.5 * iqr)
    lower_bound = q1 - (1.5 * iqr)
    return upper_bound, lower_bound

def sort_outliers(vals, depth):
    upper_bound, lower_bound = calculate_bounds(vals)
    i = len(vals) - 1
    while i >= 0:
        if vals[i] > upper_bound or vals[i] < lower_bound:
            del vals[i]
            del depth[i]
        i -= 1
    return vals, depth

def flood_fill_hull(image):
    points = np.transpose(np.where(image))
    hull = ConvexHull(points)
    deln = Delaunay(points[hull.vertices])
    idx = np.stack(np.indices(image.shape), axis=-1)
    out_idx = np.nonzero(deln.find_simplex(idx) + 1)
    out_img = np.zeros(image.shape)
    out_img[out_idx] = 1
    return out_img, hull

def downsample(data, factor):
    # Downsample the input data by a specified factor
    return zoom(data, 1/factor, order=0)

def upsample_with_padding(data, factor, original_shape):
    # Upsample the input binary array while adding padding to match the original shape

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

def graph_gloms(gloms_name, xy_scale, directory):

    gloms = tifffile.imread(gloms_name)
    if len(np.unique(gloms)) < 3:
        structure_3d = np.ones((3, 3, 3), dtype=int)
        gloms, _ = ndimage.label(gloms, structure=structure_3d)
    
    binary_gloms = binarize(gloms)

    print("Generating convex hull...")
    masks = binary_gloms
    masks, _ = flood_fill_hull(masks)
    masks = masks.astype(np.uint8)
    masks = upsample_with_padding(masks, 5, binary_gloms.shape)

    gloms = gloms * masks

    vals, depth = calculate_y_vals(binary_gloms, masks, xy_scale)

    df = pd.DataFrame({
        'Depth': depth,
        'Vals': vals
    })

    # Save the DataFrame to an Excel file
    excel_filename = f'{directory}/y_depth_gloms.xlsx'
    df.to_excel(excel_filename, index=False)

    print(f'Data saved to {excel_filename}')
    
    plot_scatter(depth, vals, directory)

    plt.close('all')

    vals, depth = calculate_y_vals_unique(gloms, masks, xy_scale)

    df = pd.DataFrame({
        'Depth': depth,
        'Vals': vals
    })

    # Save the DataFrame to an Excel file
    excel_filename = f'{directory}/y_depth_gloms_unique.xlsx'
    df.to_excel(excel_filename, index=False)

    print(f'Data saved to {excel_filename}')


    plot_scatter_unique(depth, vals, directory)