import numpy as np
import tifffile
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage import zoom
from skimage.morphology import skeletonize_3d

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

        if search_pixels > 0:
            ratio = white_pixels / search_pixels  # Calculate the ratio
            vals.append(ratio)  # Add the ratio to the list
            depth.append((y - start_y) * y_depth)

    return vals, depth



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

def dilation_length_to_pixels(xy_scaling, z_scaling):
    """Find XY and Z dilation parameters based on voxel micron scaling"""
    microns = 10 #Change this value to alter how many microns are dilating
    dilate_xy = 2 * int(round(microns/xy_scaling))

    # Ensure the dilation param is odd to have a center pixel
    dilate_xy += 1 if dilate_xy % 2 == 0 else 0

    dilate_z = 2 * int(round(microns/z_scaling))

    # Ensure the dilation param is odd to have a center pixel
    dilate_z += 1 if dilate_z % 2 == 0 else 0

    return dilate_xy, dilate_z

def invert_array(array):
    """Used to flip glom array indices. 0 becomes 255 and vice versa."""
    inverted_array = np.where(array == 0, 255, 0).astype(np.uint8)
    return inverted_array

def establish_bounds(arr1, arr2):
    """Used to isolate where nerves interact with gloms"""
    bounds = arr1 * arr2
    return bounds

def count_positive_pixels(binary_array):
    # Check if the input array is 3D
    if binary_array.ndim != 3:
        raise ValueError("Input array must be 3D")

    # Count the number of non-zero elements (positive pixels)
    count = np.count_nonzero(binary_array)

    return count

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
    plt.title("Glom Nervous Innervation vs Cortical Depth")
    plt.xlabel("Cortical Depth (Microns)")
    plt.ylabel("Voxel Density of Nerve Skeletons Near Gloms")

    plt.savefig(f'{directory}/nerve_densities_around_gloms_vertical_distribution.png')

    # Showing the plot
    plt.show()

def upsample(data, factor):
    # Upsample the input binary array

    # Get the dimensions of the original and upsampled arrays
    binary_array = zoom(data, factor, order=0)

    return binary_array

def trim_list(my_list):
    return_list = []
    i = 0
    while i < len(my_list):
        new_number = 0
        for k in range(min(5, len(my_list) - i)):
            new_number += my_list[i]
            i += 1
        return_list.append(new_number)
    return return_list

def nerve_graph(glom_name, nerve_name, xy_scale, z_scale, smallx, smallz, directory):

    ratiox = xy_scale/smallx
    ratioz = z_scale/smallz

    #Convert voxel scaling to actual pixels to dilate (note this script presumes you will dilate by 10 um)
    dilate_xy, dilate_z = dilation_length_to_pixels(smallx, smallz)

    #Convert tiffs to numpy arrays:
    gloms = tifffile.imread(glom_name)
    nerve = tifffile.imread(nerve_name)

    print("Normalizing...")
    gloms = upsample(gloms, (ratioz, ratiox, ratiox))

    nerve = upsample(nerve, (ratioz, ratiox, ratiox))


    print("Skeletonizing  nerves...")
    nerve = skeletonize_3d(nerve)


    inverted_gloms = invert_array(gloms)

    print("dilating gloms to establish bounds...")


    #Dilate glom objects to establish search region for nerves
    dilated_gloms = dilate_3D(gloms, dilate_xy, dilate_xy, dilate_z)

    print("establishing bounds...")

    search_region = establish_bounds(dilated_gloms, inverted_gloms)

    search_vol = count_positive_pixels(search_region)

    search_vol = float(search_vol)

    isolated_nerves = establish_bounds(dilated_gloms, nerve)

    nerve_vol = count_positive_pixels(isolated_nerves)

    nerve_density = nerve_vol/search_vol

    #nerve_density = nerve_density * xy_scale * xy_scale * z_scale
    

    vals, depth = calculate_y_vals(isolated_nerves, search_region, smallx)

    vals = trim_list(vals)
    depth = trim_list(depth)

    df = pd.DataFrame({
        'Depth': depth,
        'Vals': vals
    })

    # Save the DataFrame to an Excel file
    excel_filename = f'{directory}/y_depth_nerves.xlsx'
    df.to_excel(excel_filename, index=False)

    print(f'Data saved to {excel_filename}')


    plot_scatter(depth, vals, directory)