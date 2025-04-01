import numpy as np
import tifffile
import cv2
import matplotlib.pyplot as plt
import pandas as pd

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

        if search_pixels and start_y == 0:
            start_y = y

        try:

            ratio = white_pixels / search_pixels 

        except ZeroDivisionError:
            continue

        ratio = white_pixels / search_pixels  # Calculate the ratio
        vals.append(ratio)  # Add the ratio to the list
        depth.append((y - start_y) * y_depth)

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
    plt.title("Collecting Duct Density vs Cortical Depth")
    plt.xlabel("Cortical Depth (Microns)")
    plt.ylabel("Density of Collecting Duct in Tissue")
    
    plt.savefig(f'{directory}/duct_density_vertical_distribution.png')

    # Showing the plot
    plt.show()

def sort_outliers(vals, depth):
    upper_bound, lower_bound = calculate_bounds(vals)
    i = len(vals) - 1
    while i >= 0:
        if vals[i] > upper_bound or vals[i] < lower_bound:
            del vals[i]
            del depth[i]
        i -= 1
    return vals, depth

def calculate_bounds(data):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    upper_bound = q3 + (1.5 * iqr)
    lower_bound = q1 - (1.5 * iqr)
    return upper_bound, lower_bound

def duct_graph(ducts_name, mask_name, xy_scale, directory):

    ducts = tifffile.imread(ducts_name)
    masks = tifffile.imread(mask_name)

    ducts = ducts * masks

    vals, depth = calculate_y_vals(ducts, masks, xy_scale)

    vals, depths = sort_outliers(vals, depth)

    df = pd.DataFrame({
        'Depth': depth,
        'Vals': vals
    })

    # Save the DataFrame to an Excel file
    excel_filename = f'{directory}/y_depth_ducts.xlsx'
    df.to_excel(excel_filename, index=False)

    print(f'Data saved to {excel_filename}')


    plot_scatter(depth, vals, directory)


