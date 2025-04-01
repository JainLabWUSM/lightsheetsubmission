import numpy as np
import tifffile as tiff
from scipy import ndimage
from scipy.spatial.distance import cdist
import pandas as pd
from PIL import Image, ImageDraw

def compute_centroids(binary_stack):
    """
    returns centroids of all binary objects in
    a tiff stack as (Z, Y, X) coordinates in a list
    """
    
    labeled_stack, num_labels = ndimage.label(binary_stack)

    centroids = []
    for label in range(1, num_labels + 1):
        indices = np.argwhere(labeled_stack == label)
        centroid = np.round(np.mean(indices, axis=0)).astype(int)
        centroids.append(centroid)

    return centroids


def find_closest_coordinates(binary_array_path, coordinates_list):
    """
    Takes in a list of (Z, Y, X) coordinates and finds the closest
    white pixel to each coordinate.
    The closest pixel and the original coordinate are
    returned in a pairwise list
    """
    
    # Read the binary array
    binary_array = tiff.imread(binary_array_path)

    closest_coordinates = []

    # Find non-zero indices in the binary array
    binary_indices = np.argwhere(binary_array == 255)

    # Loop through each coordinate in the list
    for coord in coordinates_list:
        # Calculate distances between the coordinate and all white pixels
        distances = cdist([coord], binary_indices)

        # Find the index of the closest white pixel
        closest_index = np.argmin(distances)

        # Get the 3D coordinates of the closest white pixel
        closest_coordinate = binary_indices[closest_index]

        # Append the pair to the list
        closest_coordinates.append([closest_coordinate.tolist(), coord])

    # Sort the list based on the labels
    closest_coordinates.sort(key=lambda x: binary_array[tuple(x[0])])

    return closest_coordinates

def find_closest_centroids(binary_array, pairwise_coordinates):
    """
    Takes in the pairwise coordinate list from the above method
    and replaces the coordinate obtained from the 'closest white
    pixel' with the centroid of the object which that pixel is
    attached to. This modified list is returned.
    """
    
    # Convert tiff to array
    binary_array = tiff.imread(binary_array)

    centroids_pairs = []


    # Loop through each pair of coordinates
    for pair in pairwise_coordinates:
        first_coordinate = pair[0]

        centroid = find_object_centroid(binary_array, first_coordinate)

        # Append the pair of centroid and second coordinate to the list
        centroids_pairs.append([centroid, pair[1]])

    return centroids_pairs


def find_object_centroid(binary_array, coordinate):
    """
    Takes in a coordinate (of a white pixel) and a binary array.
    The centroid of the object the white pixel is attached to
    is returned as a (Z, Y, X) list
    """
    
    # Label connected components in the binary array
    labeled_array, num_labels = ndimage.label(binary_array)

    # Get the label of the connected component containing the given coordinate
    label = labeled_array[tuple(coordinate)]

    # Find the indices of the labeled object
    labeled_indices = np.argwhere(labeled_array == label)

    # Get the centroid coordinates of the labeled object
    centroid = np.round(np.mean(labeled_indices, axis=0)).astype(int)

    return centroid.tolist()


def calculate_angles(centroid_pairs, tiff_shape):
    """
    Takes in a pairwise list of coordinates. The angles from the first
    coordinate to the second coordinate (in respect to polar coordinates)
    are returned as a pairwise list of ((azimuth, polar angle...))
    """
    
    angles = []

    # Find the center of the TIFF file
    center = np.array([tiff_shape[2] / 2, tiff_shape[1] / 2, tiff_shape[0] / 2])

    for pair in centroid_pairs:
        base_point = np.array(pair[0])
        end_point = np.array(pair[1])

        # Calculate the vector from the base point to the end point
        vector_end = end_point - base_point

        # Calculate azimuthal angle
        azimuthal_angle = np.arctan2(vector_end[1], vector_end[2])

        if azimuthal_angle < 0:
            azimuthal_angle = azimuthal_angle + 2*np.pi

        # Calculate polar angle
        polar_angle = np.arctan2(vector_end[1], vector_end[0])
        if polar_angle < 0:
            polar_angle = polar_angle + 2*np.pi

        angles.append([azimuthal_angle, polar_angle])

    return angles

def main(input_path):
    """
    Finds centroids of white objects in binary tiff stacks and prints them
    """
    
    # Read the binary TIFF stack
    binary_stack = tiff.imread(input_path)

    # Compute centroids
    centroids = compute_centroids(binary_stack)

    return centroids

def generate_excel_file(angles, directory):
    """
    Generates an excel file from the info in a pairwise list.
    """
    
    # Prompt the user for the Excel file location
    excel_file_path = f'{directory}/angles_of_gloms.xlsx'

    # Create a DataFrame from the angles list
    df = pd.DataFrame(angles, columns=["Azimuthal", "Polar"])

    # Save the DataFrame to an Excel file
    df.to_excel(excel_file_path, index=False)

    print(f"Excel file saved at: {excel_file_path}")


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


def pair_extrapolation(start, end, shape):
    """
    For two points in a 3D array, finds the slope between the start
    and end points and uses that to find the 'furthest' point in the array
    whilst following that slope. Returns a coordinate pair of (start, furthest point)
    """
    
    # Find slope of points:
    slope = end - start

    # loop to find the furthest point in the array without interpolation
    while (
        0 <= end[0] < shape[0]
        and 0 <= end[1] < shape[1]
        and 0 <= end[2] < shape[2]
    ):
        # Update each element separately
        end[0] += slope[0]
        end[1] += slope[1]
        end[2] += slope[2]

    new_pair = [start, end]
    return new_pair

def draw_rays(centroids, tiff_shape, directory):
    """
    Draws rays between two points in a pairwise list of coordinates. First point in each pair
    is treated as the basepoint.
    """
    
    # Prompt the user for the TIFF file location to save
    tiff_file_path = f'{directory}/glomerulus_angle_rays.tif'

    # Create a new 3D NumPy array with the same dimensions as the input TIFF file
    new_binary_stack = np.zeros(tiff_shape, dtype=np.uint8)

    for pair in centroids:
        new_pair = pair_extrapolation(pair[0], pair[1], tiff_shape)
        # Get the line coordinates
        draw_line_inplace(new_pair[0], new_pair[1], new_binary_stack)

        #Make glom centroid gray (for reference in other scripts, potentially)
        new_binary_stack[new_pair[0][0], new_pair[0][1], new_pair[0][2]] = 128

        
    #for some reason the ray drawing method leaves errant pixels at the ray edge when it hits a face
    #so this call just makes its faces black instead.
    new_binary_stack = _convert_faces_to_black(new_binary_stack)

    # Save the new binary stack as a TIFF file
    tiff.imwrite(tiff_file_path, new_binary_stack)

    print(f"Binary TIFF file with extended rays saved at: {tiff_file_path}")

def _convert_faces_to_black(array):
    """
    Converts the faces of an array to black.
    """

    # Set values in the first and last planes along X dimension to 0
    array[0, :, :] = 0
    array[-1, :, :] = 0

    # Set values in the first and last planes along Y dimension to 0
    array[:, 0, :] = 0
    array[:, -1, :] = 0

    # Set values in the first and last planes along Z dimension to 0
    array[:, :, 0] = 0
    array[:, :, -1] = 0

    return array

def angle_calculation(input_path_1, input_path_2, directory):

    # Process the second TIFF file
    centroids_2 = main(input_path_2)

    # Find closest centroids and create centroid pairs list
    coord_pairs = find_closest_coordinates(input_path_1, centroids_2)

    centroid_pairs = find_closest_centroids(input_path_1, coord_pairs)

    # Calculate angles for each centroid pair
    tiff_shape = tiff.imread(input_path_1).shape
    angles = calculate_angles(centroid_pairs, tiff_shape)

    # Generate Excel file from the angles list
    generate_excel_file(angles, directory)

    #Make rays tiff
    draw_rays(centroid_pairs, tiff_shape, directory)
