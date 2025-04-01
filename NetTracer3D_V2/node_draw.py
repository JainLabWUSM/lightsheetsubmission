import numpy as np
import tifffile
from scipy import ndimage
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import zoom


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

def draw_nodes(nodes, num_nodes):
    # Find centroids
    centroids = np.array([np.mean(np.argwhere(nodes == i), axis=0) for i in range(1, num_nodes + 1)])

    # Create a new 3D array to draw on with the same dimensions as the original array
    draw_array = np.zeros_like(nodes, dtype=np.uint8)

    # Use the default font from ImageFont
    font_size = None

    # Iterate through each centroid
    for idx, centroid in enumerate(centroids, start=1):
        z, y, x = centroid.astype(int)

        try:
            draw_array = _draw_at_plane(z, y, x, draw_array, idx, font_size)
        except IndexError:
            pass

        try:
            draw_array = _draw_at_plane(z + 1, y, x, draw_array, idx, font_size)
        except IndexError:
            pass

        try:
            draw_array = _draw_at_plane(z - 1, y, x, draw_array, idx, font_size)
        except IndexError:
            pass

    # Save the draw_array as a 3D TIFF file
    tifffile.imwrite("labelled_nodes.tif", draw_array)

def draw_from_centroids(nodes, num_nodes, centroids, twod_bool, directory = None):
    """Presumes a centroid dictionary has been obtained"""
    print("Drawing node IDs. (Must find all centroids. Network lattice itself may be drawn from network_draw script with fewer centroids)")
    # Create a new 3D array to draw on with the same dimensions as the original array
    draw_array = np.zeros_like(nodes, dtype=np.uint8)

    # Use the default font from ImageFont

    # Iterate through each centroid
    for idx in centroids.keys():
        centroid = centroids[idx]
        z, y, x = centroid.astype(int)

        try:
            draw_array = _draw_at_plane(z, y, x, draw_array, idx)
        except IndexError:
            pass

        try:
            draw_array = _draw_at_plane(z + 1, y, x, draw_array, idx)
        except IndexError:
            pass

        try:
            draw_array = _draw_at_plane(z - 1, y, x, draw_array, idx)
        except IndexError:
            pass

    if twod_bool:
        draw_array = draw_array[0,:,:] | draw_array[1,:,:]


    if directory is None:
        filename = 'labelled_node_indices.tif'
    else:
        filename = f'{directory}/labelled_node_indices.tif'

    try:

        # Save the draw_array as a 3D TIFF file
        tifffile.imwrite(filename, draw_array)

    except Exception as e:
        print(f"Could not save node indices to {filename}")

def degree_draw(degree_dict, centroid_dict, nodes):
    # Create a new 3D array to draw on with the same dimensions as the original array
    draw_array = np.zeros_like(nodes, dtype=np.uint8)
    #font_size = 24

    for node, degree in degree_dict.items():
        z, y, x = centroid_dict[node].astype(int)

        try:
            draw_array = _draw_at_plane(z, y, x, draw_array, degree)
        except IndexError:
            pass

        try:
            draw_array = _draw_at_plane(z + 1, y, x, draw_array, degree)
        except IndexError:
            pass

        try:
            draw_array = _draw_at_plane(z - 1, y, x, draw_array, degree)
        except IndexError:
            pass


    return draw_array

def degree_infect(degree_dict, nodes):

    num_nodes = np.max(nodes)
    return_nodes = nodes.copy()

    for node in range(1, num_nodes + 1):
        if node not in degree_dict:
            continue
        else:
            idxs = np.argwhere(nodes == node)
            for idx in idxs:
                return_nodes[tuple(idx)] = degree_dict[node]

    return return_nodes


def _draw_at_plane(z_loc, y_loc, x_loc, array, num, font_size=None):
    # Get the 2D slice at the specified Z position
    slice_to_draw = array[z_loc, :, :]

    # Create an image from the 2D slice
    image = Image.fromarray(slice_to_draw.astype(np.uint8) * 255)

    # Create a drawing object
    draw = ImageDraw.Draw(image)

    # Load the default font with the specified font size

    font = ImageFont.load_default()

    # Draw the number at the centroid index
    draw.text((x_loc, y_loc), str(num), fill='white', font=font)

    # Save the modified 2D slice into draw_array at the specified Z position
    array[z_loc, :, :] = np.array(image)

    return array

def compute_centroid(binary_stack, label):
    """
    Finds centroid of labelled object in array
    """
    indices = np.argwhere(binary_stack == label)
    centroid = np.round(np.mean(indices, axis=0)).astype(int)

    return centroid

if __name__ == "__main__":

    nodes = tifffile.imread("nodes_for_networks.tif")

    node_shape = nodes.shape

    nodes = downsample(nodes, 5)

    # Label the connected components
    structure_3d = np.ones((3, 3, 3), dtype=int)
    nodes, num_nodes = ndimage.label(nodes, structure=structure_3d)

    # Find centroids
    centroids = np.array([np.mean(np.argwhere(node_labels == i), axis=0) for i in range(1, num_nodes + 1)])

    # Create a new 3D array to draw on with the same dimensions as the original array
    draw_array = np.zeros_like(nodes, dtype=np.uint8)

    # Use the default font from ImageFont
    font = ImageFont.load_default()

    # Iterate through each centroid
    for idx, centroid in enumerate(centroids, start=1):
        z, y, x = centroid.astype(int)

        # Get the 2D slice at the specified Z position
        slice_to_draw = draw_array[z, :, :]

        # Create an image from the 2D slice
        image = Image.fromarray(slice_to_draw.astype(np.uint8) * 255)

        # Create a drawing object
        draw = ImageDraw.Draw(image)

        # Draw the number at the centroid index
        draw.text((x, y), str(idx), fill='white', font=font)

        # Save the modified 2D slice into draw_array at the specified Z position
        draw_array[z, :, :] = np.array(image)

    if len(node_shape) == 2:
        draw_array = draw_array[0,:,:]

    # Save the draw_array as a 3D TIFF file
    tifffile.imwrite("draw_array.tif", draw_array)
