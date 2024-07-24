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

def draw_gloms(gloms, num_gloms):
    # Find centroids
    centroids = np.array([np.mean(np.argwhere(gloms == i), axis=0) for i in range(1, num_gloms + 1)])

    # Create a new 3D array to draw on with the same dimensions as the original array
    draw_array = np.zeros_like(gloms, dtype=np.uint8)

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
    tifffile.imwrite("labelled_gloms.tif", draw_array)

def draw_from_centroids(gloms, num_gloms, centroids, twod_bool):
    """Presumes a centroid dictionary has been obtained"""
    print("Drawing glom IDs. (Must find all centroids. Network lattice itself may be drawn from network_draw script with fewer centroids)")
    # Create a new 3D array to draw on with the same dimensions as the original array
    draw_array = np.zeros_like(gloms, dtype=np.uint8)

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
        print("ok")
        draw_array = draw_array[0,:,:]

    # Save the draw_array as a 3D TIFF file
    tifffile.imwrite("labelled_node_indices.tif", draw_array)

def mother_draw(mother_dict, centroid_dict, gloms):
    # Create a new 3D array to draw on with the same dimensions as the original array
    draw_array = np.zeros_like(gloms, dtype=np.uint8)
    #font_size = 24

    for glom, degree in mother_dict.items():
        z, y, x = centroid_dict[glom].astype(int)

        print(f"{z}, {y}, {x}")

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

    gloms = tifffile.imread("gloms_for_networks.tif")

    glom_shape = gloms.shape

    gloms = downsample(gloms, 5)

    # Label the connected components
    glom_labels, num_gloms = ndimage.label(gloms)


    # Find centroids
    centroids = np.array([np.mean(np.argwhere(glom_labels == i), axis=0) for i in range(1, num_gloms + 1)])

    # Create a new 3D array to draw on with the same dimensions as the original array
    draw_array = np.zeros_like(gloms, dtype=np.uint8)

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

    if len(glom_shape) == 2:
        draw_array = draw_array[0,:,:]

    # Save the draw_array as a 3D TIFF file
    tifffile.imwrite("draw_array.tif", draw_array)
