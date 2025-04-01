import numpy as np
import tifffile
import cv2
from scipy.ndimage import zoom
from skimage.morphology import skeletonize_3d


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

def upsample(data, factor):
    # Upsample the input binary array

    # Get the dimensions of the original and upsampled arrays
    binary_array = zoom(data, factor, order=0)

    return binary_array


def nerve_graph(glom_name, nerve_name, xy_scale, z_scale, smallx, smallz, directory):

    ratiox = xy_scale/smallx
    ratioz = z_scale/smallz

    #Convert voxel scaling to actual pixels to dilate (note this script presumes you will dilate by 10 um)
    dilate_xy, dilate_z = dilation_length_to_pixels(smallx, smallz)

    #Convert tiffs to numpy arrays:
    print("normalizing...")
    gloms = tifffile.imread(glom_name)
    gloms = upsample(gloms, (ratioz, ratiox, ratiox))
    nerve = tifffile.imread(nerve_name)
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

    isolated_nerves = establish_bounds(search_region, nerve)

    nerve_vol = count_positive_pixels(isolated_nerves)

    nerve_density = nerve_vol/search_vol

    print(f"Estimated density of nerves within 10 um of gloms in vol: {nerve_density}")
    sci_note = f"{nerve_density:e}"
    print(f"{sci_note}")
        
    with open(f'{directory}/nerve_density.txt', 'w') as f:
        f.write(f'Estimated nerve density is {sci_note} um3 of nerve/um3 of glom adjacent volume')
    f.close()
