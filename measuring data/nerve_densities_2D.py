import numpy as np
import tifffile
import cv2
from scipy.ndimage import zoom
from skimage.morphology import skeletonize


def dilate_2D(tiff_array, dilated_x, dilated_y):
    """Dilate an array in 3D. Consider replacing with scipy dilation method.
    Arguments are an array,  and the desired pixel dilation amounts in X, Y, Z."""

    # Create empty arrays to store the dilated results for the XY and XZ planes
    dilated_xy = np.zeros_like(tiff_array, dtype=np.uint8)

    kernel_x = int(dilated_x)
    kernel_y = int(dilated_y)
    kernel = np.ones((kernel_y, kernel_x), dtype=np.uint8)

    final_result = cv2.dilate(tiff_array, kernel, iterations=1)

    return final_result

def dilation_length_to_pixels(xy_scaling, microns):
    """Find XY and Z dilation parameters based on voxel micron scaling"""
    dilate_xy = 2 * int(round(microns/xy_scaling))

    # Ensure the dilation param is odd to have a center pixel
    dilate_xy += 1 if dilate_xy % 2 == 0 else 0

    return dilate_xy

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

    # Count the number of non-zero elements (positive pixels)
    count = np.count_nonzero(binary_array)

    return count

def upsample(data, factor):
    # Upsample the input binary array

    # Get the dimensions of the original and upsampled arrays
    binary_array = zoom(data, factor, order=0)

    return binary_array



glom_name = input("Gloms tif?: ")
nerve_name = input("Nerves tif?: ")


xy_scale = 0.62


#Convert voxel scaling to actual pixels to dilate (note this script presumes you will dilate by 10 um)
dilate_xy = dilation_length_to_pixels(0.62, 10)

#Convert tiffs to numpy arrays:

gloms = tifffile.imread(glom_name)
nerve = tifffile.imread(nerve_name)




inverted_gloms = invert_array(gloms)


#Dilate glom objects to establish search region for nerves
dilated_gloms = dilate_2D(gloms, dilate_xy, dilate_xy)


search_region = establish_bounds(dilated_gloms, inverted_gloms)

search_vol = count_positive_pixels(search_region)

search_vol = float(search_vol)

isolated_nerves = establish_bounds(search_region, nerve)

nerve_vol = count_positive_pixels(isolated_nerves)

nerve_density = nerve_vol/search_vol

total_nerve_len = count_positive_pixels(nerve)

nerve_density_real = nerve_density * 0.62 * 0.62

search_power = search_vol * 0.62 * 0.62
sci_power = f"{search_power:e}"

glom_vol = count_positive_pixels(gloms)
glom_vol = glom_vol * 0.62 * 0.62
glom_vol = f"{glom_vol:e}"

print(f"Estimated density of nerves within 10 um of gloms in vol: {nerve_density}")

print(f"Pixel len of nerves = {nerve_vol}")







#Convert voxel scaling to actual pixels to dilate (note this script presumes you will dilate by 10 um)
dilate_xy = dilation_length_to_pixels(0.62, 15)

gloms = tifffile.imread(glom_name)
nerve = tifffile.imread(nerve_name)




inverted_gloms = invert_array(gloms)


#Dilate glom objects to establish search region for nerves
dilated_gloms = dilate_2D(gloms, dilate_xy, dilate_xy)


search_region = establish_bounds(dilated_gloms, inverted_gloms)

search_vol = count_positive_pixels(search_region)

search_vol = float(search_vol)

isolated_nerves = establish_bounds(search_region, nerve)

nerve_vol = count_positive_pixels(isolated_nerves)

nerve_density = nerve_vol/search_vol

total_nerve_len = count_positive_pixels(nerve)

nerve_density_real = nerve_density * 0.62 * 0.62

search_power = search_vol * 0.62 * 0.62
sci_power = f"{search_power:e}"

glom_vol = count_positive_pixels(gloms)
glom_vol = glom_vol * 0.62 * 0.62
glom_vol = f"{glom_vol:e}"

print(f"Estimated density of nerves within 15 um of gloms in vol: {nerve_density}")
print(f"Pixel len of nerves 15 = {nerve_vol}")
print(f"Nerve in volume = {total_nerve_len}")