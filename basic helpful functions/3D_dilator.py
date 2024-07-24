import numpy as np
import cv2
import tifffile

# Info for the user
print("Custom 3D-like dilation with separate scales in X, Y, and Z")

# Obtain the input tiff stack filepath location from the user
input_tiff = input("Input tiff stack filepath location: ")

# Obtain the output tiff stack location and filename from the user
output_tiff = input("Output tiff stack location (include filename at the end of the directory, e.g., C://...//output.tif): ")

# Obtain the dilation information to make the kernel with
dilated_x = input("pixels to dilate x? (Check the scale in your tiff to decide corresponding micron distance. Note that the dilation puts the kernel in the middle of a pixel so the amount you dilate in a dimension should be about twice as far out as you want to go.): ")
dilated_y = input("pixels to dilate y?")
dilated_z = input ("pixels to dilate z? (Note: z probably has different voxel scaling than X,Y): ")

# Load the two 2D images as a stack using tifffile
tiff_array = tifffile.imread(input_tiff)

# Create an empty array to store the dilated results for the XY plane
dilated_xy = np.zeros_like(tiff_array, dtype=np.uint8)

# Create an empty array to store the dilated results for the XZ plane
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

# Save the final result as a TIFF using tifffile
tifffile.imsave(output_tiff, final_result)

print("Proportional 3D-like dilation complete and saved to output_dilated.tiff")