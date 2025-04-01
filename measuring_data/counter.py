import tifffile
import numpy as np




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

    print(unique_values)

    # Get the total number of unique values
    total_unique_values = len(unique_values)

    return total_unique_values


def count_vals(file):
    array = tifffile.imread(file)

    num = count_unique_values(array)

    print(f"There are {num} unique, non-zero values in this image")