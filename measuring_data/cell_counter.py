import tifffile
import numpy as np
from scipy import ndimage
import pandas as pd

def _label_to_boolean(label_array, label_value):
    boolean_array = (label_array == label_value)
    return boolean_array

def count_unique_values(label_array):
    # Flatten the 3D array to a 1D array
    flattened_array = label_array.flatten()

    non_zero = remove_zeros(flattened_array)

    non_zero = non_zero.tolist()

    # Find unique values
    unique_values = set(non_zero)

    # Get the total number of unique values
    total_unique_values = len(unique_values)

    return total_unique_values

def remove_zeros(input_list):
    # Use boolean indexing to filter out zeros
    result_array = input_list[input_list != 0] #note - presumes your list is an np array

    return result_array


def cell_glom(labelled_gloms, num_gloms, cells, xy_dims, z_dims):
    #Init placeholders for return vars
    glom_cells = []
    tot_cells = 0
    tot_gloms = 0
    glom_vols = []

    #Iterate through all gloms
    for label in range(num_gloms + 1):
        if label == 0:
            continue
        else:
            
            #Cast each glom to boolean mask
            mask = _label_to_boolean(labelled_gloms, label)

            #Mask cell array by glom boolean mask
            masked_array = mask * cells

            #Get cell count for aforementioned glom
            num_cells = count_unique_values(masked_array)

            vol = np.argwhere(labelled_gloms == label)

            pixel_vol = len(vol)

            vol = pixel_vol * xy_dims * xy_dims * z_dims

            if num_cells > 0:
                #Add cell count to list
                glom_cells.append(num_cells)

                glom_vols.append(vol)

            #Add cell count to total cell count and increase glom count by one if cells found (some gloms discluded in original cell mask generation)
            if num_cells > 1:
                tot_cells += num_cells
                tot_gloms += 1

    return glom_cells, tot_cells, tot_gloms, glom_vols

def calculate_cells(gloms, cells, xy_dims, z_dims, directory):

    gloms = tifffile.imread(gloms)
    cells = tifffile.imread(cells)

    #Assign unique vals to gloms
    labelled_gloms, num_gloms = ndimage.label(gloms)

    #Obtain a list of indiv cell counts for gloms, total cell count, and total glom count
    glom_cells, tot_cells, tot_gloms, glom_vols = cell_glom(labelled_gloms, num_gloms, cells, xy_dims, z_dims)

    #Calculate avg cell density
    average_cell_density = tot_cells/tot_gloms

    # Create a DataFrame with a single column
    df = pd.DataFrame({"Glom_Cells": glom_cells, "Glom_Vol": glom_vols})

    # Save the DataFrame to an Excel file
    df.to_excel(f"{directory}/glom_cells_output.xlsx", index=False)

    print(f"Average Cell Density: {average_cell_density}")