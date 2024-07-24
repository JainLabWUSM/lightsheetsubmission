import tifffile
import numpy as np
from scipy.spatial import ConvexHull

def read_tiff(file_path):
    # Read the TIFF file as a 3D numpy array
    tiff_data = tifffile.imread(file_path)
    return tiff_data

def get_convex_hull_volume(point_cloud):
    # Flatten the 3D numpy array to obtain the point cloud
    points = np.column_stack(np.where(point_cloud))
    
    # Create a convex hull around the point cloud
    hull = ConvexHull(points)
    
    # Calculate the volume of the convex hull
    volume = hull.volume
    return volume

def main():
    # Prompt the user for the input TIFF file
    tiff_path = input("Enter the path to the binary TIFF file: ")
    xy_dims = float(input("xy dims?: "))
    z_dims = float(input("z dims?: "))

    try:
        # Read the TIFF file as a 3D numpy array
        tiff_data = read_tiff(tiff_path)

        # Get the volume of the convex hull
        hull_volume = get_convex_hull_volume(tiff_data)

        # Print the volume of the convex hull
        print(f"The pixel volume of the 3D convex hull is: {hull_volume:.2f}")

        convex_vol = hull_volume * xy_dims * xy_dims * z_dims

        count = np.count_nonzero(tiff_data)

        tiff_data_vol = count * xy_dims * xy_dims * z_dims

        tiff_sci = f"{tiff_data_vol:e}"

        convol_sci = f"{convex_vol:e}"

        density = tiff_data_vol/convex_vol

        dens_sci = f"{density:e}"

        print(f"The volume of the convex hull is {convol_sci}")
        print(f"The volume of the data is {tiff_sci}")
        print(f"The density of the data is {dens_sci}")

    
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()