import tifffile
import numpy as np

my_arr = input("where array?" )
my_arr = tifffile.imread(my_arr)
my_arr = my_arr > 0
my_arr = my_arr * 255
my_arr = my_arr.astype(np.uint8)
tifffile.imwrite("output.tif", my_arr)