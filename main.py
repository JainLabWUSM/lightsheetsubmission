#This main script has been set up to run through the scripts that were used in the paper.
#Please note the codes written in this main script are intended to apply to the directory structure of the
#Code ocean capsule that was published with the paper.
#Running this script without that directory structure (if downloaded off github for example) will not do anything
#However I left this script here anyway as an example of how to use the other scripts to process example data.

#4/1/25 update:
#Note that nettracer3d is in a much more developed state now compared to when it was used to do this paper analysis and has a GUI
#Find the latest distribution here as a python package (can be installed easily in an anaconda env): https://pypi.org/project/nettracer3d/

#older:

import matplotlib.pyplot as plt
#NetTracer3D has two iterations. V1 is not actually suited to be run from a virtual python env, but V2 is. So we will be demoing V2 here. V1 and V2 have largely the same outputs when applied with the right params. See their respective readmes for more information.

#Importing NetTracer3Ds main script
from NetTracer3D_V2 import nettracer as n3d #Run everything from the nettracer script

#In nettracer3d, a lot of functions are organized around this Network_3D object. A Network_3D object stores combined properties of morphological space and network data. It contains the following properties:

#nodes - a 3D numpy array containing labelled objects that represent nodes in a network (glomeruli, in this case)
#network - a networkx graph object
#xy_scale - a float representing the scale of each pixel in the nodes array.
#z_scale - a float representing the depth of each voxel in the nodes array.
#network_lists - an internal set of lists that keep track of network data
#edges - a 3D numpy array containing labelled objects that represent edges in a network (nerves, in this case).
#search_region - a 3D numpy array containing labelled objects that represent nodes that have been expanded by some amount to search for connections.
#node_identities - a dictionary that relates all nodes to some string identity that details what the node actually represents. This property pertains to networks made of different types of nodes, from several node images, for example, for different FTUs. This property is not actually relevant in this case and will just store nothing.
#node_centroids - a dictionary containing a [Z, Y, x] centroid for all labelled objects in the nodes attribute.
#edge_centroids - a dictionary containing a [Z, Y, x] centroid for all labelled objects in the edges attribute.

my_network = n3d.Network_3D() #Initializing Network_3D object

#Calculating network for 20x image:
print(f"\nCALCULATING A NETWORK (in this case, for the image SK2_20x_FOV13. The following outputs will get generated from this. 'edge_centroids.xlsx' - centroids of nerve objects; 'labelled_edges.tif' - a grayscale labelled map of nerves as they were utilized to create the network; 'labelled_nodes.tif' - a grayscale labelled image of glomeruli as they were utilized to create the network; 'node_centroids.xlsx' - Centroids of glomerulus objects; 'search_region.tif' - a grayscale labelled image of glomeruli with their expanded search neighborhoods that were utilized to create the network")
my_network.calculate_all('/data/SK2/20x/13/labelled_downsampled_Gloms_for_networks_sk2_13.tif', '/data/SK2/20x/13/Nerves_13-1_downsampled.tif', directory = '/results', search = 10, xy_scale = 4.485, z_scale = 2.325, label_nodes = False, remove_trunk = 0) #Example of how to calculate the network for 20x. If you want to explore results from other FOVs, you will need to change the first parameter which is the file path to a node file, and the second parameter which is the file path to an edge file. Right now it is set to the files corresponding to SK2, 20x objective, FOV 13. The proper names and paths for other glom/nerve datasets can be found in the corresponding folders under the data section. Gloms must be the first parameter while nerves must be the second parameter. Additionally, the xy_scale and z_scale parameters must be changed to match the images they correspond to. These will be noted in each FOV subfolder in the scaling.txt file. For context, this image (SK2_13) and it's network are shown in detail in our supplemental movie 5 at 0:27 (https://doi.org/10.5281/zenodo.14210598).

#calculate_all() is the main method in nettracerv2 when it comes to finding networks. It takes a lot of optional params, but most aren't relevant to this specifically. The main ones to know here are search represents how far out from glomeruli we are looking for nerves (10 microns in this case). The downsampled nerves in this image were downsampled in fiji and then generously thresholded to include any nonzero, resulting in nerves being squashed together, similar to the implementation in the V1 algorithm. This is actually a somewhat necessary step, as the segmentations did have gaps that had to be restored by some kind of recombination. When executing this method on non-downsampled images, another param, diledge, should be set to an integer representing the number of microns to instead dilate nerves, which blows them up morphologically to reestablish broken connectivity. For example, a value diledge = 8 typically gives similar results to the downsamples present here (which are downsampled 5x in each dimension compared to the original, already downsampled-from-raw images).

#Below - example for calculating network for 5x image - Will not run due to hard memory constraints in results, so do not uncomment, however if you have a pc or env with higher RAM, feel free to download these files and then try to run it from a jupyter notebook. (Use 'pip install nettracer3d' for CPU version- see nettracer3d pypi page for GPU information)
#my_network.calculate_all('/data/SK2/5x/labelled_gloms_2.tif', '/data/SK2/5x/Nerves_2.tif', directory = '/results', search = 10, xy_scale = 4.575, z_scale = 2.917, down_factor = 5, remove_trunk = 1, label_nodes = False) 
#Example of how to calculate the network for 5x. remove_trunk = 1 tells the program to remove the edge trunk a single time. This is essentially the largest edge by volume after the edges have been split up. In the case of 5x images, nerves eminate from the arcuate nerves and end up on almost all glomeruli, so there exists a single edge that touches everything more or less. Removing the trunk lets us explore connections beyond this nerve plexus. Please note that due to memory limitations I was only able to upload one 5x dataset to this directory, as 5x datasets cannot be downsampled much more without causing the nerves to become too low resolution. When run at full resolution, nerves in 5x should not be dilated either for this reason as they are less likely to contain significant gaps from artifacts.

print(f"\nSTATISTICS FROM EXTRACTED NETWORK: (will be saved as community_louvain_network_plot.png: )")
my_network.show_communities_louvain(directory = '/results') #Example of showing network. This will also print a number of stats about the network, for example, the modularity. Modularities tabulated in supplemental table 9 and 13 were obtained this way.

plt.close('all') #Clearing matplotlib


#And then we have a few options to look at the graph but code ocean does not support 
#interactive matplotlib so we can only save the plot as a png in the outputs.
#Normally matplotlib will let you zoom in and out of the graph to look at nodes more closely.

#For exploring post-calculation outcomes without having to redo calculate_all() (especially when processing times are larger), the Network_3D object can be reestablished with:
#my_network = n3d.Network_3D()
#my_network.assemble('/results') #Reassemble the Network_3D object once calculated. This method assumes that the outputs are named exactly the same as created by calculate_all(), and if they are, they will be reincorperated into the object. This way, Network_3D objects can be saved in individual directories for quick analysis without having to recalculate the whole network.

#Drawing network overlays. These methods are used to transpose network information back into a 3D visualization for a number of figures and movies in this paper.
print(f"\nDRAWING THE INDICES OF NODES IN THE NETWORK (will be saved as 'labelled_node_indices.tif) AND DRAWING THE NETWORK CONNECTIONS (will be saved as 'drawn_network.tif)")
my_network.draw_node_indices(directory = '/results') #indices of nodes
my_network.draw_network(directory = '/results') #connections between nodes

print(f"\nGenerating the degree distribution for this network (Outputs will be a graph - degree_plot.png; and a spreadsheet - degree_dist.xlsx')")
my_network.degree_distribution(directory = '/results') #Get the degree distribution of the network, used to get data for Supplemental Table 10

plt.close('all') #Clearing matplotlib

print(f"\nGenerating random equivalent network:")
my_network_random = n3d.Network_3D() #Initialize a new network for an equivalent rand network
my_network_random.network = my_network.assign_random() #This is how we can compare a random network to a biological one of equivalent node/edge count, to compare degree distributions for example, as shown in Figure 4 or Supplemental Table 9.
print(f"RANDOM NETWORK STATISTICS (will be saved as community_label_propogation_network_plot.png): ")
my_network_random.show_communities(directory = '/results') #Note, normally I would use show_communities_louvain() here for equivalent community detection methods, but in this case, the earlier output file will get overwritten so I am using a seperate community detection method

plt.close('all') #Clearing matplotlib

print(f"\nIsolating connected network subcomponents:")
print(f"LARGEST ISOLATED CONNECTED NETWORK COMPONENT STATS (will be saved as network_plot.png): ")
largest_network = n3d.Network_3D()
largest_network.network = my_network.isolate_connected_component(gen_images = False, key = None, directory = 'results', full_edges = None) #gen_images can be set to true to return masks for the involved nodes. Set full edges to '/data/SK2/20x/3/Nerves_13-1_downsampled.tif' (assuming we are still using the network for SK2_13 as calculated above) and it will extract the masks of the nerves associated with only the isolated node masks. 'key' can be set to an integer value for a node in the network to isolate the component that node is attached to rather than just the largest one, which is the default behavior. For example, largest components were used prior to getting modularities for data in Table 9, or isolated 3D masks from connected components can also be seen in Figure 4a.
largest_network.show_network(directory = '/results') #Again, I am using a seperate community detection method to not overwrite my outputs since there is no interactve matplotlib in Code Ocean

plt.close('all') #Clearing matplotlib

print(f"\nISOLATING POSSIBLE MOTHER GLOMERULI FROM THIS NETWORK: ")
#isolating mother gloms
mothers = my_network.isolate_mothers(directory = '/results', ret_nodes = True) #This method will return a network of only mothers when ret_nodes is set to True. Set 'ret_nodes' to False and it will print a few types of masks pertaining to the mother nodes and return a dictionary of nodes instead, however I did not enable this here to avoid clogging up the results directory. This was used to predict possible mother gloms for Supplemental Table 17.
print(f"Predicted potential mother gloms: {list(mothers.nodes)}")



#The following methods are not involved with NetTracer3D but were used for more straightforward morphological analysis

from measuring_data import duct_grapher as dg
from measuring_data import glom_grapher as gg
from measuring_data import nerve_densities_5x as nd5
from measuring_data import nerve_densities_20x as nd20
from measuring_data import cell_counter as cc
from measuring_data import counter
from measuring_data import convex_3D as c3d
from measuring_data import nerve_densities_2D as nd2d

print(f"\nCOUNTING GLOMERULI IN THIS IMAGE:") 
counter.count_vals('/data/SK2/20x/13/labelled_downsampled_Gloms_for_networks_sk2_13.tif') #Typically used to count glomeruli, mainly, such as in Supplemental Table 11. Params are just an image with split or watershedded objects.

print(f"\nGENERATING THE CONVEX HULL AROUND GLOMERULI IN THIS IMAGE, will be saved as 'convex_hull_of_sample.tif'")
c3d.convex_main('/data/SK2/20x/13/labelled_downsampled_Gloms_for_networks_sk2_13.tif', 4.485, 2.325, '/results') #Used to get the convex hull around objects, such as glomeruli, that had to be restricted to an abstraction of the cortex for densities, such as shown in Supplemental Table 11. This method specfically provides information about the convex hull that is a surrogate measurement for cortical volume, for example. Params are a binary mask (of glomeruli mainly), the xy-scale, the z-scale, output directory.

print(f"\nAssessing density of ducts through cortical space in image SK2_20x_FOV5, will be saved as a graph - 'duct_density_vertical_distribution.png'; and a spreadsheet - 'y_depth_ducts.xlsx'") 
dg.duct_graph('/data/SK2/20x/5/SK2_20x_5_binary_ducts.tif', '/data/SK2/20x/5/SK2_20x_5_sample_mask.tif', 4.57, '/results') #Used to analyze the densities of collecting ducts with respect to the y-axis (with the sample oriented cortex-> medulla in that axis). Supplemental Table S5. Params are a binary mask (of ducts presumably), a binary mask of the entire sample (in 5x, thresholding was used to obtain a mask of the sample to use as a bounding volume. In this case, this is being demoed in 20x, so I am using a completely white array), an xy-scale, and an output directory.

plt.close('all') #Clearing matplotlib

print(f"\nAssessing cellular densities per glomeruli in image SK2_20x_FOV5, will be saved as a spreadsheet - 'glom_cells_output.xlsx'")
cc.calculate_cells('/data/SK2/20x/5/labelled_downsampled_gloms_for_networks_sk2_5.tif', '/data/SK2/20x/5/SK2_20x_5_labelled_cells.tif', 4.57, 2.33, '/results') #Used to calculate cell densities/counts for individual 20x glomeruli. (Supplemental table 15). Note that this was usually not applied on all glomeruli in a volume, but instead, those near the surface. This was because the dapi laser became weaker deeper into the tissue. Glomeruli for analysis were manually picked out in Imaris and masking over the cell masks with the glomeruli to use. Params are a set of split/watershedded glom masks, pre-glomerularlly masked cell masks, the xy-scale, the z-scale, and an output directory.

print(f"\nAssessing density of glomeruli through cortical space in image SK2_20x_FOV13, will be saved as a graph for density of number of gloms per vertical slice - 'unique_gloms_vertical_distribution.png'; and a graph for volumetric density - 'density_gloms_vertical_distribution,png'; in addition to two spreadsheets corresponding the graphs - 'y_depth_gloms_unique.xlsx' and 'y_depth_gloms.xlsx'") 
gg.graph_gloms('/data/SK2/20x/13/labelled_downsampled_Gloms_for_networks_sk2_13.tif', 4.485, '/results') #Used to calculate the density of glomeruli in the cortex with respect to the y-axis (with samples being oriented cortex -> medulla in the y axis). Supplemental Table 4. This method generates a convex hull around the glomeruli to act as a surrogate for cortical volume when calculating density. Params are a binary glom masks, an xy-scale, and an output directory.

plt.close('all') #Clearing matplotlib

print(f"\nAssessing density of nerves surrounding glomeruli compared to cortical depth in image SK2_20x_FOV13, will be saved as a graph - 'nerve_densities_around_gloms_vertical_distribution.png'; and a spreadsheet - 'y_depth_nerves.xlsx'") 
nd5.nerve_graph('/data/SK2/20x/13/labelled_downsampled_Gloms_for_networks_sk2_13.tif', '/data/SK2/20x/13/Nerves_13-1_downsampled.tif', 4.485, 2.325, 4, 2, '/results') #Used to calculate the density of nerves around glomeruli with respect to cortical depth (slicing through the y-axis) for 5x images specifically. Supplemental table 6. To normalize comparisons of densities between different samples, all samples were resampled to having the same number of voxels to enable nerve skeletons to represent the same magnitudes, more or less. In our case, the lowest 5x dims were used for these params (corresponding to the dims in SK1 - not shown here). This meant all other samples were upsampled until their voxels represented an equivalent magnitude to those in SK1. Params are a binary glomerulus mask, a binary nerve mask (these were pre-skeletonized in FIJI typically in this case), the xy-scale, the z-scale, the xy-scale of the highest res image, the z-scale of the highest res image, an output directory.

plt.close('all') #Clearing matplotlib

print(f"\nAssessing density of nerves surrounding glomeruli as a bulk measurement for SK2_20x_FOV13, will be saved as a text file - 'nerve_density.txt'") 
nd20.nerve_graph('/data/SK2/20x/13/labelled_downsampled_Gloms_for_networks_sk2_13.tif', '/data/SK2/20x/13/Nerves_13-1_downsampled.tif', 4.485, 2.325, 4, 2, '/results')
#Used to calculate the density of nerves around glomeruli in bulk for 20x images specifically. Supplemental table 7. To normalize comparisons of densities between different samples, all samples were resampled to having the same number of voxels to enable nerve skeletons to represent the same magnitudes, more or less. In our case, the lowest 20x dims were used for these params (not shown here). This meant all other samples were upsampled until their voxels represented an equivalent magnitudes. Params are a binary glomerulus mask, a binary nerve mask (these were pre-skeletonized in FIJI typically in this case), the xy-scale, the z-scale, the xy-scale of the highest res image, the z-scale of the highest res image, an output directory.

plt.close('all') #Clearing matplotlib

print(f"\nAssessing density of nerves surrounding glomeruli in bulk in 2D confocal image DM6, will just print the output here:") 
nd2d.nerve_twodim_calc('/data/CONFOCAL_MASKS/DM_CKD2/DM6_glomeruli.tif','/data/CONFOCAL_MASKS/DM_CKD2/DM6_nerveskele.tif', 0.62, 10)
#Used to calculate the density of nerves around glomeruli with respect to cortical depth (slicing through the y-axis) for 2D confocal images specifically. Supplemental table 14. Samples did not need to be normalized for comparison here, as they all had the same resolutions of 0.62 microns/pixel. Params are a binary glomerulus mask, a binary nerve skeleton (these were pre-skeletonized in FIJI), the xy-scale, an output directory.

print(f"\nAssessing density of nerves surrounding glomeruli one at a time (for individual glom comparisons) in image SK2_13, will save the output as 'edge_node_quantity.xlsx'") 
from NetTracer3D_V2 import morphology as mpg
mpg.quantify_edge_node('/data/SK2/20x/13/labelled_downsampled_Gloms_for_networks_sk2_13.tif', '/data/SK2/20x/13/Nerves_13-1_downsampled.tif', search = 10, xy_scale = 4.485, z_scale = 2.325, cores = 1, directory = '/results')
#Used to calculate the density of nerves around individual glomeruli in an image. Supplemental table 17. Samples did not need to be normalized for comparison here, as they were only ever compared to other glomeruli in the same image for the purposes of this analysis. Params are a binary glomerulus mask, a binary nerve mask (these were pre-skeletonized in FIJI), the amount of microns to search outwards, the xy-scale, the z-scale, a param to control whether space within the glomeruli ought to be searched for nerves (it was for 5x images, it was not for 20x images; nerves do not actually visibly enter glomeruli but at 5x res, they can overlap somewhat), an output directory.

print(f"\nAssessing angles of glomeruli in image SK14_15, will save the outputs as a spreadsheet, 'angles_of_gloms.xlsx'; and as an image of rays that demonstrate the angle of the glom, 'glomerulus_angle_rays.tif'") 
from measuring_data import angle_extractor as ae

ae.angle_calculation('/data/SK14/20x/15/binary_gloms_SK14_20x_15.tif', '/data/SK14/20x/15/vascular_poles_SK14_20x_15.tif', '/results') #Used to calculate the angles of glomeruli in an image. Vascular poles had to be manually labelled in Imaris. For this image specfically, not all glomeruli were included from the image due to the huge density of glomeruli in neonatal kidneys - instead about 1/3 in this image were randomly sampled. Params are a split/watershedded glomeruli mask, a mask of binarized markers on their vascular poles, an output directory.