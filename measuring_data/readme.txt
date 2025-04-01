cell_counter: calculates cell densities between two masks (cell mask, and object containing cells mask); used
for glom cell densities in this experiment. Returns cell counts and glom volumes as well.

convex_3D: Used to create convex hull around a binary mask. Returns the convex hull volume, volume of binary mask,
and relative density of binary mask in volume. Used to estimate 'cortical volume' from glomeruli masks in this experiment.

counter: Counts labelled objects in an array. Used to estimate glomerulus count for a sample at 5x in this experiment,
from the watershedded glom masks.

duct_grapher: Graphs collecting duct density along the longitidunal axis, comparing it to its full sample mask. Used for
duct densities distribution in this experiment.

glom_grapher: Graphs collecting glom density along the longitidunal axis, comparing it to its convex hull (to estimate cortex boundaries). Used for
glom densities distribution in this experiment.

nerve_densities_5x: calculates automatic nerve densities for an object and nerve skeleton mask at 5x. Resolution normalization corresponds to samples analyzed for
this experiment only. Also returns nerve densities around gloms graphed as a function of distance along the longitudinal axis.

nerve_densities_20x: calculates automatic nerve densities for an object and nerve skeleton mask at 20x. Resolution normalization corresponds to samples analyzed for
this experiment only.

nerve_densities_2D: Calculates automatic nerve densities for the confocal images between an object and nerve skeleton.

angle_extractor: Used to create angle distributions from an inputted binary mask and a second angular marker mask. Also 
will draw the angle vectors for validaton purposes.
