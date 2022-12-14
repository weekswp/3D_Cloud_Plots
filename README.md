# 3D_Cloud_Plots
This repository contains codes related to the generation of 3D cloud plots from aligned 3-dimensional structures. This repository was created in support of the publication TODO. The process of making 3D plots is split into 3 steps.

1.) Read in Point Pattern Matching (PPM) alignment data for HDBSCAN grouping (see https://github.com/weekswp/liquid_structure_analysis) and save as an ".xyz" file with all constitutive structures of the group.

2.) Create 3D "Cloud" plots for each HDBSCAN group to show the spatial distribution of the structures.

3.) Create K-means "centroid" plots to quantify factors such as relative size of k-means groups and "occupation probaility"
