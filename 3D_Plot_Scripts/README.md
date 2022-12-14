This directory contains the python scripts for the analyses.

The "read_alignments.py" script reads in alignments from ../data/temp/comp/results and saves the outputs of the file to ../data/temp/comp/global_xyz.

The "generate_cloud_plots.py" script reads the files from ../data/temp/comp/global_xyz and creates 3D "cloud" plots for each HDBSCAN group to ../data/temp/comp/cloud_plots.

The "kmeans_plots.py" script reads the files from ../data/global_xyz and performs KMeans analysis on the 3D plots and then saves the KMeans clustered plots to ../data/temp/comp/kmeans_plots. Finally, the code creates "centroid" plots capturing relevant information such as the density of the kmeans groups and the occupation probability of the sites.

For clarity the "temp/comp/" architecture was used because we performed this analysis on 13 compositions and 3 temperatures and splitting it in this way allows for clear and obvious segregation of the resultant files from the various investigated systems.