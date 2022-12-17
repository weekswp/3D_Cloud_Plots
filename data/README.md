This folder contains both the data that feeds into the analyses and the results of the analyses. The contents of the relevant folders will be discussed here.

The format of the directory is temperature/composition/... where the ... is where the relevant directories come into play. The reasoning behind this format is so that the analysis can be easily scaled to multiple compositions/temperatures. In our case, for the sake of example, everything will be within the 1450K/Cu65Zr35 directory.

Feeding into Analyses:

1.) clusters: this directory contains the local clusters from the PPM/HDBSCAN analysis that are needed to make the cloud plots. The files in this directory are of the ".xyz" format.

2.) results: this directory contains the results files comparing each of the files in the clusters directory to all of the other files in the clusters directory. These files are of the format "*.xyz.json" or readable python dictionaries.

3.) affinities: this directory contains the "affinity" results from the PPM/HDBSCAN analysis.

Results of the Analysis:

1.) global_xyz: created from ../3D_Plot_Script/read_alignments.py. See code for meaning.

2.) 3D_Plots: created from ../3D_Plot_Script/generate_cloud_plots.py. See code for meaning.

3.) kmeans_plots: created from ../3D_Plots_Script/kmeans_plots.py. See code for meaning.

4.) kmeans_centroid_plots: created from ../3D_Plo_Scripts/kmeans_plots.py. See code for meaning.