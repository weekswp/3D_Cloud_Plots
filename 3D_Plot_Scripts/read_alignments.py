#This script reads in the data from constituent clusters of each HDBSCAN group 
#and saves the data to global xyz file in the ../data/temp/comp/global_xyz folder. 
#In order to do this, we read in the cluster files (../data/temp/comp/clusters) and the PPM results
#files (../data/temp/comp/results). In the paper associated with this repository, 13 Cu-Zr compositions and 3 temperatures
#were analyzed. For the purposes of this explanation, this has been reduces to a single
#composition and a single temperature.
#
#As a general note, many of the functions used here are covered in greater detail in other github repositories and one is
#directed to these for further inforamtion:
#   https://github.com/weekswp/liquid_structure_analysis
#   https://github.com/paul-voyles/motifextraction
#   https://github.com/spatala/ppm3d

from path import Path
import os
import numpy as np
import json
from scipy import stats
import scipy.spatial
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import matplotlib as mpl
import seaborn as sns

#cwd is the current working directory where the codes are located.
cwd = Path()
#Global variables:
# c: (array) used to tell the code which compositions we want to investigate.
# T: (array) used to tell the code which temperatures we would like to investigate.
# files_already_made: (Boolean) variable that allows one to bypass the most computationally-intensive 
#part of the analysis if it has already been performed by setting variable to True. For clarity, if this variable is set to True,
#this code will essentially do nothing and was installed as a failsafe to ensure that we haven't already done the analysis.

c = ["Cu65Zr35"]
T = ["1450K"]
files_already_made = False
compositional_x_dict = {}
compositional_y_dict = {}
compositional_z_dict = {}

#Reads atomic positions from xyz file
#inputs:
#   f: filename
#returns:
#   data: numpy array of atomic positions from xyz file
def read_xyz(f):
    data = open(f).readlines()
    data.pop(0)  # Number of atoms
    data.pop(0)  # Comment
    data = np.array([[float(x) for x in line.split()[1:]] for line in data])
    return data

#Reads atom types from xyz file
#inputs:
#   f: filename
#returns:
#   data: numpy array of atomic types from xyz file
def read_types(f):
    data = open(f).readlines()
    data.pop(0)
    data.pop(0)
    type_array = []
    for line in data:
        type_array.append(line.split()[0])
    return np.array(type_array)

#Part of the Point Pattern Matching (PPM) process (see above repositories).
def normalize_edge_lengths(coordinates):
    pdists = scipy.spatial.distance.pdist(coordinates)
    mean = np.mean(pdists)
    coordinates /= mean
    return coordinates, mean

#Loads in "affinity" file denoted by filename input (see above repositories for details on affinity creation).
def load_affinity(filename, normalize=True) -> np.ndarray:
    print("Loading {} affinity...".format(filename))
    affinity = np.load(filename)
    if not normalize:
        return affinity
    #print("Making symmetric...")
    affinity = np.minimum(affinity, affinity.T)
    #print("Finished making symmetric!")
    #print("Normalizing...")
    good_affinity = affinity[np.isfinite(affinity) & ~np.isnan(affinity)]
    print("Setting all values above {} to np.inf".format(np.mean(good_affinity) * 5))
    affinity[np.where(affinity > np.mean(good_affinity) * 5)] = np.inf
    good_affinity = affinity[np.isfinite(affinity) & ~np.isnan(affinity)]
    mean = np.mean(good_affinity)
    max_ = np.amax(good_affinity)
    #print("Mean before normalizing: {}".format(mean))
    #print("Max  before normalizing: {}".format(max_))
    affinity[np.where(np.isinf(affinity))] = max_ * 3.0
    affinity[np.where(np.isnan(affinity))] = max_ * 3.0
    print("Setting inf and nan values to {}".format(max_ * 3.0))
    min_val = np.amin(affinity)
    assert np.isclose(min_val, 0)
    #print("Subtracting {}".format(min_val))
    #affinity = affinity - min_val
    print("Dividing by {}".format(mean))
    affinity = affinity / mean
    norm_results = {'set_to_inf_before_dividing': max_ * 3.0, 'divide_by': mean}
    print(norm_results)
    #print("New mean: {}  (should be a bit larger than 1.0)".format(np.mean(affinity)))
    #print("New max:  {}".format(np.amax(affinity)))
    return affinity

#Find motif for the HDBSCAN group associated with input filename where the motif is defined as the cluster that 
# exhibits the lowest average dissimilarity to the other group members. temp and comp also fed in to lead the system to
#the appropriate location for the files.
#variables:
#   motif_dir: (Path) pointer for main directory associated with input temp and comp
#   affinity_dir: (Path) pointer for directory where "affinities" files are.
#   group_list: (array) used to keep track of structures that are in the HDBSCAN group.
#   minimum, pointer: (int) used to track the lowest average dissimilarity in the group.
#   local_diss, local_count: (int) used to track the average dissimilarity of a given structure to the other structures in the group.
#returns:
#   motif: (str) representative "motif" from the HDBSCAN group associated with the input filename.
def find_motifs(temp,comp,filename):
    motif_dict = {}
    motif_dir = Path("../data/"+temp+"/"+comp)
    affinity_dir = Path(motif_dir+"/affinities")
    os.chdir(affinity_dir)
    affinities = load_affinity('combined_affinity.npy')
    os.chdir(affinity_dir+"/refined_indices")
    group_list = []
    with open(filename,"r") as current_group:
        current_group.readline()
        line = current_group.readline()
        while line != "":
            group_list.append(int(line))
            line = current_group.readline()
        current_group.close()
    os.chdir(motif_dir)
    minimum = 1000
    pointer = 0
    #iterate through the group_list and find the structure with lowest average dissimilarity.
    for i in range (0,len(group_list)):
        local_diss = 0.0
        local_count = 0
        for j in range(0,len(group_list)):
            if i != j:
                local_diss += float(affinities[group_list[i]][group_list[j]])
                local_count += 1
        average = round(float(local_diss/local_count),4)
        #If the average value is lower than the smallest one we've seen so far, make it the new minimum and alter the pointer object to 
        #point to the new minimum.
        if average < minimum:
            minimum = average
            pointer = i
    motif = str(group_list[pointer])
    return motif

#Method closely tied to the Point Pattern Matching (PPM) and affinity creation process. See repositories referenced at the top of the file
#for additional information. In short, this method reads in cluster files, reads in results files from the PPM alignment, and then rotates
#the target structure to that of the model such that rotational degrees of freedom are minimized.
#returns:
#   new_coordinates: (numpy array) aligned coordinates for each atom in the structure.
#   new_types: (array) atom types of each atom in the structure
#   a_count: tracks the number of operations that were NOT swapped.
#   b_count: tracks the number of operations that WERE swapped.
def compare(model,target,comp,temp,cluster_dir,results_dir):
    original_dir = os.getcwd()
    a_count = 0
    b_count = 0
    os.chdir(cluster_dir)
    A_model = read_xyz(str(model)+".xyz")
    A_types = read_types(str(model)+".xyz")
    B_target = read_xyz(str(target)+".xyz")
    B_types = read_types(str(target)+".xyz")
    A_model = np.array(A_model.T)
    B_target = np.array(B_target.T)
    A_model,mscale = normalize_edge_lengths(A_model)
    B_target,tscale = normalize_edge_lengths(B_target)
    #swapped variable a necessity for the way that numpy functions for this operation where whichever is larger (model or target)
    #needs to be the results file that is read.
    if B_target.shape[0] > A_model.shape[0]:
        swapped = True
    else:
        swapped = False
    os.chdir(results_dir)
    try:
        if swapped == False:
            f = open(str(target)+".xyz.json")
            desired = str(model)+".xyz"
            A_model_new = A_model
            B_target_new = B_target
            A_types_new = A_types
            B_types_new = B_types
        else:
            f = open(str(model)+".xyz.json")
            desired = str(target)+".xyz"
            A_model_new = B_target
            B_target_new = A_model
            A_types_new = B_types
            B_types_new = A_types
        results_file = json.load(f)
        f.close()
        index = 0
        found = False
        for i in range(len(results_file)):
            if found == True:
                break
            if results_file[i]["model"] == desired:
                index = i
                found = True
        local_dict = results_file[index]
        best_fit = local_dict['aligned_model']
        lc = np.mean(A_model_new, axis=1)
        rc = np.mean(B_target_new, axis=1)  # Centroids
        left = A_model_new - lc.reshape(3, 1)  # Center coordinates at centroids
        right = B_target_new - rc.reshape(3, 1)
        final_centroid = np.mean(best_fit,axis=1)
        fitted = np.array(best_fit)-final_centroid.reshape(3,1)
        if (fitted.shape == A_model_new.shape):
            a_count += 1
            returned_types = np.array(A_types_new).tolist()
        elif (fitted.shape == B_target_new.shape):
            b_count += 1
            returned_types = np.array(B_types_new).tolist()
        else:
            raise AssertionError("shapes of fitted and target arrays not the same")
        if local_dict['inverted'] == True:
            new_coordinates = np.array((-fitted).T).tolist()
        else:
            new_coordinates = np.array(fitted.T).tolist()
        os.chdir(original_dir)
        return new_coordinates,returned_types,a_count,b_count
    except:
        print("Error. No coordinates returned")
        new_coordinates = []
        new_types = []
        os.chdir(original_dir)
        return new_coordinates,new_types,a_count,b_count
    
#This function is time-consuming and traces back the constituent clusters of a group (input local_list) to get the aligned positions.
#inputs:
#   comp: (str) composition of interest
#   temp: (str) temperature of interest
#   local_list: (array) array of constituent atoms in the HDBSCAN group
#   filename: (str) file associated with HDBSCAN group
#   cluster_directory: (Path) clusters location to be fed into compare()
#   results_directory: (Path) results location to be fed into compare()
#variables:
#   model: (str) "motif" structure that we align the rest of the structures to.
#   target: (str) used in the for loop to iterate over the constituent atoms in local_list
#   traced_array: (array) used to keep track of the aligned x,y,z positions to be added to the new_xyz file
#   traced_types: (array) used to keep track of the aligned atom types of the atoms to be added to the new_xyz file.
#   a_count, b_count: (int) used to monitor the alignment process (see compare() method)
#   comparison: (array) aligned positions of structure returned from compare() method.
#   types: (array) atom types of the structure returned from the compare() method.
#returns:
#   new_xyz: (str) large XYZ style string that contains the aligned positions of all constituent structures from the HDBSCAN group.
def traceback(comp,temp,local_list,filename,cluster_directory,results_directory):
    traced_array = []
    traced_types = []
    a_count = 0
    b_count = 0
    original_dir = os.getcwd()
    #This is the motif that are are aligning all of the other structures to. Essentially, we need a "model" that will be held constant
    #for us to then align the other structures to (for target in local_list:).
    model = find_motifs(temp,comp,filename)
    for target in local_list:
        comparison,types,a_count_local,b_count_local = compare(target,model,comp,temp,cluster_directory,results_directory)
        a_count += a_count_local
        b_count += b_count_local
        for i in range(len(comparison)):
            traced_array.append(comparison[i])
            traced_types.append(types[i])
    new_xyz = str(len(traced_array))+"\n\n"
    for i in range(len(traced_array)):
        string = str(traced_types[i])+"  "+str(traced_array[i][0])+"  "+str(traced_array[i][1])+"  "+str(traced_array[i][2])+"\n"
        new_xyz += string
    os.chdir(original_dir)
    print("A Count:  "+str(a_count))
    print("B Count:  "+str(b_count))
    return new_xyz
    
#This file makes the aligned xyz file for a given combination of temp and comp. Input directories used to 
#make sure that the system knows where to go for the relevant files.
def get_aligned_xyz(temp,comp,commonality_directory,affinity_directory,cluster_directory,results_directory,save_directory):
    local_dict = {}
    os.chdir(commonality_directory)
    files = os.listdir(os.getcwd())
    file_array = []
    length_array = []
    #This block of code is used to get a sorted list of the HDBSCAN groups (ranked list from largest to smallest).
    for file in files:
        file_array.append(str(file))
        with open(file,'r') as in_file:
            length = len(in_file.readlines())
            length_array.append(length)
            in_file.close()
    sorted_files = []
    while len(sorted_files) < len(os.listdir(os.getcwd())):
        maximum = max(length_array)
        print(maximum)
        max_index = length_array.index(maximum)
        sorted_files.append(file_array[max_index])
        del file_array[max_index]
        del length_array[max_index]
    os.chdir(affinity_directory+"/refined_indices")
    working = os.getcwd()
    #This is where the global boolean variable comes into play. The block inside of this if statement is the most time consuming
    #part of this code, so setting "files_already_made" to True will allow us to skip over this block of code entirely if we've already done it.
    if files_already_made == False:
        for i in range(len(sorted_files)):
            print("Entering "+str(sorted_files[i]))
            os.chdir(working)
            local_list = []
            #open the file of interest and add all of the constituent clusters to local_list.
            with open(sorted_files[i],'r') as in_file:
                for line in in_file:
                    if line.startswith("#") == False:
                        local_list.append(int(line.strip()))
                in_file.close()
            #Feed this local list into the "traceback()" method.
            new_xyz = traceback(comp,temp,local_list,sorted_files[i],cluster_directory,results_directory)
            os.chdir(save_directory)        
            filename = str(sorted_files[i].strip(".txt"))+".xyz"
            with open(filename,'w') as out_file:
                out_file.write(new_xyz)
                out_file.close()
    return sorted_files

def plot_full(x_array,y_array,z_array,i,temp):
    mu, sigma = 0, 0.1
    if temp == "1900K":
        color = mpl.cm.Reds
    elif temp == "1450K":
        color = mpl.cm.Oranges
    elif temp == "700K":
        color = mpl.cm.Blues
    elif temp == "350K":
        color = mpl.cm.Greens
    x = np.array(x_array)
    y = np.array(y_array)
    z = np.array(z_array)

    xyz = np.vstack([x,y,z])
    density = stats.gaussian_kde(xyz).evaluate(xyz)
    
    idx = density.argsort()
    x, y, z, density = x[idx], y[idx], z[idx], density[idx]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c=density, cmap=color, norm=mpl.colors.Normalize(),vmin=min(density),vmax=max(density))
    filename = "Motif "+str(i+1)+".png"
    plt.title(filename.strip(".png"),fontsize=20)
    plt.xticks(rotation=90)
    plt.yticks(rotation=90)
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position')
    ax.set_xlim3d(-1.0,1.0,0.25)
    ax.set_ylim3d(-1.0,1.0,0.25)
    ax.set_zlim3d(-1.0,1.0,0.25)
    plt.rc('axes', titlesize=30)
    color_labels = np.round(np.linspace(min(density),max(density),15),3)
    cb = fig.colorbar(mpl.cm.ScalarMappable(cmap=color),label='Density')
    cb.ax.set_yticklabels(color_labels)
    image_format = 'svg' # e.g .png, .svg, etc.
    image_name = "Motif "+str(i+1)+".svg"
    plt.savefig(image_name, format=image_format, dpi=1200)
    plt.savefig(image_name.strip(".svg")+".png")
    plt.close()
    fig, axes = plt.subplots(3, 1)
    fig.subplots_adjust(hspace=0.5)
    axes[0]=     sns.distplot(x, kde=True, 
             bins=50, color = 'darkblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4},ax=axes[0])
##    ax1.hist(x,bins=50,label="x",color="b")
    axes[1]=     sns.distplot(y, kde=True, 
             bins=50, color = 'darkgreen', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4},ax=axes[1])

##    ax2.hist(y,bins=50,label="y",color="g")
    axes[2]=     sns.distplot(z, kde=True, 
             bins=50, color = 'darkred', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4},ax=axes[2])

##    ax3.hist(z,bins=50,label="z",color="r")
    axes[0].legend(['x'],loc='upper right')
    axes[1].legend(['y'],loc='upper right')
    axes[2].legend(['z'],loc='upper right')
    axes[2].set_xlabel("Position")
    axes[1].set_ylabel("Density")
    axes[0].set_ylabel("Density")
    axes[2].set_ylabel("Density")
    fig.suptitle("Motif "+str(i+1)+" Histogram")
    image_format = 'svg' # e.g .png, .svg, etc.
    image_name = "Motif "+str(i+1)+" Histogram.svg"
    #plt.show()
    plt.savefig(image_name, format=image_format, dpi=1200)
    plt.savefig(image_name.strip(".svg")+".png")
    plt.close()
    compositional_x_dict["Motif "+str(i+1)]=x
    compositional_y_dict["Motif "+str(i+1)]=y
    compositional_z_dict["Motif "+str(i+1)]=z

def plot_types(x_array,y_array,z_array,tag,i,temp):
    mu, sigma = 0, 0.1 
    if temp == "1900K":
        color = mpl.cm.Reds
    elif temp == "1450K":
        color = mpl.cm.Oranges
    elif temp == "700K":
        color = mpl.cm.Blues
    elif temp == "350K":
        color = mpl.cm.Greens
    x = np.array(x_array)
    y = np.array(y_array)
    z = np.array(z_array)

    size = int(len(x_array))
    xyz = np.vstack([x,y,z])
    density = stats.gaussian_kde(xyz).evaluate(xyz)
    
    idx = density.argsort()
    x, y, z, density = x[idx], y[idx], z[idx], density[idx]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c=density, cmap=color, norm=mpl.colors.Normalize(),vmin=min(density),vmax=max(density))
    filename = "Motif "+str(i+1)+" "+tag+" "+str(size)+" Atoms.png"
    plt.title(filename.strip(".png"),fontsize=20)
    plt.xticks(rotation=90)
    plt.yticks(rotation=90)
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position')
    ax.set_xlim3d(-1.0,1.0,0.25)
    ax.set_ylim3d(-1.0,1.0,0.25)
    ax.set_zlim3d(-1.0,1.0,0.25)
    plt.rc('axes', titlesize=30)
    color_labels = np.round(np.linspace(min(density),max(density),15),3)
    cb = fig.colorbar(mpl.cm.ScalarMappable(cmap=color),label='Density')
    cb.ax.set_yticklabels(color_labels)
    image_format = 'svg' # e.g .png, .svg, etc.
    image_name = "Motif "+str(i+1)+" "+tag+" "+str(size)+" Atoms.svg"
    plt.savefig(image_name, format=image_format, dpi=1200)
    plt.savefig(image_name.strip(".svg")+".png")
    plt.close()
    fig, axes = plt.subplots(3, 1)
    fig.subplots_adjust(hspace=0.5)
    axes[0]=     sns.distplot(x, kde=True, 
             bins=50, color = 'darkblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4},ax=axes[0])
##    ax1.hist(x,bins=50,label="x",color="b")
    axes[1]=     sns.distplot(y, kde=True, 
             bins=50, color = 'darkgreen', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4},ax=axes[1])

##    ax2.hist(y,bins=50,label="y",color="g")
    axes[2]=     sns.distplot(z, kde=True, 
             bins=50, color = 'darkred', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4},ax=axes[2])

##    ax3.hist(z,bins=50,label="z",color="r")
    axes[0].legend(['x'],loc='upper right')
    axes[1].legend(['y'],loc='upper right')
    axes[2].legend(['z'],loc='upper right')
    axes[2].set_xlabel("Position")
    axes[1].set_ylabel("Density")
    axes[0].set_ylabel("Density")
    axes[2].set_ylabel("Density")
    fig.suptitle("Motif "+str(i+1)+" "+tag+" "+str(size)+" Histogram")
    image_format = 'svg' # e.g .png, .svg, etc.
    image_name = "Motif "+str(i+1)+" "+tag+" "+str(size)+" Atoms Histogram.svg"
    #plt.show()
    plt.savefig(image_name, format=image_format, dpi=1200)
    plt.savefig(image_name.strip(".svg")+".png")
    plt.close()

#main function that accomplishes the goal of the code.
#variables:
#   cluster_directory: (Path) leads the system to the cluster files.
#   results_directory: (Path) leads the system to the results files.
#   commonality_directory: (Path) leads system to commonality_groups
#   affinity_directory: (Path) leads system to location of affinities
#   save_directory: (Path) tells the system where to save the outputs of the script.
def main():
    for temp in T:
        for comp in c:
            cluster_directory = Path("../data/"+temp+"/"+comp+"/clusters")
            results_directory = Path("../data/"+temp+"/"+comp+"/results")
            commonality_directory = Path("../data/"+temp+"/"+comp+"/affinities/commonality_groups")
            affinity_directory = Path("../data/"+temp+"/"+comp+"/affinities")
            save_directory = Path("../data/"+temp+"/"+comp+"/global_xyz")
            get_aligned_xyz(temp,comp,commonality_directory,affinity_directory,cluster_directory,results_directory,save_directory)

main()
