#This script reads in the data from constituent clusters of each HDBSCAN group 
#and saves the data to global xyz file in the ../data/global_xyz folder. 
#In order to do this, we read in the cluster files (../data/clusters) and the PPM results
#files (../data/results). In the paper associated with this repository, 13 Cu-Zr compositions and 3 temperatures
#were analyzed. For the purposes of this explanation, this has been reduces to a single
#composition and a single temperature.
from path import Path
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
# cluster_directory: (Path) leads the system to the cluster files.
# results_directory: (Path) leads the system to the results files.
cluster_directory = Path("../data/clusters")
results_directory = Path("../data/results")
c = ["Cu65Zr35"]
T = ["1450K"]
files_already_made = False
compositional_x_dict = {}
compositional_y_dict = {}
compositional_z_dict = {}

def read_xyz(f):
    data = open(f).readlines()
    data.pop(0)  # Number of atoms
    data.pop(0)  # Comment
    data = np.array([[float(x) for x in line.split()[1:]] for line in data])
    return data

def read_types(f):
    data = open(f).readlines()
    data.pop(0)
    data.pop(0)
    type_array = []
    for line in data:
        type_array.append(line.split()[0])
    return np.array(type_array)

def normalize_edge_lengths(coordinates):
    pdists = scipy.spatial.distance.pdist(coordinates)
    mean = np.mean(pdists)
    coordinates /= mean
    return coordinates, mean

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

#Find motifs for groups of given replicate
def find_motifs(temp,comp,filename):
    motif_dict = {}
    if temp != "1450K":
        affinity_dir = cwd+"/"+temp+"/"+comp+"/affinities"
        motif_dir = cwd+"/"+temp+"/"+comp
    else:
        affinity_dir = "F://CuZr_Motif/1000_atom_simulations/Mendelev/"+comp+"/affinities"
        motif_dir = "F://CuZr_Motif/1000_atom_simulations/Mendelev/"+comp
    os.chdir(affinity_dir)
    affinities = load_affinity('combined_affinity.npy')
    refined_dir = affinity_dir+"/refined_indices"
    os.chdir(refined_dir)
    group_list = []
    with open(filename,"r") as current_group:
        current_group.readline()
        line = current_group.readline()
        while line != "":
            group_list.append(int(line))
            line = current_group.readline()
        current_group.close()
    os.chdir(motif_dir)
    if os.path.isdir("motifs") == False:
        os.mkdir("motifs")
    os.chdir("motifs")
    minimum = 1000
    pointer = 0
    for i in range (0,len(group_list)):
        local_diss = 0.0
        local_count = 0
        for j in range(0,len(group_list)):
            if i != j:
                local_diss += float(affinities[group_list[i]][group_list[j]])
                local_count += 1
        average = round(float(local_diss/local_count),4)
        if average < minimum:
            minimum = average
            pointer = i
    motif = str(group_list[pointer])
    return motif

def compare(model,target,comp,temp):
    original_dir = os.getcwd()
    a_count = 0
    b_count = 0
    if temp != "1450K":
        cluster_dir = "Z:/Active/metallic-glass/Working_Directories/Mendelev_Potential_CuZr_Temp_Analysis_Corrected/"+temp+"/"+comp+"/Combined/data/clusters"
        results_dir = "Z:/Active/metallic-glass/Working_Directories/Mendelev_Potential_CuZr_Temp_Analysis_Corrected/"+temp+"/"+comp+"/Combined/data/results"
    else:
        cluster_dir = "Z:/Active/metallic-glass/Working_Directories/Mendelev_Potential_CuZr_PPM_HDBSCAN/"+comp+"/Combined/data/clusters"
        results_dir = "Z:/Active/metallic-glass/Working_Directories/Mendelev_Potential_CuZr_PPM_HDBSCAN/"+comp+"/Combined/data/results"
    os.chdir(cluster_dir)
    A_model = read_xyz(str(model)+".xyz")
    A_types = read_types(str(model)+".xyz")
    B_target = read_xyz(str(target)+".xyz")
    B_types = read_types(str(target)+".xyz")
    A_model = np.array(A_model.T)
    B_target = np.array(B_target.T)
    A_model,mscale = normalize_edge_lengths(A_model)
    B_target,tscale = normalize_edge_lengths(B_target)
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
        #print(new_coordinates)
        os.chdir(original_dir)
        return new_coordinates,returned_types,a_count,b_count
    except:
        print("Error. No coordinates returned")
        new_coordinates = []
        new_types = []
        os.chdir(original_dir)
        return new_coordinates,new_types,a_count,b_count
    
def traceback(comp,temp,local_list,filename):
    traced_array = []
    traced_types = []
    a_count = 0
    b_count = 0
    original_dir = os.getcwd()
    model = find_motifs(temp,comp,filename)
    for target in local_list:
        comparison,types,a_count_local,b_count_local = compare(target,model,comp,temp)
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
    
def get_aligned_xyz(temp,comp,commonality_directory,affinity_location):
    local_dict = {}
    os.chdir(commonality_directory)
    files = os.listdir(os.getcwd())
    file_array = []
    length_array = []
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
    os.chdir(affinity_location+"/affinities/refined_indices")
    working = os.getcwd()
    if files_already_made == False:
        for i in range(len(sorted_files)):
            print("Entering "+str(sorted_files[i]))
            os.chdir(working)
            local_list = []
            with open(sorted_files[i],'r') as in_file:
                for line in in_file:
                    if line.startswith("#") == False:
                        local_list.append(int(line.strip()))
                in_file.close()
            new_xyz = traceback(comp,temp,local_list,sorted_files[i])
            os.chdir(cwd+"/"+temp+"/"+comp)
            if os.path.isdir("3D_Plots") == False:
                os.mkdir("3D_Plots")
            os.chdir(cwd+"/"+temp+"/"+comp+"/3D_Plots")            
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

def make_plots():
    for temp in T:
        for comp in c:
            if temp != "1450K":
                commonality_directory = cwd+"/"+temp+"/"+comp+"/commonality_groups"
                affinity_location = cwd+"/"+temp+"/"+comp
            else:
                commonality_directory = "F://CuZr_Motif/1000_atom_simulations/Mendelev/"+comp+"/commonality_groups"
                affinity_location = "F://CuZr_Motif/1000_atom_simulations/Mendelev/"+comp           
            sorted_files = get_aligned_xyz(temp,comp,commonality_directory,affinity_location)
            os.chdir(cwd+"/"+temp+"/"+comp+"/3D_Plots")
            for i in range(len(sorted_files)):
                x_array = []
                y_array = []
                z_array = []
                Zr_x_array = []
                Zr_y_array = []
                Zr_z_array = []
                Cu_x_array = []
                Cu_y_array = []
                Cu_z_array = []
                with open(sorted_files[i].strip(".txt")+".xyz") as in_file:
                    in_file.readline()
                    in_file.readline()
                    line = in_file.readline()
                    while line != "":
                        line_list = line.split()
                        x_array.append(float(line_list[1]))
                        y_array.append(float(line_list[2]))
                        z_array.append(float(line_list[3]))
                        if line_list[0].strip() == "Zr":
                            Zr_x_array.append(float(line_list[1]))
                            Zr_y_array.append(float(line_list[2]))
                            Zr_z_array.append(float(line_list[3]))
                        elif line_list[0].strip() == "Cu":
                            Cu_x_array.append(float(line_list[1]))
                            Cu_y_array.append(float(line_list[2]))
                            Cu_z_array.append(float(line_list[3]))                    
                        line = in_file.readline()
                    in_file.close()
                plot_full(x_array,y_array,z_array,i,temp)
                if (len(Zr_x_array) > 250) and (len(Cu_x_array) > 250):
                    if len(Zr_x_array) < ((0.85)*len(x_array)):
                        plot_types(Zr_x_array,Zr_y_array,Zr_z_array,"Zr",i,temp)
                    if len(Cu_x_array) < ((0.85)*len(x_array)):
                        plot_types(Cu_x_array,Cu_y_array,Cu_z_array,"Cu",i,temp)
##            os.chdir(cwd+"/"+temp+"/"+comp)
##            with open('x_positions.json','w') as fp:
##                json.dump(compositional_x_dict,fp,indent=4)
##                fp.close()
##            with open('y_positions.json','w') as fp:
##                json.dump(compositional_y_dict,fp,indent=4)
##                fp.close()
##            with open('z_positions.json','w') as fp:
##                json.dump(compositional_z_dict,fp,indent=4)
##                fp.close()                
            compositional_x_dict.clear()
            compositional_y_dict.clear()
            compositional_z_dict.clear()
def main():
    make_plots()
    
main()
