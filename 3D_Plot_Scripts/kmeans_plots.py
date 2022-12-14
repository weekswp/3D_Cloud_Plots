#This script reads in the data from global xyz files in the ../data/temp/comp/global_xyz folder 
#and makes KMeans 3D "cloud" plots for each, along with KMeans "centroid" plots.
#Vector images and png images of the full kmeans cloud plots are saved in the ../data/temp/comp/kmeans_plots folder. 
#Vector images and png images of the kmeans centroid plots are save in the ../data/temp/comp/kmeans_centroid_plots folder.
#In the initial publication, we analyzed 13 compositions and 3 temperatures. For the sake of simplicity, 
#we have limited the data here to one of each.

from path import Path
import numpy as np
import json
from scipy.spatial import distance_matrix
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib as mpl
import seaborn as sns
import os
import math

#cwd is the current working directory where the codes are located.
cwd = Path()
#Global variables:
# c: (array) used to tell the code which compositions we want to investigate.
# T: (array) used to tell the code which temperatures we would like to investigate.

T = ["1450K"]
c = ["Cu65Zr35"]

#This method gets a sorted list of HDBSCAN groups from largest to smallest for the provided locations.
def get_sorted_files(commonality_directory):
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
    return sorted_files

#This function reads in the global_xyz file for each composition and temperature in the arrays at the top of the script and
#saves them into the global_dict object. Notably, this is an alternative approach taken to that of the generate_cloud_plots.py
#that allows us to centralize all of the "file reading" within a single for loop and then store the associated data in a returnable
#dictionary for future use.
#returns:
# global_dict: (dict) dictionary containing the aligned x,y,z coordinates and atom types for each HDBSCAN group.
def make_positional_dict():
    global_dict = {}
    for comp in c:
        comp_dict = {}
        for temp in T:
            xyz_directory = Path("../data/"+temp+"/"+comp+"/global_xyz")
            commonality_directory = Path("../data/"+temp+"/"+comp+"/affinities/commonality_groups")
            sorted_files = get_sorted_files(commonality_directory)
            os.chdir(xyz_directory)
            for i in range(len(sorted_files)):
                local_dict = {}
                filename = sorted_files[i].strip(".txt")+".xyz"
                x_array = []
                y_array = []
                z_array = []
                type_array = []
                with open(filename,'r') as in_file:
                    in_file.readline()
                    in_file.readline()
                    line = in_file.readline()
                    while line != "":
                        line_list = line.split()
                        type_array.append(line_list[0].strip())
                        x_array.append(float(line_list[1]))
                        y_array.append(float(line_list[2]))
                        z_array.append(float(line_list[3]))
                        line = in_file.readline()
                    in_file.close()
                x = np.array(x_array)
                y = np.array(y_array)
                z = np.array(z_array)
                numpy_stacked = np.vstack([x,y,z])
                local_dict["coordinates"] = numpy_stacked
                local_dict["types"] = type_array
                dict_key = "Motif "+str(i+1)+"_"+temp
                comp_dict[dict_key] = local_dict
        global_dict[comp] = comp_dict
    return global_dict

#This is the main function that carries out the primary functionality of the code: making 3D KMeans plots and KMeans centroid
#plots from the global_xyz files.
def main():
    original_dir = cwd
    #Read in all of the global files at once and then save it as "global_dict" for future use.
    global_dict = make_positional_dict()
    kmeans_dict = {}
    json_kmeans_dict = {}
    viridis = mpl.cm.get_cmap('viridis',8)
    for comp in c:
        print(comp)
        comp_dict = global_dict[comp]
        kmeans_comp_dict = {}
        json_kmeans_comp_dict = {}
        for temp in T:
            xyz_directory = Path("../data/"+temp+"/"+comp+"/global_xyz")
            save_directory = Path("../data/"+temp+"/"+comp+"/3D_Plots")
            commonality_directory = Path("../data/"+temp+"/"+comp+"/affinities/commonality_groups")
            affinity_directory = Path("../data/"+temp+"/"+comp+"/affinities")
            os.chdir(cwd+"/"+temp+"/"+comp+"/3D_Plots")
            if os.path.isdir("kmeans") == False:
                os.mkdir("kmeans")
            os.chdir(cwd+"/"+temp+"/"+comp+"/3D_Plots/kmeans")
            for motif in comp_dict.keys():
                print(motif)
                if temp in motif:
                    data = comp_dict[motif]["coordinates"].T
                    maximum = 0.0
                    for i in range(6,15):
                        kmeans = KMeans(n_clusters=i, random_state=0).fit(data)
                        labels = kmeans.labels_
                        centers = kmeans.cluster_centers_
                        silhouette_avg = silhouette_score(data, labels)
                        if silhouette_avg > maximum:
                            best_labels = labels
                            best_centers = centers
                            maximum = silhouette_avg
                    fig=plt.figure()
                    colors = ['red','green','blue','cyan','magenta','orange','olive','gold','black','purple','lightgray','deeppink','skyblue','darkturquoise','tan','forestgreen','salmon','red','green','blue','cyan','magenta','orange','olive','gold','black','purple','lightgray','deeppink','skyblue','darkturquoise','tan','forestgreen','salmon','red','green','blue','cyan','magenta','orange','olive','gold','black','purple','lightgray','deeppink','skyblue','darkturquoise','tan','forestgreen','salmon','red','green','blue','cyan','magenta','orange','olive','gold','black','purple','lightgray','deeppink','skyblue','darkturquoise','tan','forestgreen','salmon']
                    ax=Axes3D(fig)
                    cluster_dict = {}
                    for i in range(len(best_labels)):
                        if best_labels[i] not in cluster_dict.keys():
                            cluster_dict[int(best_labels[i])]= {}           
                            cluster_dict[int(best_labels[i])]["x"] = [data[i][0]]
                            cluster_dict[int(best_labels[i])]["y"] = [data[i][1]]
                            cluster_dict[int(best_labels[i])]["z"] = [data[i][2]]
                            diff = [abs(data[i][0]),abs(data[i][1]),abs(data[i][2])]
                            cluster_dict[int(best_labels[i])]["average"] = [float(sum(diff))/3.0]
                        else:
                            cluster_dict[int(best_labels[i])]["x"].append(data[i][0])
                            cluster_dict[int(best_labels[i])]["y"].append(data[i][1])
                            cluster_dict[int(best_labels[i])]["z"].append(data[i][2])
                            diff = [abs(data[i][0]),abs(data[i][1]),abs(data[i][2])]
                            cluster_dict[int(best_labels[i])]["average"].append(float(sum(diff))/3.0)
                    average_tracker = 100.0
                    stdev_tracker = 100.0
                    average_pointer = 0
                    stdev_pointer = 0
                    for i in range(len(cluster_dict.keys())):
                        distances = []
                        for j in range(len(cluster_dict.keys())):
                            if i != j:
                                distances.append(np.linalg.norm(best_centers[i]-best_centers[j]))
                        distances = np.array(distances)
                        cluster_dict[i]["ave_dist"] = float(np.mean(distances))
                        cluster_dict[i]["std_dist"] = float(np.std(distances))
                        max_point = np.array([max(cluster_dict[i]["x"]),max(cluster_dict[i]["y"]),max(cluster_dict[i]["z"])])
                        min_point = np.array([min(cluster_dict[i]["x"]),min(cluster_dict[i]["y"]),min(cluster_dict[i]["z"])])
                        distance = np.linalg.norm(max_point-min_point)
                        cluster_dict[i]["size"] = distance
                        if (cluster_dict[i]["ave_dist"]) < average_tracker:
                            average_tracker = cluster_dict[i]["ave_dist"]
                            average_pointer = i
                        if (cluster_dict[i]["std_dist"]) < stdev_tracker:
                            stdev_tracker = cluster_dict[i]["std_dist"]
                            stdev_pointer = i
                    pointer = stdev_pointer
                    global_count = 0
                    for i in range(len(cluster_dict.keys())):
                        local_dict = cluster_dict[i]
                        ax.scatter(local_dict["x"],local_dict["y"],local_dict["z"],color=colors[i])
                        global_count += len(local_dict["x"])
                    likelihood_array = []
                    for i in range(len(cluster_dict.keys())):
                      fraction = float(len(cluster_dict[i]["x"]))/float(global_count)
                      if i == pointer:
                        likelihood_array.append(1.0)
                      else:
                        pointer_fraction = (float(len(cluster_dict[pointer]["x"]))/float(global_count))*float(len(cluster_dict.keys()))
                        value = (float(len(cluster_dict[i]["x"]))/float(global_count))*float(len(cluster_dict.keys()))
                        likelihood_array.append(value/pointer_fraction)
                    print(likelihood_array)
                    image_format = 'svg' # e.g .png, .svg, etc.
                    image_name = motif+" kmeans.svg"
                    plt.title(image_name.strip(".svg"),fontsize=20)
                    plt.xticks(rotation=90)
                    plt.yticks(rotation=90)
                    ax.set_xlabel('X Position')
                    ax.set_ylabel('Y Position')
                    ax.set_zlabel('Z Position')
                    ax.set_xlim3d(-1.0,1.0,0.25)
                    ax.set_ylim3d(-1.0,1.0,0.25)
                    ax.set_zlim3d(-1.0,1.0,0.25)
                    plt.savefig(image_name, format=image_format, dpi=1200)
                    plt.savefig(image_name.strip(".svg")+".png")
                    plt.close()
                    fig=plt.figure()
                    colors = ['red','green','blue','cyan','magenta','orange','olive','gold','black','purple','lightgray','deeppink','skyblue','darkturquoise','tan','forestgreen','salmon']
                    ax = fig.add_subplot(111, projection='3d')
                    print("Max liklihood : "+str(max(likelihood_array)))
                    print("Min liklihood : "+str(min(likelihood_array)))
                    for i in range(len(best_centers)):
                        print(cluster_dict[i]["size"])
                        if i != pointer:
                            ax.scatter(best_centers[i][0],best_centers[i][1],best_centers[i][2],s=250.0*cluster_dict[i]["size"],c=likelihood_array[i],cmap=viridis,vmin=min(likelihood_array),vmax=max(likelihood_array))
                        else:
                            ax.scatter(best_centers[i][0],best_centers[i][1],best_centers[i][2],s=250.0*cluster_dict[i]["size"],edgecolor = "black",c=likelihood_array[i],cmap=viridis,vmin=min(likelihood_array),vmax=max(likelihood_array))                          
                    color_labels = np.round(np.linspace(min(likelihood_array),max(likelihood_array),6),3)
                    cb = fig.colorbar(mpl.cm.ScalarMappable(cmap=viridis),label='Probability')
                    cb.ax.set_yticklabels(color_labels)
                    data = str(len(best_centers))+"\n\n"
                    for i in range(len(best_centers)):
                        data += "1 "+str(best_centers[i][0])+" "+str(best_centers[i][1])+" "+str(best_centers[i][2])+"\n"
                    with open(str(motif)+"_centroids.xyz",'w') as out_file:
                        out_file.write(data)
                        out_file.close()
                    image_format = 'svg' # e.g .png, .svg, etc.
                    image_name = motif+" centroids.svg"
                    plt.title(image_name.strip(".svg"),fontsize=20)
                    plt.xticks(rotation=90)
                    plt.yticks(rotation=90)
                    ax.set_xlabel('X Position')
                    ax.set_ylabel('Y Position')
                    ax.set_zlabel('Z Position')
                    ax.set_xlim3d(-1.0,1.0,0.25)
                    ax.set_ylim3d(-1.0,1.0,0.25)
                    ax.set_zlim3d(-1.0,1.0,0.25)
                    plt.savefig(image_name, format=image_format, dpi=1200)
                    plt.savefig(image_name.strip(".svg")+".png")
                    plt.close()
                    kmeans_comp_dict[motif] = np.array(best_centers)
                    json_kmeans_comp_dict[motif] = best_centers.tolist()
        kmeans_dict[comp] = kmeans_comp_dict
        json_kmeans_dict[comp] = json_kmeans_comp_dict
    #This block of code saves the global dictionary as a readable file for later usage.
    os.chdir(cwd)
    json_object = json.dumps(json_kmeans_dict)
    with open("kmeans_dict.json",'w') as f:
        f.write(json_object)
        f.close()
    os.chdir(original_dir)
    return kmeans_dict


main()