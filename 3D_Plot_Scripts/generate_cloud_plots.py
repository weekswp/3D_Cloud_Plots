#This script reads in the data from global xyz files in the ../data/temp/comp/global_xyz folder and makes 3D "cloud" plots for each,
#along with x,y,z, positional histograms and type-specific "cloud" plots (i.e. only Cu atoms, only Zr atoms). 
#Vector images and bmp images of the files are saved in the ../data/temp/comp/3D_Plots folder. In the initial publication, we analyzed
#13 compositions and 3 temperatures. For the sake of simplicity, we have limited the data here to one of each.

import os
from path import Path
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

#cwd is the current working directory where the codes are located.
cwd = Path()
#Global variables:
# c: (array) used to tell the code which compositions we want to investigate.
# T: (array) used to tell the code which temperatures we would like to investigate.

c = ["Cu65Zr35"]
T = ["1450K"]

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

#This function makes the complete 3D cloud plots for the constituent atomic positions of an HDBSCAN group where the positions
#of the atoms have already been calculated and are fed into the function as x_array,y_array, and z_array. Temp is fed in to the
#function to provide some colorimetric differentiation between temperatures and the "i" variable tells the method which HDBSCAN group
#we are doing the calculation on (i.e. most common, least common, etc.).
def plot_full(x_array,y_array,z_array,i,temp,save_directory):
    mu, sigma = 0, 0.1
    if temp == "1900K":
        color = mpl.cm.Reds
    elif temp == "1450K":
        color = mpl.cm.Oranges
    elif temp == "700K":
        color = mpl.cm.Blues
    elif temp == "350K":
        color = mpl.cm.Greens
    else:
        raise AssertionError("Color not defined for the input temperature!")
    #Convert the input arrays into numpy arrays.
    x = np.array(x_array)
    y = np.array(y_array)
    z = np.array(z_array)

    #vstack the coordinate arrays into a 3D array and calculate the Gaussian density of the points.
    xyz = np.vstack([x,y,z])
    density = stats.gaussian_kde(xyz).evaluate(xyz)
    
    idx = density.argsort()
    x, y, z, density = x[idx], y[idx], z[idx], density[idx]

    #At this point, we have everything we need at can start making the plots.
    fig = plt.figure()
    #Add 3D projection.
    ax = fig.add_subplot(111, projection='3d')
    #Plot the data points and add density based fills to each data point where the min and max values are set 
    #to min(density) and max(density), respectively.
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
    #Add color bar to the plot.
    color_labels = np.round(np.linspace(min(density),max(density),15),3)
    cb = fig.colorbar(mpl.cm.ScalarMappable(cmap=color),label='Density')
    cb.ax.set_yticklabels(color_labels)
    image_format = 'svg' # e.g .png, .svg, etc.
    image_name = "Motif "+str(i+1)+".svg"
    #Change to the save_directory and save png and svg images of the plot.
    os.chdir(save_directory)
    plt.savefig(image_name, format=image_format, dpi=1200)
    plt.savefig(image_name.strip(".svg")+".png")
    plt.close()
    #The "cloud" plots are now made and saved. Now, we make the histograms for x, y, and z positions.
    fig, axes = plt.subplots(3, 1)
    fig.subplots_adjust(hspace=0.5)
    axes[0]=     sns.distplot(x, kde=True, 
             bins=50, color = 'darkblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4},ax=axes[0])
    axes[1]=     sns.distplot(y, kde=True, 
             bins=50, color = 'darkgreen', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4},ax=axes[1])

    axes[2]=     sns.distplot(z, kde=True, 
             bins=50, color = 'darkred', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4},ax=axes[2])

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
    #Make sure that we're still in the right directory.
    os.chdir(save_directory)
    #Save the histogram files again as both an svg and png.
    plt.savefig(image_name, format=image_format, dpi=1200)
    plt.savefig(image_name.strip(".svg")+".png")
    plt.close()

#This function is essentially identical to the workflow of the plot_full() function. The sole difference is that
#we feed in a tag variable to tell the system what atom type we're plotting for and that we only feed in the atomic positions
#of atoms associated with that type instead of performing the analysis on ALL of the atoms like we did in plot_full(). See
#plot_full() method for details of the workflow and commented descriptions of what this method is doing.
def plot_types(x_array,y_array,z_array,tag,i,temp,save_directory):
    mu, sigma = 0, 0.1 
    if temp == "1900K":
        color = mpl.cm.Reds
    elif temp == "1450K":
        color = mpl.cm.Oranges
    elif temp == "700K":
        color = mpl.cm.Blues
    elif temp == "350K":
        color = mpl.cm.Greens
    else:
        raise AssertionError("Color not defined for the input temperature!")
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
    os.chdir(save_directory)
    plt.savefig(image_name, format=image_format, dpi=1200)
    plt.savefig(image_name.strip(".svg")+".png")
    plt.close()
    fig, axes = plt.subplots(3, 1)
    fig.subplots_adjust(hspace=0.5)
    axes[0]=     sns.distplot(x, kde=True, 
             bins=50, color = 'darkblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4},ax=axes[0])
    axes[1]=     sns.distplot(y, kde=True, 
             bins=50, color = 'darkgreen', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4},ax=axes[1])

    axes[2]=     sns.distplot(z, kde=True, 
             bins=50, color = 'darkred', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4},ax=axes[2])

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
    os.chdir(save_directory)
    plt.savefig(image_name, format=image_format, dpi=1200)
    plt.savefig(image_name.strip(".svg")+".png")
    plt.close()

#Main function that executes the purpose of the code: make 3D cloud plots and corresponding histograms for each of the global xyz files.
#variables:
#   x_array, y_array, z_array: (array) keeps track of the x,y,z positions that we are reading in from the global xyz file.
#   Zr_x_array, Zr_y_array, Zr_z_array: (array) keeps track of the x,y,z positions of Zr atoms for the type plots.
#   Cu_x_array, Cu_y_array, Cu_z_array: (array) keeps track of the x,y,z positions of Cu atoms for the type plots.

def main():
    for temp in T:
        for comp in c:
            xyz_directory = Path("../data/"+temp+"/"+comp+"/global_xyz")
            save_directory = Path("../data/"+temp+"/"+comp+"/3D_Plots")
            commonality_directory = Path("../data/"+temp+"/"+comp+"/affinities/commonality_groups")
            affinity_directory = Path("../data/"+temp+"/"+comp+"/affinities")
            sorted_files = get_sorted_files(commonality_directory,affinity_directory)
            os.chdir(xyz_directory)
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
                #Load a global xyz file that we want to make the 3D plots for.
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
                #We have now read in the global_xyz_file and all that it left to do is feed the resultant arrays into the plot_full()
                #and plot_types() functions. Note that we only make the type plots if there are more than 250 Cu atoms and 250 Zr atoms
                #and there are less than 85% of the atoms are of the given type.
                #Otherwise, the type plots would look very similar to the complete plots and would serve little informative purpose.
                #If the aforementioned conditions are met, we proceed to make the 3D type plots.
                plot_full(x_array,y_array,z_array,i,temp,save_directory)
                if (len(Zr_x_array) > 250) and (len(Cu_x_array) > 250):
                    if len(Zr_x_array) < ((0.85)*len(x_array)):
                        #Plot_types for Zr
                        plot_types(Zr_x_array,Zr_y_array,Zr_z_array,"Zr",i,temp)
                    if len(Cu_x_array) < ((0.85)*len(x_array)):
                        #Plot_types for Cu
                        plot_types(Cu_x_array,Cu_y_array,Cu_z_array,"Cu",i,temp)
    
if __name__ == "__main__":
    main()
