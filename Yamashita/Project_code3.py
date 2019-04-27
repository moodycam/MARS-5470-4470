# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 19:44:52 2019

@author: tomyamashita
"""

#%% Non-function version for bug testing

# Import Packages required for the function
import fiona
from shapely.geometry import Polygon, Point
import numpy as np
import netCDF4
import matplotlib.pyplot as plt

# Load Files 
netcdf_file = "Cube.nc"                                    # This is the netcdf created in ArcGIS that needed to be modified
StudyArea = "buffer_shapefiles\\SH100_Buffer_100m.shp"     # This is the area that serves as the new mask for the data

# These 2 defined variables allow the function to be more generalized
mod_field = "OCCURRENCE_SUM_ZEROS"                         # Define the field to modify
mask_field = "OCCURRENCE_SUM_ZEROS_MASK"                   # Define the field containing the original mask


# Load the netcdf file as a netCDF4 object
cube = netCDF4.Dataset(netcdf_file, 'r+')                  # Need to open with 'r+' so that the file can be edited


# Load the analysis field in the netCDF file as an object
modify = np.array(cube[mod_field])                         # Define the field that needs to be modified


# Create copy of modify field for comparison purposes
mod_original = np.copy(modify)                             # This is for comparison and checking if it works. Not necessary in function


# Load the x and y coordinates from the netcdf file
x = np.array(cube["x"])                                    # Converts the x coordinates (longitude) into a numpy array
y = np.array(cube["y"])                                    # Converts the y coordiantes (latitude) into a numpy array


# Load the study area shapefile and modify it for use in the meshgrid
file = fiona.open(StudyArea)                               # Converts the shapefile into a collection datatype
pol = list(file)                                           # Converts the shapefile collection into a list containing a dictionary datatype
poly_data = pol[0]["geometry"]["coordinates"][0]           # File with the geometry of the shapefile
poly = Polygon(poly_data)                                  # Creates a new polygon from the shapefile geometry


# Create a meshgrid from the x/y coordinates arrays
X, Y = np.meshgrid(x, y, indexing = 'xy')                  # Create a meshgrid containing the coordinates of the original data
coords = np.dstack((X, Y))                                 # Combine the X and Y meshgrids into a single stacked grid


# Create array with the correct shape but with all zeros in it
sh = (len(y), len(x))                                      # This is the shape of the data needing modification
mask = np.zeros(sh)                                        # Create a 2D array of 0s of the shape of the data


# Create loop that checks if the points in the grid are within the study area polygon
# It fills the mask with 1s where the study area polygon is and leaves the 0s where it isn't
for a in range(0, coords.shape[0]):
    for b in range(0, coords.shape[1]):
        mask[a,b] = np.array([poly.contains(Point(coords[a,b,:]))])


# For loop that replaces all NaN values within the masked area to 0
# NaN values in netcdf4 are represented as -9999
i = 0
for i in range(0, modify.shape[0]):
    mod_temp = modify[i] + mask
    mod_temp[mod_temp==-9998] = 1
    mod_temp = mod_temp - mask
    modify[i] = mod_temp
    i += 1 


# Save the created objects in the original netCDF file
cube[mask_field][:] = mask                               # Saves the mask into the original netcdf file
cube[mod_field][:] = modify                              # Saves the updated data into the original netcdf file
#cube.close()  Deactivate this part so it doesn't overwrite the original netcdf file while testing

#%% Plots related to the function

# Plot the mask. Not necessary in the function
fig, ax = plt.subplots(figsize = (16,6))
m = ax.pcolormesh(x, y, mask, cmap = "Blues")
ax.set_title("Study area", fontsize = 18)
ax.set_xlabel("X Position (in UTM Coordinates)", fontsize = 14)
ax.set_ylabel("Y Position (in UTM Coordinates)", fontsize = 14)
plt.colorbar(m)

# Plot the original and new grids. Not necessary for the function
colormap = "Reds"
t = 0
fig, ax = plt.subplots(nrows = 2, figsize = (16,8), sharey=True, sharex=True)
m1 = ax[0].pcolormesh(x, y, mod_original[t], cmap = colormap, vmin = -1, vmax = 8)
ax[0].set_title("Original Mortality Occurrence File", fontsize = 16)
#ax[0].set_xlabel("X Position (In UTM Coordinates)", fontsize = 12)
ax[0].set_ylabel("Y Position (In UTM Coordinates)", fontsize = 12)
m2 = ax[1].pcolormesh(x, y, modify[t], cmap = colormap, vmin = -1, vmax = 8)
ax[1].set_title("Updated Mortality Occurrence File", fontsize = 16)
ax[1].set_xlabel("X Position (In UTM Coordinates)", fontsize = 12)
ax[1].set_ylabel("Y Position (In UTM Coordinates)", fontsize = 12)
fig.colorbar(m2, ax = ax, pad = 0.05, orientation = "vertical")
fig.subplots_adjust(top = 0.90, right = 0.75)
fig.suptitle("Comparison original mortality file and updated mortality file", fontsize = 20, x = 0.425)
fig.suptitle_adjust(top = 0.00, left = 0.4)
#fig.colorbar_adjust(left = 0.50)
fig.tight_layout()


#%% Create a function
def emergingHS_fix(netcdf_file, StudyArea, mod_field, mask_field):
    """
    This function takes a netcdf file created in ArcGIS for Emerging Hot Spot Analysis and changes the mask
    The original purpose was to force ArcGIS to include all parts of the study transect in the analysis
    The ArcGIS function excludes all locations where no mortalities were found, however this does not mean there is missing data, just that there were 0 mortalities
    
    netcdf_file = input netcdf file which needs modification
    StudyArea = input shapefile which will be used as the study area where data will be replaced
    mod_field = the input field/column that will be modified
    mask_field = the input variable containing the original mask that will be replaced
    """
    
    # Import Packages required for the function
    import fiona
    from shapely.geometry import Polygon, Point
    import numpy as np
    import netCDF4
    
    # Load the netcdf file as a netCDF4 object
    cube = netCDF4.Dataset(netcdf_file, 'r+')
    
    # Load the analysis field in the netCDF file as an object
    modify = np.array(cube[mod_field])
    
    # Load the x and y coordinates from the netcdf file
    x = np.array(cube["x"])
    y = np.array(cube["y"])
    
    # Load the study area shapefile and modify it for use in the meshgrid
    file = fiona.open(StudyArea)
    pol = list(file)
    poly_data = pol[0]["geometry"]["coordinates"][0]
    poly = Polygon(poly_data)

    # Create a meshgrid from the x/y coordinates arrays
    X, Y = np.meshgrid(x, y, indexing = 'xy')
    coords = np.dstack((X, Y))

    # Create array with the correct shape but with all zeros in it
    sh = (len(y), len(x))
    mask = np.zeros(sh)

    # Create loop that checks if the points in the grid are within the SH100 polygon
    for x in range(0, coords.shape[0]):
        for y in range(0, coords.shape[1]):
            mask[x,y] = np.array([poly.contains(Point(coords[x,y,:]))])
    
    # For loop that modifies NaN values within the mask to 0 for each level of the netcdf file
    i = 0
    for i in range(0, modify.shape[0]):
        occur_temp = modify[i] + mask
        occur_temp[occur_temp==-9998] = 1
        occur_temp = occur_temp - mask
        modify[i] = occur_temp
        i += 1 

    cube[mask_field][:] = mask
    cube[mod_field][:] = modify
    cube.close()

#%% Testing the function

import netCDF4 as nc
import numpy as np

cube_file = "Cube.nc"  # This is the space time cube created by the previous tool
Studyarea = "buffer_shapefiles\\SH100_Buffer_100m.shp"
mod_field = "OCCURRENCE_SUM_ZEROS"
mask_field = "OCCURRENCE_SUM_ZEROS_MASK"

print(nc.Dataset(cube_file))
occ_pre = np.array(nc.Dataset(cube_file)["OCCURRENCE_SUM_ZEROS"])
mask_pre = np.array(nc.Dataset(cube_file)["OCCURRENCE_SUM_ZEROS_MASK"])
emergingHS_fix(cube_file, Studyarea, mod_field, mask_field)
occ_post = np.array(nc.Dataset(cube_file)["OCCURRENCE_SUM_ZEROS"])
mask_post = np.array(nc.Dataset(cube_file)["OCCURRENCE_SUM_ZEROS_MASK"])
