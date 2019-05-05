# -*- coding: utf-8 -*-
"""
Created on Sat May  4 18:40:25 2019

@author: tomyamashita
"""

#%%
"""
This module contains a function that redefines a mask of a NetCDF file to be that of a user defined study area

This was originally developed for use with NetCDF files created in ArcGIS
    This function takes a netcdf file created in ArcGIS for Emerging Hot Spot Analysis and changes the mask. 
    The original purpose was to force ArcGIS to include all parts of the study transect in the analysis. 
    The ArcGIS function excludes all locations where no mortalities were found, however this does not mean there is missing data, just that there were 0 mortalities. 
   
"""

def emergingHS_fix(netcdf_file, StudyArea, mod_field, mask_field):
    """
    This function replaces the mask of a NetCDF file with a user defined study area and NA data within the study area to 0
    This can be used when cells of a NetCDF file are incorrectly classified as NA 
    
    Required Packages: 
    fiona, 
    shapely, 
    numpy, 
    netCDF4
    
    Inputs: 
    
    netcdf_file = input netcdf file which needs to be changed
    
    StudyArea = input shapefile which will be used as the study area representing the area where data will be replaced
    
    mod_field = the input field/column/variable that will be modified
    
    mask_field = the input field/column/variable containing the original mask that will be replaced with the study area
    
    Output: 
    This function will modify the input netcdf file. No new files will be created
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
    for i in range(0, modify.shape[0]):
        occur_temp = modify[i] + mask
        occur_temp[occur_temp==-9998] = 1
        occur_temp = occur_temp - mask
        modify[i] = occur_temp

    # Save the newly created values in the original netCDF file
    cube[mask_field][:] = mask
    cube[mod_field][:] = modify
    cube.close()
