# -*- coding: utf-8 -*-
import sys
import netCDF4 as nc
import os
import numpy as np
import scipy.interpolate
import time
#import matplotlib.pyplot as plt
import pickle
import lzma
import calendar


def interpolate_ERA5_sm(path, interp_points, year, var_str, latitude_new, longitude_new, gs_bool):
    
    # define years also here, so that no need to import from function
    years = np.linspace(1981,2010,30).astype(int)
    
    # open netcdf ERA5 soil moisture data
    os.chdir(path+'data/era5/soil_moisture')
    filename = 'soil_moisture_global_'+str(year)+'_30armin.nc'
    nc_raw = nc.Dataset(filename)
    
    # get latitude and longitude of the imported data, flip latitude data as it needs to be ascending
    lat_orig = np.flip(nc_raw['latitude'][:])
    lon_orig = nc_raw['longitude'][:]
    steps2roll_lon = np.sum(nc_raw['longitude'][:] < 180)
    lon_orig = np.roll(lon_orig, steps2roll_lon) # shift longitude origo by 180 degrees 
    lon_orig = (lon_orig + 180) % 360 - 180 
    lon_orig = np.append(lon_orig, 180) # add latitude value 180 to the array
    
    # get coordinate points for each cell in the raster
    # is mask different for every year?
    interp_mesh_orig = np.array(np.meshgrid(lon_orig, lat_orig))
    interp_points_orig = np.rollaxis(interp_mesh_orig, 0, 3).reshape((lat_orig.shape[0]*lon_orig.shape[0], 2))
    interp_points_orig = np.flip(interp_points_orig, axis = 1)        
    
    # extract numpy array from the netcdf
    var_raw = np.flip(nc_raw[var_str][:],axis = 1)
    var_raw = np.ma.filled(var_raw, np.nan) # doesn't actually do anything but transforms the variable to a regular numpy array
    
    # handle no data values (set to nan if any point in the whole daily time series doesn't have a value
    var_mask = []
    
    for year_i in years:
        nc_raw_i = nc.Dataset('soil_moisture_global_'+str(year_i)+'_30armin.nc')
        var_raw_i = np.flip(nc_raw_i[var_str][:],axis = 1)
        var_raw_i = np.ma.filled(var_raw_i, np.nan)
        var_mask.append(np.any(var_raw_i < 0, axis = 0))
    
    var_mask = np.any(np.stack(var_mask, 0), 0)
    var_raw[..., var_mask] = np.nan
    var_raw = np.roll(var_raw, steps2roll_lon, axis = 2) # shift longitude origo by 180 degrees 
    var_raw = np.dstack((var_raw, var_raw[:,:,0][:,:,None])) # add fist longitude column to the end, for interpolation purposes
    
    # create output variable for the temperature data interpolated to 0.5deg resolution
    var_interpolated = np.zeros( (var_raw.shape[0], 360, 720) ) * np.nan
    
    for day in range(0, var_raw.shape[0]):
        
        # get data about growing season
        gs_bool_temp = ~np.isnan(gs_bool[day,:,:].reshape(-1))
        if np.all(gs_bool_temp == False):
            continue
        
        interp_points_temp = interp_points[gs_bool_temp]
        
        # get data grid for each day and vectorize
        var_temp = var_raw[day,:,:].reshape(-1)
        
        # remove nan values from the original data
        not_nan = ~np.isnan(var_temp)
        var_temp = var_temp[not_nan]
        interp_points_orig_temp = interp_points_orig[not_nan]
        
        # linearly interpolate temperature to 0.5deg resolution for cells with growing season during that day
        var_interp_temp_v = scipy.interpolate.griddata(interp_points_orig_temp, var_temp, interp_points_temp, method = 'linear')
        
        # fill the interpolated values to full matrix and export
        var_interp_temp = np.zeros((360*720))*np.nan
        var_interp_temp[gs_bool_temp] = var_interp_temp_v

        var_interpolated[day,:,:] = var_interp_temp.reshape(360,720)[np.newaxis,:,:]

    return var_interpolated


def interpolate_gleam_sm(path, interp_points, year, var_str, latitude_new, longitude_new, gs_bool):
    
    # var_str = 'SMsurf'
    
    # open netcdf gleam soil moisture data
    os.chdir(path+'data/gleam/v3.2a/'+str(year))
    raw_data = nc.Dataset(var_str+'_'+str(year)+'_GLEAM_v3.2a.nc',mode = 'r', format = 'NETCDF4')
    
    # get latitude and longitude of the imported data, flip latitude data as it needs to be ascending
    lat_orig = np.flip(raw_data['lat'][:])
    lon_orig = raw_data['lon'][:]
    
    # get coordinate points for each cell in the raster
    interp_mesh_orig = np.array(np.meshgrid(lon_orig, lat_orig))
    interp_points_orig = np.rollaxis(interp_mesh_orig, 0, 3).reshape((lat_orig.shape[0]*lon_orig.shape[0], 2))
    interp_points_orig = np.flip(interp_points_orig, axis = 1)    

    # extract numpy array from the netcdf and fill nan values
    var_raw = np.flip(np.swapaxes(raw_data[var_str][:],1,2), axis = 1)
    var_raw = np.ma.filled(var_raw, np.nan)

    var_interpolated = np.zeros((var_raw.shape[0], latitude_new.shape[0], longitude_new.shape[0])) * np.nan
    
    for day in range(0,var_raw.shape[0]):
        
        # get data about growing season
        gs_bool_temp = ~np.isnan(gs_bool[day,:,:].reshape(-1))
        if np.all(gs_bool_temp == False):
            continue
        
        interp_points_temp = interp_points[gs_bool_temp]

        # get data grid for each day and vectorize
        var_temp = var_raw[day,:,:].reshape(-1)
        
        # remove nan values from the original data
        not_nan = ~np.isnan(var_temp)
        
        var_temp = var_temp[not_nan]
        interp_points_orig_temp = interp_points_orig[not_nan]
        
        # linearly interpolate temperature to 0.5deg resolution for cells with growing season during that day
        var_interp_temp_v = scipy.interpolate.griddata(interp_points_orig_temp, var_temp, interp_points_temp, method = 'linear')

        # fill the interpolated values to full matrix and export
        var_interp_temp = np.zeros((360*720))*np.nan
        var_interp_temp[gs_bool_temp] = var_interp_temp_v
        
        var_interpolated[day,:,:] = var_interp_temp.reshape(360,720)[np.newaxis,:,:]
                
    return var_interpolated


def import_functions(path):
    sys.path.insert(0, path+'research/crop_failures/scripts/crop_failures')
    from general_functions import to_growing_season
    from general_functions import get_crop_and_irrig_list
    
    return to_growing_season, get_crop_and_irrig_list


def main(path, gs, comm, size, rank):

    # import 'to_growing_season' and 'get_crop_and_irrig_list' functions
    # from general_functions.py
    to_growing_season, get_crop_and_irrig_list = import_functions(path)
        
    # check that process is running correctly for each MPI run
    print('Rank:',rank,'Soil moisture: process started')
    sys.stdout.flush()
    # Initialize variables related to temporal span, crops, and irrigation
    years = np.linspace(1981,2010,30).astype(int)
    
    # initialize latitude and longitude vectors corresponding to 0.5deg resolution
    latitude_new = np.linspace(89.75,-89.75,360)
    longitude_new = np.linspace(-179.75,179.75,720)
    
    # create a 0.5deg latitude-longitude mesh based to be used in the interpolation later
    interp_mesh = np.array(np.meshgrid(longitude_new, latitude_new))
    interp_points = np.rollaxis(interp_mesh, 0, 3).reshape((latitude_new.shape[0]*longitude_new.shape[0], 2))
    interp_points = np.flip(interp_points, axis = 1)        
    
    # obtain  crop and irrig set-up in question based on the rank and size of the MPI run
    crop_list, irrig_list = get_crop_and_irrig_list(rank,size)
    crop = crop_list[0]
    irrig = irrig_list[0]
    year = years[0]
        
    for year in years:

        start_t_yr = time.time()
        
        def combine_annual_data(path, interp_points, year, var_str, latitude_new, longitude_new, crop, irrig, gs, fun):
            
            # get number of days per
            days_t1 = 365 + int(calendar.isleap(year-1))
            days_t2 = 365 + int(calendar.isleap(year))
            
            gs_bool = np.ones(( days_t1 + days_t2, latitude_new.shape[0], longitude_new.shape[0] ))
            gs_bool = to_growing_season(gs_bool, crop, irrig, path, days_t1, gs)
            
            # stack two years of daily data
            var_t1 = fun(path, interp_points, year-1, var_str, latitude_new, longitude_new, gs_bool[:days_t1, ...])           
            var_t2 = fun(path, interp_points, year, var_str, latitude_new, longitude_new, gs_bool[days_t1:, ...])
            var_gs = np.vstack((var_t1, var_t2))
           
            # remove those days, where there is no growing season weather data in any grid cell
            var_gs  = var_gs[~np.all(np.isnan(var_gs),axis = (1,2)), :, :]
            
            return var_gs
        
        # export annual daily soil moisture data
        sm_era = combine_annual_data(path, interp_points, year, 'swvl1', latitude_new, longitude_new, crop, irrig, gs, interpolate_ERA5_sm)
        
        os.chdir(path+'research/crop_failures/data/sm_data_jan2021')
        pickle.dump(sm_era, lzma.open('soil_moisture_era_'+irrig+'_'+crop+'_'+str(year)+'_gs'+str(gs)+'.pkl.lzma', 'wb' ) ) 
        del sm_era
        
        print('rank',rank,'year '+str(year)+' took '+str(time.time()-start_t_yr)+' to run')          
        sys.stdout.flush()


if __name__== "__main__":
    
    from mpi4py import MPI
    
    run_location = {'cluster': '/scratch/work/heinom2/',
                    'local_d': 'D:/work/',
                    'local_c': 'C:/Users/heinom2/'} 

    path = run_location['cluster'] # get root path 
    
    # initialize MPI run
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    # run main script with different (90 and real) growing season setting
    main(path, '90', comm, size, rank)
    main(path, 'real', comm, size, rank)
    
    
    
    

 
