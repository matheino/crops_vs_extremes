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


def import_functions(path):
    sys.path.insert(0, path+'research/crop_failures/scripts/crop_failures')
    from general_functions import to_growing_season
    from general_functions import get_crop_and_irrig_list
    from soil_moisture_data_ERA5_mpi import interpolate_gleam_sm
    
    return to_growing_season, get_crop_and_irrig_list, interpolate_gleam_sm


def main(path, gs, comm, size, rank):

    # import 'to_growing_season' and 'get_crop_and_irrig_list' functions
    # from general_functions.py
    to_growing_season, get_crop_and_irrig_list, interpolate_gleam_sm = import_functions(path)
        
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
        sm_gleam = combine_annual_data(path, interp_points, year, 'SMsurf', latitude_new, longitude_new, crop, irrig, gs, interpolate_gleam_sm)
        os.chdir(path+'research/crop_failures/data/sm_data_jan2021')
        pickle.dump(sm_gleam, lzma.open('soil_moisture_gleam_'+irrig+'_'+crop+'_'+str(year)+'_gs'+str(gs)+'.pkl.lzma', 'wb' ) ) 
        del sm_gleam
        
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
    # main(path, 'real', comm, size, rank)
    
    
    
    

 
