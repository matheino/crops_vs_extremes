# -*- coding: utf-8 -*-
from mpi4py import MPI
import os
import numpy as np
#import matplotlib.pyplot as plt
import lzma
import pickle
import time
import sys

def import_SM_data(path, src, gs, crop, irrig, year):
    
    # import numpy arrays of daily era5 or gleam soil moisture data during the growing season
    os.chdir(path+'research/crop_failures/data/sm_data_jan2021')
    
    if src == 'era':
        sm_data_gs = pickle.load(lzma.open('soil_moisture_era_'+irrig+'_'+crop.capitalize()+'_'+str(year)+'_gs'+str(gs)+'.pkl.lzma','rb'))
    elif src == 'gleam':
        sm_data_gs = pickle.load(lzma.open('soil_moisture_gleam_'+irrig+'_'+crop.capitalize()+'_'+str(year)+'_gs'+str(gs)+'.pkl.lzma','rb'))
    
    # reshape data so that the the dimensions are ordered lat, lon, time
    sm_data_gs = np.moveaxis(sm_data_gs, 0, -1)
    
    return sm_data_gs


def max_and_min_soil_moisture(path, src, gs, crop, irrig, years):

    # loop across all years    
    for year in years:
        
        # import soil moisture data for a given year
        sm_temp = import_SM_data(path, src, gs, crop, irrig, year)
        
        # calculate minimum and maximum for each grid cell
        sm_data_yrly_max = np.nanmax(sm_temp, axis = 2)
        sm_data_yrly_min = np.nanmin(sm_temp, axis = 2)
        
        # stack the minimum and maximum of each year into a single variable
        if year == years[0]:
            sm_data_max_all_years = sm_data_yrly_max[:,:,np.newaxis]
            sm_data_min_all_years = sm_data_yrly_min[:,:,np.newaxis]

        else:
            sm_data_max_all_years = np.dstack((sm_data_max_all_years,sm_data_yrly_max[:,:,np.newaxis]))
            sm_data_min_all_years = np.dstack((sm_data_min_all_years,sm_data_yrly_min[:,:,np.newaxis]))
        
    # obtain the minimum and maximum of the whole time span for each grid cell
    sm_data_max = np.nanmax(sm_data_max_all_years,axis = 2)
    sm_data_min = np.nanmin(sm_data_min_all_years,axis = 2)
    
    return sm_data_max[:,:,np.newaxis], sm_data_min[:,:,np.newaxis]


def calculate_bin_data(SM_annual, SM_intervals):
    
    # digitize the soil moisture data according to SM_intervals
    # digitize function assigns each cell a value that correspond to a
    # temperature bin (in SM_intervals)
    digitized = np.digitize(SM_annual, SM_intervals, right = True)

    # initialize output variable showing number of days in different soil moisture bins
    SM_bin_data = np.zeros((SM_annual.shape[0], SM_annual.shape[1], SM_intervals.shape[0]-1))

    for i in np.unique(digitized)[:-1]:
        # calculate number of days in each bin and fill data to initialized output variable
        digitized_bool = digitized == i
        
        daily_vals_stkd_boolean_sum = np.sum(digitized_bool, axis = 2)
        SM_bin_data[:,:,i-1] = daily_vals_stkd_boolean_sum

    return SM_bin_data


def import_functions(path):
    sys.path.insert(0, path+'research/crop_failures/scripts/crop_failures')
    from general_functions import get_crop_and_irrig_list
    
    return get_crop_and_irrig_list


def main(path, src, gs, comm, size, rank):
    
    # import 'get_crop_and_irrig_list' functions from general_functions.py
    get_crop_and_irrig_list = import_functions(path)
    
    # check that process is running correctly for each MPI run
    print('Rank:',rank,'Soil moisture standardization: process started')

    years = np.linspace(1981,2010,30).astype(int)    
    
    # obtain  crop and irrig set-up in question based on the rank and size of the MPI run
    crop_list, irrig_list = get_crop_and_irrig_list(rank,size)
    crop = crop_list[0]
    irrig = irrig_list[0]
    
    # create a vector that has values from 0 to 1 with 0.001 spacing
    # change first and last values to -inf and inf for indexing purposes
    SM_intervals = np.linspace(0,1,1001)
    SM_intervals[0] = -np.inf
    SM_intervals[-1] = np.inf
    
    # initialize lists for annual aggregates
    sm_mean = []

    
    os.chdir(path+'research/crop_failures/data/sm_data_jan2021')
    if rank == 0:
        pickle.dump(SM_intervals, lzma.open('soil_moisture_deficit_bins.pkl.lzma', 'wb' ) )
    
    # get minimum and maximum soil moisture for each grid cell
    sm_data_max, sm_data_min = max_and_min_soil_moisture(path, src, gs, crop, irrig, years)
    
    for year in years:
        
        time_st = time.time()
        
        # import soil moisture data for a given year
        sm_data_raw = import_SM_data(path, src, gs, crop,irrig,year)

        # calculate annual values for average soil moisture       
        sm_mean.append(np.nanmean(sm_data_raw, axis = 2))
        
        # min-max scaling for the daily values
        sm_data_anom = (sm_data_max - sm_data_raw)/(sm_data_max - sm_data_min)
        
        # calculate days in each soil moisture bin across the whole greowing season (year)
        sm_data_anom_bins = calculate_bin_data(sm_data_anom, SM_intervals)
        
        # export data
        os.chdir(path+'research/crop_failures/data/sm_data_jan2021')
        pickle.dump(sm_data_anom_bins, lzma.open('soil_moisture_deficit_bins_'+src+'_'+irrig+'_'+crop+'_'+str(year)+'_gs'+str(gs)+'.pkl.lzma', 'wb' ))
        
        del sm_data_raw
        del sm_data_anom
        del sm_data_anom_bins
        
        print('Rank:',rank,'Soil moisture standardization: year '+str(year)+' completed in '+str(time.time()-time_st)+' seconds.')
        sys.stdout.flush()
        
    # format export annual average soil moisture
    sm_mean_out = np.stack(sm_mean, axis=2)

    os.chdir(path+'research/crop_failures/data/sm_data_jan2021')
    pickle.dump(sm_mean_out, lzma.open('soil_moisture_'+src+'_mean_'+irrig+'_'+crop+'_gs'+str(gs)+'.pkl.lzma', 'wb' ) )
    
           
if __name__ == '__main__':
#    main()
    
    run_location = {'cluster': '/scratch/work/heinom2/',
                    'local_d': 'D:/work/',
                    'local_c': 'C:/Users/heinom2/'}
    
    path = run_location['cluster'] # get root path 
    
    # initialize MPI run
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    # run main script with different (90 and real) growing season setting
    main(run_location['cluster'], 'era', '90', comm, size, rank)
    main(run_location['cluster'], 'era', 'real', comm, size, rank)            

    # main(run_location['cluster'], 'gleam', '90', comm, size, rank)
    
    
    
    
    
    