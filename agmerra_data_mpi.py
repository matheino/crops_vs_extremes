from mpi4py import MPI
import sys
import pickle
import lzma
import netCDF4 as nc
import os
import numpy as np
import math
import time
import scipy.interpolate


def interpolate_T(path, interp_points, year, var_str, latitude_new, longitude_new):
    
    os.chdir(path+'data/agmerra')
    
    # open netcdf temperature data
    raw_data = nc.Dataset('AgMERRA_'+str(year)+'_'+var_str+'.nc4', mode = 'r', format = 'NETCDF4')
    
    # get latitude and longitude of the imported data, flip latitude data as it needs to be ascending
    lat_orig = np.flip(raw_data['latitude'][:])
    lon_orig = raw_data['longitude'][:]
    
    # create output variable for the temperature data interpolated to 0.5deg resolution
    var_interpolated = np.zeros((raw_data['time'][:].shape[0], latitude_new.shape[0], longitude_new.shape[0]))
    
    # loop through each day of the year
    for day in range(0, raw_data['time'][:].shape[0]):
        
        # get data temperature grid for each day, flip it so that it aligns with lat_orig
        var_temp = np.flip(raw_data[var_str][day,:,:], axis = 0)
        var_temp = np.ma.filled(var_temp,np.nan) # fill masked cells with np.nan
        
        # linearly interpolate temperature to 0.5deg resolution
        var_interp_temp = scipy.interpolate.interpn((lat_orig,lon_orig), var_temp, interp_points, method = 'linear').reshape(360,720)
        var_interpolated[day,:,:] = var_interp_temp[np.newaxis,:,:] # save the interpolated value into the output variable
    
    return var_interpolated


def calculate_daily_values(T_min_daily, T_amp_daily, sin_interval, T_intervals):
    
    # sinosodially interpolate daily temperature between minimum and maximum temperature
    T_dist_daily = (T_min_daily+T_amp_daily*sin_interval).reshape(sin_interval.shape[0], -1)
    
    # initialize daily output variable showing number of days in different temperature bins
    T_bin_data_daily = np.zeros((T_amp_daily.shape[0]*T_amp_daily.shape[1], T_intervals.shape[0]-1))
    
    # remove nans from the data for quicker computations
    not_nan = np.all(~np.isnan(T_dist_daily), axis = 0)
    T_dist_daily = T_dist_daily[:, not_nan].T
    
    # digitize the daily distribution data according to T_intervals
    # digitize function assigns each cell a value that correspond to a
    # temperature bin (in T_intervals)
    digitized = np.digitize(T_dist_daily, T_intervals, right = True)

    for i in np.unique(digitized):
        # calculate number of days in each bin and fill data to initialized output variable
        daily_vals_stkd_boolean_sum = np.sum(digitized == i, axis = 1)
        T_bin_data_daily[not_nan,i-1] = daily_vals_stkd_boolean_sum
    
    # reshape output data to mapped form
    T_bin_data_daily = T_bin_data_daily.reshape(T_amp_daily.shape[0], T_amp_daily.shape[1], T_intervals.shape[0]-1)
    
    # divide output with the length of the sin_intervals vector, so that the unit is day
    T_bin_data_daily = T_bin_data_daily / sin_interval.shape[0]
    
    return T_bin_data_daily


def import_functions(path):
    sys.path.insert(0, path+'research/crop_failures/scripts/crop_failures')
    from general_functions import to_growing_season
    from general_functions import get_crop_and_irrig_list
    
    return to_growing_season, get_crop_and_irrig_list


def main(run_location, gs, comm, size, rank):
    
    # import 'to_growing_season' and 'get_crop_and_irrig_list' functions
    # from general_functions.py
    to_growing_season, get_crop_and_irrig_list = import_functions(path)
 
    # check that process is running correctly for each MPI run
    print('Rank:',rank,'Temperature: process started')
    sys.stdout.flush()

    years = np.linspace(1981,2010,30).astype(int) # vecotor having a value for each year
    
    # create a vector that has values from -20 to 60 with 0.1 spacing
    # change first and last values to -inf and inf for indexing purposes
    T_intervals = np.linspace(-20,60,(60-(-20))*10 + 1)
    T_intervals[0] = -np.inf # change first cell to -inf
    T_intervals[-1] = np.inf # change last cell to inf
    
    
    # create a sin function to interpolate max and min temperature to 10 daily values
    # max of sin_interval should be approximately 2, as difference between max and min
    # is twice the amplitude
    sin_interval = np.sin(np.linspace(0,2*math.pi,10)+math.pi)+1
    sin_interval = np.tile(sin_interval[:,np.newaxis,np.newaxis],(1,360,720))
    
    # initialize latitude and longitude vectors corresponding to 0.5deg resolution
    latitude_new = np.linspace(89.75,-89.75,360)
    longitude_new = np.linspace(0.25,359.75,720)
    
    # create a 0.5deg latitude-longitude mesh based to be used in the interpolation later
    interp_mesh = np.array(np.meshgrid(longitude_new, latitude_new))
    interp_points = np.rollaxis(interp_mesh, 0, 3).reshape(-1, 2) # lat-long coordinates as columns of the matrix
    interp_points = np.flip(interp_points, axis = 1)        
    
    # obtain  crop and irrig set-up in question based on the rank and size of the MPI run
    crop_list, irrig_list = get_crop_and_irrig_list(rank, size)
    crop = crop_list[0]
    irrig = irrig_list[0]
    
    year = years[0]
    
    # initialize lists for annual aggregates
    T_avg_mean = []
    T_year_avg = []
    P_gs_sum = []
    P_year_sum = []
#    wind_mean = []
#    wind_max = []
    
    os.chdir(path+'research/crop_failures/data/temp_precip_wind_data_jan2021') # set output path
    
    if rank == 0:
        pickle.dump(T_intervals, lzma.open('temperature_bins.pkl.lzma', 'wb' ) ) # export intervals data
    
    # loop through all years
    for year in years:
                
        start_t_yr = time.time()
        
        def combine_annual_data(path, interp_points, year, var_str, latitude_new, longitude_new, crop, irrig, gs):
            
            # stack two years of daily data
            var_t1 = interpolate_T(path, interp_points, year-1, var_str, latitude_new, longitude_new)           
            var_t2 = interpolate_T(path, interp_points, year, var_str, latitude_new, longitude_new)
            var = np.vstack((var_t1, var_t2))
            
            # shift latitude origo by 180 degrees
            var = np.roll(var,np.sum((longitude_new < 180)), axis = 2)
            
            # number of days per year
            days_t1 = var_t1.shape[0]
            
            # for each cell, change days that are outside the growing season to nan
            var_gs = to_growing_season(var, crop, irrig, path, days_t1, gs)
            
            # remove those days, where there is no growing season weather data in any grid cell
            var_gs  = var_gs[~np.all(np.isnan(var_gs),axis = (1,2)), :, :]
            
            return var_gs
        
        # obtain interpolated (to 0.5deg) data for daily average, maximum and maximum temperature
        # as well as precipitation and wind speed for two years and stack those into a single variable
        T_max = combine_annual_data(path, interp_points, year, 'tmax', latitude_new, longitude_new, crop, irrig,  gs)
        T_min = combine_annual_data(path, interp_points, year, 'tmin', latitude_new, longitude_new, crop, irrig, gs)
        T_avg = combine_annual_data(path, interp_points, year, 'tavg', latitude_new, longitude_new, crop, irrig, gs)
        T_year = combine_annual_data(path, interp_points, year, 'tavg', latitude_new, longitude_new, crop, irrig, '365')
        P_gs = combine_annual_data(path, interp_points, year, 'prate', latitude_new, longitude_new, crop, irrig, gs)
        P_year = combine_annual_data(path, interp_points, year, 'prate', latitude_new, longitude_new, crop, irrig, '365')
#        wind = combine_annual_data(path, interp_points, year, 'wndspd', latitude_new, longitude_new, crop, irrig, gs)

        # here, when sinusodially interpolated, amplitude is half of the difference between max and min temperatures
        T_amplitude = ((T_max-T_min)/2)
        
        # initialize the output variable showing number of days in different temperature bins
        T_bin_annual = np.zeros( (T_amplitude.shape[1], T_amplitude.shape[2], T_intervals.shape[0]-1) )
        
        # loop across each day
        for day in range(0, T_amplitude.shape[0]):
            
            # select daily matrix from the data
            T_amp_daily = T_amplitude[day,:,:]
            T_min_daily = T_min[day,:,:]
            
            # calculate time in each temperature bin and sum to T_bin_annual, which
            # has the number of days in each interval across the whole greowing season (year)
            T_bin_data_daily = calculate_daily_values(T_min_daily, T_amp_daily, sin_interval, T_intervals)
            T_bin_annual = T_bin_annual + T_bin_data_daily
        
        # export annual binned temperature data
        os.chdir(path+'research/crop_failures/data/temp_precip_wind_data_jan2021')
        pickle.dump(T_bin_annual, lzma.open('temperature_'+irrig+'_'+crop+'_'+str(year)+'_gs'+str(gs)+'.pkl.lzma', 'wb' ) )
        
        # calculate annual values for average temperature, precipitaion and wind speed
        T_avg_mean.append(np.nanmean(T_avg, axis = 0))
        T_year_avg.append(np.nanmean(T_year, axis = 0))
        P_gs_sum.append(np.nansum(P_gs, axis = 0))
        P_year_sum.append(np.nansum(P_year, axis = 0))
#        wind_mean.append(np.nanmean(wind, axis = 0))
#        wind_max.append(np.nanmax(wind, axis = 0))        
        
        print('rank',rank,'year '+str(year)+' took '+str(time.time()-start_t_yr)+' to run')          
        sys.stdout.flush()
    
    # format and temperature, precipitaion and wind speed data for exporting
    T_avg_out = np.stack(T_avg_mean, axis=2)
    T_year_avg_out = np.stack(T_year_avg, axis=2)
    P_gs_out = np.stack(P_gs_sum, axis=2)
    P_year_out = np.stack(P_year_sum, axis=2)
#    wind_mean_out = np.stack(wind_mean, axis=2)
#    wind_max_out = np.stack(wind_max, axis=2)

    # for the precipitation data, set values to nan if they are zero for all years
    P_gs_out[np.all(P_gs_out == 0, axis = 2 ), ...] = np.nan
    P_year_out[np.all(P_year_out == 0, axis = 2 ), ...] = np.nan
    
    # export data
    os.chdir(path+'research/crop_failures/data/temp_precip_wind_data_jan2021')
    pickle.dump(T_avg_out, lzma.open('Tavg_'+irrig+'_'+crop+'_gs'+str(gs)+'.pkl.lzma', 'wb' ) )
    pickle.dump(T_year_avg_out, lzma.open('Tyear_'+irrig+'_'+crop+'_gs'+str(gs)+'.pkl.lzma', 'wb' ) )
    pickle.dump(P_gs_out, lzma.open('P_gs_'+irrig+'_'+crop+'_gs'+str(gs)+'.pkl.lzma', 'wb' ) )
    pickle.dump(P_year_out, lzma.open('P_year_'+irrig+'_'+crop+'_gs'+str(gs)+'.pkl.lzma', 'wb' ) )
#    pickle.dump(wind_mean_out, lzma.open('wind_mean_'+irrig+'_'+crop+'_gs'+str(gs)+'.pkl.lzma', 'wb' ) )
#    pickle.dump(wind_max_out, lzma.open('wind_max_'+irrig+'_'+crop+'_gs'+str(gs)+'.pkl.lzma', 'wb' ) )

    print('AgMERRA rank',rank,'finished')          
    sys.stdout.flush()
        

if __name__== "__main__":
    
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
