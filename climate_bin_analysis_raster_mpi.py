# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import netCDF4 as nc
import xarray as xr
import lzma
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import gzip
import rasterio
# from sklearn.inspection import plot_partial_dependence, partial_dependence
from sklearn.inspection._partial_dependence import _partial_dependence_brute
from sklearn.utils.extmath import cartesian

from sklearn.model_selection import train_test_split
from scipy.signal import detrend

from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

import shap

def import_ray_crop_data(path, crop):
    
    # set up paths to yield and harvested areas data sets
    path_in = path+'data/deepaks_data/2020'
    path_y = os.path.join(path_in, crop.capitalize()+'_yield_1970To2013')
    path_ha = os.path.join(path_in, crop.capitalize()+'_harvestedarea_1970To2013')
    
    # create file lists for later use and sort files based on the year
    # (the only characters that change in the filenames)
    file_list_y = sorted(os.listdir(path_y))
    file_list_ha = sorted(os.listdir(path_ha))
    
    # create function for data extraction
    def extract_data(file, path):
        
        os.chdir(path)
        
        # open netcdf data, either gzipped or not
        if file.endswith('.gz'):
            unzipped_file = gzip.open(file)
            nc_data = nc.Dataset('dummy', mode='r', memory=unzipped_file.read())
        else:
            nc_data = nc.Dataset(file, mode = 'r')
        
        # get numpy array data from the netcdf
        lat = nc_data['latitude'][:].squeeze()
        lon = nc_data['longitude'][:].squeeze()
        var = nc_data['Data'][:].squeeze()
        
        # format the data so that it's the same way
        if lat[0] < lat[1]:
            lat = np.flip(lat)
            var = np.flip(var, axis = 0)            
        if lon[0] > lon[1]:
            lon = np.flip(lon)
            var = np.flip(var, axis = 1)
        
        # save variables to a python dictionary
        data_dict = {'data': var,
                   'lat': lat,
                   'lon': lon,
                   'filename': file}
        
        return data_dict
 
    # extract crop yield and harvested areas data for each year
    y_data = [extract_data(file, path_y) for file in file_list_y if 'areaweightedyield' in file]
    ha_data = [extract_data(file, path_ha) for file in file_list_ha if 'harvestedarea' in file]
    
    # Check that latitutde and longitude is same for all data           
    lat_test1 = all([(y['lat'] == ha['lat']).all() for y, ha in zip(y_data, ha_data)])
    lon_test1 = all([(y['lon'] == ha['lon']).all() for y, ha in zip(y_data, ha_data)])
    lat_test2 = all([(y['lat'] == y_data[0]['lat']).all() for y, ha in zip(y_data, ha_data)])
    lon_test2 = all([(y['lon'] == ha_data[0]['lon']).all() for y, ha in zip(y_data, ha_data)])
    
    if lat_test1 and lon_test1 and lat_test2 and lon_test2:
        None
    else:
        print('ray crop data: error! difference in lat and lon')
    
    # get crop yield data out from the dictionary list
    y_out = np.array([y['data'] for y in y_data])
    y_year = np.array([int(''.join(filter(str.isdigit, y['filename']))[0:4]) for y in y_data])
    
    # sort crop yield data based on year, should already be in the correct order, but just to make sure
    y_out = y_out[np.argsort(y_year),:,:]
    y_year = y_year[np.argsort(y_year)]
    
    # get crop yield data out from the dictionary list
    ha_out = np.array([ha['data'] for ha in ha_data])
    ha_year = np.array([int(''.join(filter(str.isdigit, ha['filename']))[0:4]) for ha in ha_data])

    # sort harvested areas data based on year, should already be in the correct order, but just to make sure
    ha_out = ha_out[np.argsort(ha_year),:,:]
    ha_year = ha_year[np.argsort(ha_year)]
    
    # extract latitude and longitude data as numpy array
    lat = np.array(y_data[0]['lat'])
    lon = np.array(y_data[0]['lon'])
    
    # put data into correct shape
    y_out = y_out.transpose(1,2,0)
    ha_out = ha_out.transpose(1,2,0)
    
    # change crop yield values above 25 to nan (threshold based on e-mail exchange with Deepak Ray)
    y_out[y_out > 25] = np.nan
    
    # change harvested areas data to nan in cells where yield data is nan
    ha_out[np.isnan(y_out)] = np.nan
    
    # put crop yield data into xarray format
    yield_xarray = xr.DataArray(y_out,
                            dims = ('lat','lon','time'),
                            coords={'lat': lat,
                                    'lon': lon,
                                    'time': y_year
                                    }).to_dataset(name = 'yield')
    
    
    # calculate a 5-year running mean of the data
    yield_5yr_mean_xarray = yield_xarray.rolling(time = 5, center = True).mean(skipna = True)
    
    # de-trend the crop yield data by subtracting a 5-year running mean, and dividing with the same value
    y_dtrnd_xarray = ((yield_xarray - yield_5yr_mean_xarray) / yield_5yr_mean_xarray).rename({'yield':'detrended_yield'})
           
    # put harvested areas data into xarray format
    ha_xarray = xr.DataArray(ha_out,
                            dims = ('lat','lon','time'),
                            coords={'lat': lat,
                                    'lon': lon,
                                    'time': ha_year
                                    }).to_dataset(name = 'harvested_area')
    
    # merge xarray arrays into a xarray dataset
    xarray_out = xr.merge((yield_xarray.sel(time = slice(1981,2009)), y_dtrnd_xarray.sel(time = slice(1981,2009)), ha_xarray.sel(time = slice(1981,2009))))
    
    # Plot data for testing
    # import cartopy.crs as ccrs

    # plot_specs = {'transform': ccrs.PlateCarree(),
    #               'robust': True}
    
    # xarray_out['detrended_yield'].sel(time = 2009).plot(ax = plt.axes(projection=ccrs.Robinson()), **plot_specs);plt.show()

    # xarray_out['harvested_area'].sel(time = 2009).plot(ax = plt.axes(projection=ccrs.Robinson()), **plot_specs);plt.show()
    
    return xarray_out


def import_iizumi_crop_data(path, crop):
    
    # Use major cropping system for maize and rice (as only a single harvest season is considered in this study)
    if crop == 'maize' or crop == 'rice':
        crop_str = crop+'_major'
    else:
        crop_str = crop
    
    # create a file listing of the folder with the data
    path_y = path+'data/iizumi_2020/gdhy_v1.2_v1.3_20190128/'+crop_str
    
    # sort data (by year, as that's the only thing varying in the file names)
    file_list = sorted(os.listdir(path_y))
    
    
    os.chdir(path_y)
    years = []
    
    # loop through the list with file names
    for i, file_name in enumerate(file_list,0):
        
        # open the netcdf data
        nc_data = nc.Dataset(file_name, mode = 'r')
        
        # extract crop yield, latitude and longitude data from the detcdf
        var = nc_data['var'][:]
        lon = nc_data['lon'][:]
        lat = nc_data['lat'][:]
        
        # change negative values (missing data) to nan
        var[var < 0] = np.nan
        
        
        # transform the data to same coordinates as the other data 
        var = np.array(np.flip(var, 0)) # flip data along the latitude dimension
        lat = np.flip(lat) 

        var = np.roll(var, np.sum(lon < 180)) # shift longitude origo by 180 degrees
        lon = np.roll(lon, np.sum(lon < 180))
        lon = ((lon + 180) % 360) - 180
        
        # save information about the year of the extracted data
        y_year = int(file_name[6:10])
        years.append(y_year)
        
        # check that everything is in order with the latitude and longitude dimensions
        if i > 0:
            if np.all(lat == lat_old) and np.all(lon == lon_old):
                None
            else:
                print('iizumi crop data: error! difference in lat and lon')            
        
        lon_old = lon.copy()
        lat_old = lat.copy()
        
        # save the variable as a numpy array
        if i == 0:
            y_out = var[:,:,np.newaxis]
        else:
            y_out = np.dstack((y_out, var[:,:,np.newaxis]))
            
    years = np.array(years) # list to numpy array
    
    # put crop yield data into xarray format
    yield_xarray = xr.DataArray(y_out,
                            dims = ('lat','lon','time'),
                            coords={'lat': lat,
                                    'lon': lon,
                                    'time': years
                                    }).to_dataset(name = 'yield')
    
    
    # calculate a 5-year running mean of the data
    yield_5yr_mean_xarray = yield_xarray.rolling(time = 5, center = True).mean(skipna = True)
    
    # de-trend the crop yield data by subtracting a 5-year running mean, and dividing with the same value
    y_dtrnd_xarray = ((yield_xarray - yield_5yr_mean_xarray) / yield_5yr_mean_xarray).rename({'yield':'detrended_yield'})
    
    # obtain irrigated and rainfed harvested areas from MIRCA 2000
    mirca_ha_rfc, mirca_ha_irc = import_mirca(path, crop, mask = False)
    
    # calculate total harvested areas, and change zeros to nan
    ha_tot = mirca_ha_rfc + mirca_ha_irc
    ha_tot[ha_tot == 0] = np.nan
    ha_out = np.tile(ha_tot[:,:,None], (1,1, years.shape[0])) # use year 2000 data for each year
    
    # put harvested areas data into xarray format
    ha_xarray = xr.DataArray(ha_out,
                        dims = ('lat','lon','time'),
                        coords={'lat': lat,
                                'lon': lon,
                                'time': years
                                }).to_dataset(name = 'harvested_area')
    
    # merge xarray arrays into a xarray dataset
    xarray_out = xr.merge((yield_xarray.sel(time = slice(1984,2009)), y_dtrnd_xarray.sel(time = slice(1984,2009)), ha_xarray.sel(time = slice(1984,2009))))

    # import cartopy.crs as ccrs

    # plot_specs = {'transform': ccrs.PlateCarree(),
    #               'robust': True}
    
    # xarray_out['detrended_yield'].sel(time = 2009).plot(ax = plt.axes(projection=ccrs.Robinson()), **plot_specs);plt.show()

    # xarray_out['yield'].sel(time = 2009).plot(ax = plt.axes(projection=ccrs.Robinson()), **plot_specs);plt.show()
    # xarray_out['harvested_area'].sel(time = 2009).plot(ax = plt.axes(projection=ccrs.Robinson()), **plot_specs);plt.show()
    
    return xarray_out


def import_mirca(path, crop, mask = True):
    
    # import harvested areas data (MIRCA2000).
    os.chdir(os.path.join(path, 'data/MIRCA2000/harvested_area_grids'))
    
    # define crop id variables that align with those used in mirca
    crop_ids = {'maize':'crop02',
                'rice':'crop03',
                'soybean':'crop08',
                'wheat':'crop01'}
    
    # file names for irrigated and rainfed harvested area for the crop in question
    file_name_irc = 'annual_area_harvested_irc_'+crop_ids[crop]+'_ha_30mn.asc.gz'
    file_name_rfc = 'annual_area_harvested_rfc_'+crop_ids[crop]+'_ha_30mn.asc.gz'
    
    # import the mirca harvested areas data
    with gzip.open(file_name_irc, 'rb') as file_in:            
        with rasterio.open(file_in) as raster_in:
            ha_per_crop = np.squeeze(raster_in.read()) # read raster as numpy array
            ha_per_crop[np.isnan(ha_per_crop)] = 0.0 # change nan values to zero (i.e. no harvested area)
            mirca_ha_irc = ha_per_crop
            raster_in.close()          
        file_in.close()
            
    with gzip.open(file_name_rfc, 'rb') as file_in:            
        with rasterio.open(file_in) as raster_in:
            ha_per_crop = np.squeeze(raster_in.read()) # read raster as numpy array
            ha_per_crop[np.isnan(ha_per_crop)] = 0.0 # change nan values to zero (i.e. no harvested area)
            mirca_ha_rfc = ha_per_crop 
            mirca_meta = raster_in.meta.copy()
            raster_in.close()
        file_in.close()
    
    # if mirca data is not exported for masking purposes, return irrigated and rainfed harvested area grids
    if not mask:
        return mirca_ha_rfc, mirca_ha_irc
    
    # if a rainfed / irrigated crop mask is imported
    elif mask:
        
        mirca_ha_combined = mirca_ha_rfc + mirca_ha_irc
        mirca_ha_combined[mirca_ha_combined == 0] = np.nan
        
        # check if rainfed / irrigated harvested area is more than 90% / 60% of total harvested area
        # for a specific grid cell
        mirca_rfc_mask = (mirca_ha_rfc / mirca_ha_combined) > 0.9
        mirca_irc_mask = (mirca_ha_irc / mirca_ha_combined) > 0.6
        
        # for simpler coding syntax later, create a variable for a no-mask scenario also
        mirca_combined_mask = mirca_ha_combined > 0
        
        mirca_mask = {'rf': mirca_rfc_mask[:,:,np.newaxis],
                      'ir': mirca_irc_mask[:,:,np.newaxis],
                      'combined': mirca_combined_mask[:,:,np.newaxis]}
    
        return mirca_mask
    
    
def import_other_clim_data(dtype, gs, crop,irrig, stdz, path):
    
    if irrig == 'combined':
        
        # import rainfed and irrigated harvested areas grid
        ha_rfc, ha_irc = import_mirca(path, crop, mask = False)
        if 'soil_moisture' in dtype:
            os.chdir(path+'research/crop_failures/data/sm_data_jan2021')
            data_rfc = pickle.load(lzma.open(dtype+'_rf_'+crop.capitalize()+'_gs'+str(gs)+'.pkl.lzma', 'rb'))
            data_irc = pickle.load(lzma.open(dtype+'_ir_'+crop.capitalize()+'_gs'+str(gs)+'.pkl.lzma', 'rb'))
        else:
            os.chdir(path+'research/crop_failures/data/temp_precip_wind_data_jan2021')
            data_rfc = pickle.load(lzma.open(dtype+'_rf_'+crop.capitalize()+'_gs'+str(gs)+'.pkl.lzma', 'rb'))
            data_irc = pickle.load(lzma.open(dtype+'_ir_'+crop.capitalize()+'_gs'+str(gs)+'.pkl.lzma', 'rb'))
        
        # weight the binned temperature and soil moisture data with harvested areas
        data_sum = data_rfc*ha_rfc[:,:,None] + data_irc*ha_irc[:,:,None] # calculate irrigation scenario harvested areas weighted sum of days in each bin
        ha_combined = ha_rfc[:,:,None] + ha_irc[:,:,None] # calculate total (irrigated + rainfed) harvested area
        ha_combined[ha_combined == 0] = np.nan # change zero values to nan in order to not raise errors
        
        data_out = data_sum / ha_combined # calculate harvested area weighted temperature / soil moisture
                
    # if only either rainfed or irrigated irrigation scenario is used, use either of those
    elif irrig == 'rf' or irrig == 'ir':
        if 'soil_moisture' in dtype:
            os.chdir(path+'research/crop_failures/data/sm_data_jan2021')
        else:
            os.chdir(path+'research/crop_failures/data/temp_precip_wind_data_jan2021')

        filename = dtype+'_'+irrig+'_'+crop.capitalize()+'_gs'+str(gs)+'.pkl.lzma'
        data_out = pickle.load(lzma.open(filename,'rb'))        

    mask = np.any(np.isnan(data_out), axis = 2)
    data_out[np.isnan(data_out)] = 0
    
    # if the output is set to be an anomaly calculate the z-score of the data
    if stdz == 'anom':
        data_out = (data_out - np.nanmean(data_out, axis = 2)[:,:,np.newaxis]) / (np.nanstd(data_out, axis  = 2)[:,:,np.newaxis])        

    # if the output is set to be an detrended anomaly, detrend the data and then calculate a z-score
    elif stdz == 'detrended_anom':
        data_out = detrend(data_out, axis = 2, type = 'linear')
        data_out = (data_out - np.nanmean(data_out, axis = 2)[:,:,np.newaxis]) / (np.nanstd(data_out, axis  = 2)[:,:,np.newaxis])        
    
    # mask the data to be exported
    data_out[mask, ...] = np.nan
    
    # plt.imshow(data_out[150:250,300:500,0], clim = (-2,2)); plt.colorbar; plt.show()
    # plt.imshow(data_out[150:250,300:500,1], clim = (-2,2)); plt.colorbar; plt.show()

    # plt.imshow(data_out[150:250,300:500,-1], clim = (-2,2)); plt.colorbar; plt.show()
    # plt.imshow(data_out[150:250,300:500,-2], clim = (-2,2)); plt.colorbar; plt.show()

    return data_out
    

def import_binned_SM_and_T_data(dtype, gs, crop,irrig, year, path):
    
    # import the bin information for temperature or soil moisture
    if dtype == 'temperature':
        os.chdir(path+'research/crop_failures/data/temp_precip_wind_data_jan2021')
        file_info = dtype
        bins = pickle.load(lzma.open(file_info+'_bins.pkl.lzma','rb'))[1:] # since there are 1 more interval than bins, remove the first cell
        bins[-1] = bins[-2] + 0.1 # change last cell from inf to the value it represents
        
    elif 'soil_moisture' in dtype:
        # correct string for the file names defined previously
        if 'era' in dtype:
            file_info = 'soil_moisture_deficit_bins_era'
        elif 'gleam' in dtype:
            file_info = 'soil_moisture_deficit_bins_gleam'

        os.chdir(path+'research/crop_failures/data/sm_data_jan2021')
        bins = pickle.load(lzma.open('soil_moisture_deficit_bins.pkl.lzma','rb'))[1:] # since there are 1 more interval than bins, remove the first cell
        bins[-1] = bins[-2]+0.001 # change last cell from inf to the value it represents
        
    # import the binned temperature or soil moisture data
    bin_data = pickle.load(lzma.open(file_info+'_'+irrig+'_'+crop.capitalize()+'_'+str(year)+'_gs'+str(gs)+'.pkl.lzma','rb'))
    
    return bin_data, bins


def SM_and_T_bins_weighted_w_irrig(dtype, gs, crop, irrig, year, path):
    
    # if using the combined irrigation scenario, combine binned temperature / soil moisture data 
    # from the rainfed and irrigated scenarios
    if irrig == 'combined':
        
        # import rainfed and irrigated harvested areas grid
        ha_rfc, ha_irc = import_mirca(path, crop, mask = False)    
        
        # import temperature / soil moisture data for both irrigation scenarios
        data_rf, bins = import_binned_SM_and_T_data(dtype, gs, crop, 'rf', year, path)
        data_ir, bins = import_binned_SM_and_T_data(dtype, gs, crop, 'ir', year, path)
        
        # weight the binned temperature and soil moisture data with harvested areas
        data_sum = data_rf*ha_rfc[:,:,None] + data_ir*ha_irc[:,:,None] # calculate irrigation scenario harvested areas weighted sum of days in each bin
        ha_combined = ha_rfc[:,:,None] + ha_irc[:,:,None] # calculate total (irrigated + rainfed) harvested area
        ha_combined[ha_combined == 0] = np.nan # change zero values to nan in order to not raise errors
        
        data = data_sum / ha_combined # calculate harvested area weighted number of day sin each temperature / soil moisture bin
        
        data[np.isnan(data)] = 0 # format nan values back to zero
        
    # if only either rainfed or irrigated irrigation scenario is used, import those
    elif irrig == 'rf' or irrig == 'ir':
        data, bins = import_binned_SM_and_T_data(dtype, gs, crop, irrig, year, path)

    return data, bins


def obtain_clim_ref(dtype, gs, crop, irrig, years, perc_max, perc_min, path):
        
    # sum days per bin across all years used in the reference scenario
    for i, year in enumerate(years,0):
        if i == 0:
            data, bins = SM_and_T_bins_weighted_w_irrig(dtype, gs, crop, irrig, year, path)
        else:
            data = data + SM_and_T_bins_weighted_w_irrig(dtype, gs, crop, irrig, year, path)[0]
    
    # calculate the cumulative in number of days per temperature / soil moisture bin
    summed = np.sum(data,2)[:,:,np.newaxis] # for each grid cell, calculate sum across all bins
    summed[summed == 0] = np.nan # if no data exists, assign nan
    data_cumulative = np.cumsum(data,2) / summed # calculate cumulative sum across the bins, and then divide with the total sum of days, to get a proportional value
    
    # create 10% percentile threshold vector
    perc_thresholds = np.linspace(0.0,1.0,11)
    perc_thresholds[0] = -1.0 # change first and last values to -1 and 2 to avoid rounding problems
    perc_thresholds[-1] = 2.0
    
    # digitize the cumulative data for the 10 % thresholds defined above
    data_cumulative_digitized = np.digitize(data_cumulative, perc_thresholds, right = True).astype(float)
    
    # find the corresponding digitization to the minimum and maximum percentiles given as input to the function,
    # also check that the other type of exterme has values for each grid cell
    if perc_min == 0.9 and perc_max == 1.0:   
        dig_v = [10, 1]

    elif perc_min == 0.0 and perc_max == 0.1:
        dig_v = [1, 10]
    
    # find the the percentile threshold for the scenario in question
    data_bool = data_cumulative_digitized == dig_v[0]
    
    # find the percentile threshold for the reference scenario
    data_bool_ref = data_cumulative_digitized == dig_v[1]
    
    # find the areas where percentiles could not be reliably quantified
    data_main = data/years.shape[0]
    data_ref = data/years.shape[0]
    
    if gs == 'real':
        data_main = data_main/data_main.sum(2)[:,:,None]*90
        data_ref = data_ref/data_ref.sum(2)[:,:,None]*90
    
    data_main[~data_bool] = 0
    data_ref[~data_bool_ref] = 0
    
    mask_main = np.all([data_main.sum(2) > 5, data_main.sum(2) < 20], axis = 0)
    mask_ref = np.all([data_ref.sum(2) > 5, data_main.sum(2) < 20], axis = 0)
        
    # mask the binned data for the scenario
    data_bool[~mask_ref, ...] = False
    data_bool[~mask_main,...] = False
    mask_out = data_bool.sum(2) == 0
    
    # calculate rasters for percentile limits
    bin_array = np.tile(bins[np.newaxis,np.newaxis,:],(360,720,1))
    bin_array[~data_bool] = np.nan
    
    bin_max = np.nanmax(bin_array, axis = 2)
    bin_min = np.nanmin(bin_array, axis = 2)
    
    return data_bool, bin_min, bin_max, mask_out


def number_of_days_vs_ref(path, dtype, gs, crop, years, irrig, stdz, perc_max = None, perc_min = None, return_ref = False):
    
    # in order to check if data already exists and import processed data, create name for an output file
    filename = stdz+'_'+dtype+'_'+str(gs)+'_'+crop+'_'+irrig+'_'+str(perc_min)+'_to_'+str(perc_max)+'.pkl.lzma'
    
    # if only the data array is imported and it already exist, import from file
    if filename in os.listdir(os.path.join(path, 'research/crop_failures/data/annual_bins_2022')) and not return_ref:
        os.chdir(os.path.join(path, 'research/crop_failures/data/annual_bins_2022'))
        return pickle.load(lzma.open(filename,'rb'))
    
    # get reference data for the whole time period and percentile limits, defined by perc_min and perc_max     
    ref_bin_bool, bin_min, bin_max, mask = obtain_clim_ref(dtype, gs, crop, irrig, years, perc_max, perc_min, path)
    
    # if the data already exist and also the metadata of the binned data (i.e. bin limits) is imported
    if filename in os.listdir(os.path.join(path, 'research/crop_failures/data/annual_bins_2022')) and return_ref:
        os.chdir(os.path.join(path, 'research/crop_failures/data/annual_bins_2022'))
        return pickle.load(lzma.open(filename,'rb')), bin_min, bin_max
        
    days_summed_out = []
    # loop across all years
    for year in years:
        # import bin data per year
        days_per_bin, bins = SM_and_T_bins_weighted_w_irrig(dtype, gs, crop, irrig, year, path)
        # change all cells that are outside the percentile category defined above to zero
        days_per_bin[~ref_bin_bool] = 0.0
        # sum the number of days in the percentile category during the year in question
        days_summed_out.append(days_per_bin.sum(axis = 2))
        
    days_summed_out = np.stack(days_summed_out, axis = 2)
        
    if 'soil_moisture' in dtype:
        days_summed_out = np.sqrt(days_summed_out)
    
    # if the output is set to be an anomaly calculate the z-score of the data
    if stdz == 'anom':
        days_summed_out = (days_summed_out - np.nanmean(days_summed_out, axis = 2)[:,:,np.newaxis]) / (np.nanstd(days_summed_out, axis  = 2)[:,:,np.newaxis])        

    # if the output is set to be an detrended anomaly, detrend the data and then calculate a z-score
    elif stdz == 'detrended_anom':
        days_summed_out = detrend(days_summed_out, axis = 2, type = 'linear')
        days_summed_out = (days_summed_out - np.nanmean(days_summed_out, axis = 2)[:,:,np.newaxis]) / (np.nanstd(days_summed_out, axis  = 2)[:,:,np.newaxis])                
    
    # set masked areas to nan
    days_summed_out[mask, ...] = np.nan
    
    # mask data in case there are nan values in the data still
    any_is_nan = np.any(np.isnan(days_summed_out), axis = 2)
    days_summed_out[any_is_nan, ...] = np.nan

    # save data for future use    
    os.chdir(os.path.join(path, 'research/crop_failures/data/annual_bins_2022'))
    pickle.dump(days_summed_out, lzma.open(filename, 'wb' ))
    
    if return_ref:
        return days_summed_out, bin_min, bin_max
        
    else:
         return days_summed_out


def raster_to_df(data, years, dtype):
    
    # reshape rasterized data to tabulated format, each row represents one cell
    data_table = data.reshape(360*720,-1)

    # create a cell id variable for each cell
    cell_id = np.arange(0,360*720,1)
    
    # create a pandas dataframe of the data
    if years[0] == None:
        df_long = pd.DataFrame(data_table, columns = [dtype])
        df_long['cell_id'] = cell_id
    
    else:
        df = pd.DataFrame(data_table, columns = years)
        df['cell_id'] = cell_id
        
        # change dataframe format to long, meaning that each row is one observation
        df_long =  pd.melt(df, id_vars = 'cell_id', var_name = 'year', value_name = dtype).sort_values('year')

    return df_long


def import_climate_zone_data(crop_data_is_nan, T_tot, P_tot):
    
    # create climate zoning used in the analyses
    
    # calculate the average temperature and precipitation across the whole time period studied
    T_tot = np.nanmean(T_tot,2)
    P_tot = np.nanmean(P_tot,2)

    # vectorize the data
    T_tot_v = T_tot.reshape(-1)
    P_tot_v = P_tot.reshape(-1)
    
    # check which cells have data
    PT_is_na = np.any([np.isnan(T_tot_v), np.isnan(P_tot_v), crop_data_is_nan], axis = 0)

    # remove cells without data in any of the two data sets (P and T)
    T_tot_v = T_tot_v[~PT_is_na]
    P_tot_v = P_tot_v[~PT_is_na]
    
    # divide temperature into quintiles
    T_perc = np.percentile(T_tot_v, np.linspace(0,100,6))
    T_perc[-1] = T_perc[-1]+1 # increase last value by one to have it included in in the intervalling
    T_tot_vq = np.digitize(T_tot_v, T_perc)
    
    # initialize a climate zone array
    clim_zones_vq = np.zeros_like(T_tot_vq)
    
    # loop across the temperature quintiles
    for cat in np.unique(T_tot_vq):
        
        # obtain the values in the precipitation that belong to the temperature quintile in question
        P_tot_v_temp = P_tot_v[T_tot_vq == cat]
        
        # split values into quintiles based on precipitation
        P_temp_perc = np.percentile(P_tot_v_temp, np.linspace(0,100,6))
        P_temp_perc[-1] = P_temp_perc[-1]+1
        
        # categorize values based on temperature and precipitation
        P_temp_q = np.digitize(P_tot_v_temp, P_temp_perc)+(cat-1)*10
        
        # save those values to the climate zone array
        clim_zones_vq[T_tot_vq == cat] = P_temp_q
    
    # list of data to export
    data_list = [clim_zones_vq, T_tot_v, P_tot_v]
    
    # put the climate zone array, as well as temperature and precipitation arrays
    # to a global map
    data_out = []
    for temp in data_list:      
        temp_out = np.zeros(360*720)
        temp_out[~PT_is_na] = temp
        temp_out[PT_is_na] = np.nan
        temp_out = temp_out.reshape(360,720)
        data_out.append(temp_out)
    
    return tuple(data_out)


def combine_data_to_dataframe(crop_data_yield, crop_data_ha, clim_variables, var_names, T_tot, P_tot, irrig_mask, clim_years):
    
    # obtain information about years with crop yield data (varies between iizumi and ray data sets)
    crop_years = crop_data_yield['time'].values
           
    # de-trended crop yield data as numpy array
    crop_data_dtnd_values = np.copy(crop_data_yield['detrended_yield'].values)
    
    # first mask with irrigation extent
    crop_data_dtnd_values[~irrig_mask[...,0], ...] = np.nan
    any_is_nan_yield = np.any(np.isnan(crop_data_dtnd_values), axis = 2)
    crop_data_dtnd_values[any_is_nan_yield, ...] = np.nan
    
    # crop yield data to tabulated format
    df_combined = raster_to_df(crop_data_dtnd_values, crop_years, 'Crop yield anomaly')
    
    # harvested areas data to tabulated format
    crop_data_ha_values = np.copy(crop_data_ha['harvested_area'].sel(time = crop_years).values)
    crop_data_ha_values[~irrig_mask[...,0], ...] = np.nan
    any_is_nan_ha = np.any(np.isnan(crop_data_ha_values), axis = 2)

    crop_data_ha_values[any_is_nan_ha, ...] = np.nan
    
    # harevested areas data to tabulated format
    df_ha = raster_to_df(crop_data_ha_values, crop_years, 'harvested_area')

    # merge yield and harvested areas dataframes    
    df_combined = df_combined.merge(df_ha, on = ('cell_id', 'year'))

    # combine the climate variables to the dataframe
    for i, var in enumerate(clim_variables, 0):
        df_temp = raster_to_df(var, clim_years, var_names[i])        
        df_combined = df_combined.merge(df_temp, on = ('cell_id', 'year'))
    
    crop_data_is_nan = np.any([any_is_nan_ha.reshape(-1), any_is_nan_yield.reshape(-1)], axis = 0)
    clim_zones = import_climate_zone_data(crop_data_is_nan, T_tot, P_tot)
    
    clim_zone_names = ['climate_zone', 'climzone_temperature','climzone_precipitation']
    # combine the climate zone variables to the dataframe
    for i, var in enumerate(clim_zones, 0):
        df_temp = raster_to_df(var, [None], clim_zone_names[i])        
        df_combined = df_combined.merge(df_temp, on = ('cell_id'))
    
    # drop the row if there's any variable without an observation
    df_combined = df_combined.dropna(how = 'any')

    return df_combined


def model_with_optimal_hyperparameters(X, y, ha, years_all, model_type, N_folds, parameters = None):
    
    # function for exploring the hyperparameter space of the XGBoost model
    
    from sklearn.model_selection import RandomizedSearchCV
    
    # initialize the models
    model_dict = {'XGB': xgb.XGBRegressor(base_score=0.5, booster='gbtree', importance_type = 'gain',
                                          missing=np.nan, n_jobs = 1, nthreads = None,
                                          seed = None),
                  'RF':  RandomForestRegressor() }
    
    model_in = model_dict[model_type]
    
    # if hyperparameterspace is not explored return default model
    if parameters != 'optimal':

        model = model_in.set_params(**parameters)
        
        return model
    
    # define the hyperparamter space explored with the XGBoost model    
    elif parameters == 'optimal' and model_type == 'XGB':

        parameters = {
          'objective':['reg:squarederror'],
          'eta': [0.1],
          'n_estimators': [400],
          'max_depth': [3],
          'min_child_weight': [1],
          'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1],
          'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9, 1],
          'colsample_bylevel': [1],
          'colsample_bynode': [1],
          'gamma': [0, 0.01, 0.02, 0.03, 0.04, 0.05],
          'reg_alpha': [0],
          'reg_lambda': [0.5, 0.75, 1, 1.25, 1.5],
          'max_delta_step': [0],
          'scale_pos_weight': [1],
          'verbosity': [1]
          }
        

    df_scores = pd.DataFrame()
    
    # randomly split years of data to N_folds number of categories
    years_unique = np.unique(years_all)
    np.random.shuffle(years_unique) # shuffle the unique years in the data set with an inplace function
    years_test_shuffle = np.array_split(years_unique, 3)
    
    year_cats = np.zeros_like(years_all)
    year_cats[np.isin(years_all, years_test_shuffle[0])] = 0
    year_cats[np.isin(years_all, years_test_shuffle[1])] = 1
    year_cats[np.isin(years_all, years_test_shuffle[2])] = 2
    
    # use two thirds of the data for model training and a third for evaluating
    # the training progress
    cv_year_cats = year_cats[year_cats != 2]
        
    X_train = X[year_cats != 2, ...]
    y_train = y[year_cats != 2]
    ha_train = ha[year_cats != 2]

    X_val = X[year_cats == 2, ...]
    y_val = y[year_cats == 2]
    
    eval_set = [(X_val, y_val)]
    
    # define dictionary for some fitting parameters of the XGBoost model
    fit_params={'early_stopping_rounds': 40, 
        'eval_set': eval_set,
        'verbose': False}
    
    # conduct five iterations in the random grid search
    n_iter = 5
    
    # create a list inidcating different cross-validation samples
    cv_list = [[cv_year_cats != cv_group, cv_year_cats == cv_group]  for cv_group in np.unique(cv_year_cats)]

    # initialize the randomized grid search 
    xgb_grid = RandomizedSearchCV(model_in,
        parameters,
        cv = cv_list,
        n_jobs = 1,
        return_train_score = True,
        n_iter = n_iter,
        verbose = 0)
      
    # fit randomized grid search to find optimal hyperparamterss§
    xgb_grid.fit(X_train, y_train, **fit_params)
    
    model = xgb_grid.best_estimator_
    
#        print(xgb_grid.best_params_)
    
    return model


def calculate_climate_vs_crop_failures(path,
                                        crop,
                                        X,
                                        y,
                                        ha,
                                        cell_id,
                                        clim_zones,
                                        years_all,
                                        df_ha_2000,
                                        parameters,
                                        clim_zone_id,
                                        var_names,
                                        N_folds,
                                        model_type,
                                        cmap,
                                        plot = True,
                                        N = 100):    
    
    # clim_zone_id = 0
    # create a boolean vector based on clim_zone_id and climate zones
    if clim_zone_id > 0:
        X = X[clim_zones == clim_zone_id, :]
        y = y[clim_zones == clim_zone_id]
        years_all = years_all[clim_zones == clim_zone_id]
        ha = ha[clim_zones == clim_zone_id]
        cell_id = cell_id[clim_zones == clim_zone_id]
        
    # get the size of the total sample
    sample_size = X.shape[0]
    
    # check how many grid cells fall out due to nan in some of the input data sets
    if clim_zone_id == 0:
        print('number of sample points and data points compared to crop_yield data:')
        print(sample_size)
        sys.stdout.flush()
    
    # initialize list for r-squared
    rsq = []
    
    # define the values where partial dependence is calculated
    # and create a list of corresponding coordinates
    cats = np.linspace(-3.0,3.0,25)
    cats_coords = cartesian([cats, cats])  
    
    # define output arrays for calculating the partial dependences
    y_dh_out = np.zeros((N, cats.shape[0], cats.shape[0]))*np.nan
    y_wc_out = np.zeros((N, cats.shape[0], cats.shape[0]))*np.nan
    y_wh_out = np.zeros((N, cats.shape[0], cats.shape[0]))*np.nan
    y_dc_out = np.zeros((N, cats.shape[0], cats.shape[0]))*np.nan
    
    y_h_out = np.zeros((N, cats.shape[0]))*np.nan    
    y_d_out = np.zeros((N, cats.shape[0]))*np.nan
    y_c_out = np.zeros((N, cats.shape[0]))*np.nan
    y_w_out = np.zeros((N, cats.shape[0]))*np.nan
    
    # conduct N number of loops
    for i in range(N):
        
        # if no data, move forward
        if y.shape[0] <= 1:
            rsq.append(np.nan)
            continue
        
        # randomly split years of data to N_folds number of categories
        years_unique = np.unique(years_all)
        np.random.shuffle(years_unique) # shuffle the unique years in the data set with an inplace function
        years_test_shuffle = np.array_split(years_unique, N_folds)
        
        year_cats = np.zeros_like(years_all)
        year_cats[np.isin(years_all, years_test_shuffle[0])] = 0
        year_cats[np.isin(years_all, years_test_shuffle[1])] = 1
        year_cats[np.isin(years_all, years_test_shuffle[2])] = 2
        year_cats[np.isin(years_all, years_test_shuffle[3])] = 3
        
        # initialize temporary lists /  arrays
        y_dh_temp_out = []
        y_wc_temp_out = []
        y_wh_temp_out = []
        y_dc_temp_out = []
        
        y_h_temp_out = []
        y_d_temp_out = []
        y_c_temp_out = []
        y_w_temp_out = []
        
        y_pred_all = np.zeros_like(y)
        
        shap_values_out = []
        pd_eval_out = pd.DataFrame()
        
        # loop across the shuffled year categories
        for j in np.unique(year_cats):
            
            val_j = [3,0,1,2]
       
            # create a boolean array, with true values for those data points (rows), which belong to the years in question
            testing_rows = year_cats == j
            valid_rows = year_cats == val_j[j]
            training_rows = np.all([year_cats != j, year_cats != val_j[j]], axis = 0)
            
            if model_type == 'XGB':
                
                # for fitting the XGBoost model, create three separate data sets for training, and testing 
                # as well as evaluating the performance of the training
                X_train = X[training_rows]            
                y_train = y[training_rows]
                ha_train = ha[training_rows]
                years_train = years_all[training_rows]
                
                y_val = y[valid_rows]
                X_val = X[valid_rows]
                
                X_test = X[testing_rows,:]
                y_test = y[testing_rows]
                
                # create an evaluation set 
                eval_set = [(X_val, y_val)]
                
                # create the model with 
                model = model_with_optimal_hyperparameters(X_train, y_train, ha_train, years_train, model_type, N_folds, parameters)
                                            
                # fit the model, stop fitting the model when the training performance
                # as inidcated by eval_set doesn't improve
                model.fit(X_train, y_train, early_stopping_rounds = 40, eval_set = eval_set, verbose = False)
                
            elif model_type == 'RF':
                
                X_train = X[~testing_rows]            
                y_train = y[~testing_rows]
       
                X_test = X[testing_rows,:]
                y_test = y[testing_rows]
             
                model = RandomForestRegressor(n_estimators = 50, max_depth = None, min_samples_split = 2, min_samples_leaf = 1)
                
                # fit the model
                model.fit(X_train, y_train)

            # use the model to predict crop yield anomaly with the testing variables
            y_pred = model.predict(X_test)
                        
            # fill the predicted yield values to the 
            y_pred_all[testing_rows] = y_pred
            
            # if the size of the training data is larger than 1000, randomly select 1000 data points
            # otherwise, use the whole training data for calculating the partial dependencies
            if X_train.shape[0] > 1000:
                ind2sel = np.random.choice(X_train.shape[0], 1000, replace = False)
                X_eval = X_train[ind2sel,:]    
            else:
                X_eval = X_train
            
            if clim_zone_id == 0 and plot:
            # calculate shap values
                X_eval_pd = pd.DataFrame(X_eval, columns = var_names)
                shap_values = shap.TreeExplainer(model, feature_perturbation = 'tree_path_dependent').shap_values(X_eval_pd)
                
                # accumulate shap from different training sets
                shap_values_out.append(shap_values)
                pd_eval_out = pd_eval_out.append(X_eval_pd)
            
            # calculate the partial dependecies from the model, for each point defined in the cats_coords variable
            # numbering of the inspected variables - (anomaly in) hot days: 0, dry days: 1, cold days: 2, wet days: 3
            y_dh_temp = _partial_dependence_brute(model, cats_coords, [0,1], X_eval, response_method = 'auto')[0].reshape(cats.shape[0], cats.shape[0])
            y_wc_temp = _partial_dependence_brute(model, cats_coords, [2,3], X_eval, response_method = 'auto')[0].reshape(cats.shape[0], cats.shape[0])
            y_wh_temp = _partial_dependence_brute(model, cats_coords, [0,3], X_eval, response_method = 'auto')[0].reshape(cats.shape[0], cats.shape[0])
            y_dc_temp = _partial_dependence_brute(model, cats_coords, [2,1], X_eval, response_method = 'auto')[0].reshape(cats.shape[0], cats.shape[0])
            
            y_h_temp = _partial_dependence_brute(model, cats[:, None], [0], X_eval, response_method = 'auto')[0].reshape(-1)
            y_d_temp = _partial_dependence_brute(model, cats[:, None], [1], X_eval, response_method = 'auto')[0].reshape(-1)
            y_c_temp = _partial_dependence_brute(model, cats[:, None], [2], X_eval, response_method = 'auto')[0].reshape(-1)
            y_w_temp = _partial_dependence_brute(model, cats[:, None], [3], X_eval, response_method = 'auto')[0].reshape(-1)
            
            # a function that changes values that are not within the training data limits to nan
            def filter_nodata(data, cats, mat, x1, x2 = None):
               
                # find minimum and maximum values in the data
                mat_min1 = mat[:,x1].min()
                mat_max1 = mat[:,x1].max()
                
                if x2 != None:     
                    mat_min2 = mat[:,x2].min()
                    mat_max2 = mat[:,x2].max()
                    
                    # change values that are larger than data maximum or smaller than the data minium to nan
                    data[~np.all([mat_min1 <= cats, mat_max1 >= cats], axis = 0),:] = np.nan
                    data[:, ~np.all([mat_min2 <= cats, mat_max2 >= cats], axis = 0)] = np.nan
                    
                else:
                    data[~np.all([mat_min1 <= cats, mat_max1 >= cats], axis = 0)] = np.nan
                
                return data
            
            # add the filtered partial dependency values to a list
            y_dh_temp_out.append( filter_nodata(y_dh_temp, cats, X_train, 0, 1) )
            y_wc_temp_out.append( filter_nodata(y_wc_temp, cats, X_train, 2, 3) )
            y_wh_temp_out.append( filter_nodata(y_wh_temp, cats, X_train, 0, 3) )
            y_dc_temp_out.append( filter_nodata(y_dc_temp, cats, X_train, 2, 1) )
            
            y_h_temp_out.append( filter_nodata(y_h_temp, cats, X_train, 0) )
            y_d_temp_out.append( filter_nodata(y_d_temp, cats, X_train, 1) )
            y_c_temp_out.append( filter_nodata(y_c_temp, cats, X_train, 2) )
            y_w_temp_out.append( filter_nodata(y_w_temp, cats, X_train, 3) )   
            
        # calculate mean partial dependence values across the annual splits, and save it to the output numpy array
        y_dh_out[i,:,:] = np.nanmean(np.stack(y_dh_temp_out, 0), 0)
        y_wc_out[i,:,:] = np.nanmean(np.stack(y_wc_temp_out, 0), 0)
        y_wh_out[i,:,:] = np.nanmean(np.stack(y_wh_temp_out, 0), 0)
        y_dc_out[i,:,:] = np.nanmean(np.stack(y_dc_temp_out, 0), 0)
        
        y_h_out[i,:] = np.nanmean(np.stack(y_h_temp_out, 0), 0)
        y_d_out[i,:] = np.nanmean(np.stack(y_d_temp_out, 0), 0)
        y_c_out[i,:] = np.nanmean(np.stack(y_c_temp_out, 0), 0)
        y_w_out[i,:] = np.nanmean(np.stack(y_w_temp_out, 0), 0)
        
        # to calculate the r-squared for area weighted anomalies, combine the original and predicted yield values to a dataframe
        df = pd.DataFrame(np.array([cell_id, years_all, y, y_pred_all]).T, columns = ['cell_id', 'year', 'y', 'y_pred'])
        
        # merge the harvested areas data of year 2000 to the dataframe
        df = df.merge(df_ha_2000, on = 'cell_id')
        
        # calculate the area-weighted yield (i.e. production) of each grid cell
        df['y_wt'] = df['harvested_area'] * df['y']
        df['y_pred_wt'] = df['harvested_area'] * df['y_pred']

        # calculate harvested area weighted yield anomaly for the whole spatial unit (climate zone or globally)
        out = pd.DataFrame()
        out['y_tot']        = df[['y_wt','year']].groupby('year').sum()['y_wt']           / df[['harvested_area','year']].groupby('year').sum()['harvested_area']
        out['y_tot_pred']   = df[['y_pred_wt','year']].groupby('year').sum()['y_pred_wt'] / df[['harvested_area','year']].groupby('year').sum()['harvested_area']
        
        # calculate r-squared (i.e. squared correlation coeffcient) multiplied with the sign of the original correlation in case the relationship is negative
        rsq_temp = np.corrcoef(out['y_tot'], out['y_tot_pred'])[0,1]**2 * np.sign(np.corrcoef(out['y_tot'], out['y_tot_pred'])[0,1])
        
        # plt.scatter(y, y_pred_all, alpha = 0.005)
        # plt.title(np.corrcoef(y, y_pred_all)[0,1])
        # plt.show()
        # plt.close()
        
        # plt.scatter(out['y_tot'], out['y_tot_pred'])
        # plt.title(rsq_temp)
        # plt.show()
        # plt.close()
        
        # save r-squared value to a list
        rsq.append(rsq_temp)
        
    
    if clim_zone_id == 0 and plot:
        # combine shap balues from list to numpy array
        shap_values_out = np.vstack(shap_values_out)
    
        # set directory
        os.chdir(os.path.join(path, 'research/crop_failures/results/shap_and_cor_figs2022'))
    
        # shap dependence plot for hot and dry days
        shap.dependence_plot("Hot days", shap_values_out, pd_eval_out, interaction_index="Dry days", show = False)
        plt.savefig(crop+'_shap_hot_dry.png', dpi = 300, bbox_inches='tight')
        plt.show()
        plt.close()
        
        # shap dependence plot for cold and wet days
        shap.dependence_plot("Cold days", shap_values_out, pd_eval_out, interaction_index="Wet days", show = False)    
        plt.savefig(crop+'_shap_cold_wet.png', dpi = 300, bbox_inches='tight')
        plt.show()
        plt.close()
        
        # shap summary plots for all variables
        ax = shap.summary_plot(shap_values_out, pd_eval_out, show = False, color_bar = False)
        fig = plt.gcf()
        fig.set_figheight(5)
        fig.set_figwidth(5)
        ax = plt.gca()
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_xlim(-0.15,0.15)
        ax.set_xticklabels([])
        plt.savefig(crop+'_shap_summary.png', dpi = 300, bbox_inches='tight')
    
        plt.show()
        plt.close()
        
        if crop == 'maize':
            ax = shap.summary_plot(shap_values_out, pd_eval_out, show = False)
            fig = plt.gcf()
            plt.savefig('info_shap_summary.png', dpi = 300, bbox_inches='tight')
            plt.show()
            plt.close()
            
    # plt.imshow(y_dh_out[0,:,:]);plt.colorbar();plt.show()
    # plt.plot(y_h_out[0,:]);plt.show()
    # plt.plot(y_d_out[0,:]);plt.show()
    # plt.plot(y_c_out[0,:]);plt.show()
    # plt.plot(y_w_out[0,:]);plt.show()
    
    rsq_out = np.array(rsq)
    
    return y_dh_out, y_wc_out, y_wh_out, y_dc_out, y_h_out, y_d_out, y_c_out, y_w_out, rsq_out, sample_size


def main(path, comm, size, rank, t_src, sm_src, irrig, transformation, gs, N, y_src, model_type, parameters, N_zones, plot, reduced = False, incl_climate_params = False):
    
    sys.path.insert(0, path+'research/crop_failures/scripts/crop_failures')
    from general_functions import get_scico_colormap
    

    # select crop based on rank of MPI run
    crop_list = ['maize','rice','soybean','wheat']
    crop = crop_list[rank % 4]
 
    print('Process has started for: ' + str(rank) + ' ' + crop + ' ' + t_src + ' ' + sm_src + ' ' + irrig + ' ' + transformation + ' ' + gs + ' ' + str(N))
    sys.stdout.flush()
    
    # import harvested areas data
    mirca_mask = import_mirca(path, crop)

    # temporal metadata for the climatological data
    clim_years = np.arange(1981,2010,1)
    
    # import average climate data for climate zoning
    T_tot = import_other_clim_data('Tavg', '90', crop,irrig, 'no_transformation', path) # average growing season temperature
    P_tot = import_other_clim_data('P_year', '90', crop,irrig, 'no_transformation', path) # total growing season precipitation
        
    # import reference climatological data
    sm_mean = import_other_clim_data(sm_src + '_mean', gs, crop,irrig, 'detrended_anom', path) # average growing season soil moisture
    t_mean = import_other_clim_data('Tavg', gs, crop,irrig, 'detrended_anom', path) # average growing season temperature
    P_year = import_other_clim_data('P_year', gs, crop,irrig, 'detrended_anom', path) # total annual precipitation (the year before harvest date)
    P_gs = import_other_clim_data('P_gs', gs, crop,irrig, 'detrended_anom', path) # total growing season precipitation
    
    # extreme temperature data (i.e. days in growing season above (below) 90th (10th) percentile)
    T_00_01 = number_of_days_vs_ref(path, t_src, gs, crop, clim_years, irrig, transformation, perc_min = 0.0, perc_max = 0.1)
    T_09_10 = number_of_days_vs_ref(path, t_src, gs, crop, clim_years, irrig, transformation, perc_min = 0.9, perc_max = 1.0)
    
    # extreme soil moisture data (i.e. days in growing season above (below) 90th (10th) percentile)
    SM_00_01 = number_of_days_vs_ref(path, sm_src, gs, crop, clim_years, irrig, transformation, perc_min = 0.0, perc_max = 0.1)
    SM_09_10 = number_of_days_vs_ref(path, sm_src, gs, crop, clim_years, irrig, transformation, perc_min = 0.9, perc_max = 1.0)
    
    # create a python list of all the variables
    if reduced:
        vars_all = [T_09_10, SM_09_10, T_00_01, SM_00_01]
        vars_all_names = ['Hot days','Dry days','Cold days','Wet days']
    if incl_climate_params:
        P_clim = np.tile(np.nanmean(P_tot,2)[:,:,None], (1, 1, T_09_10.shape[2]))
        T_clim = np.tile(np.nanmean(T_tot,2)[:,:,None], (1, 1, T_09_10.shape[2]))
        vars_all = [T_09_10, SM_09_10, T_00_01, SM_00_01, sm_mean[:,:,:-1], t_mean[:,:,:-1], P_year[:,:,:-1], P_gs[:,:,:-1], P_clim, T_clim]
        vars_all_names = ['Hot days','Dry days','Cold days','Wet days','Soil moisture','Temperature','Precipitation (yearly)','Precipitation', 'Climatological Precipitation', 'Climatological Temperature']
    elif not reduced and not incl_climate_params:
        vars_all = [T_09_10, SM_09_10, T_00_01, SM_00_01, sm_mean[:,:,:-1], t_mean[:,:,:-1], P_year[:,:,:-1], P_gs[:,:,:-1]]
        vars_all_names = ['Hot days','Dry days','Cold days','Wet days','Soil moisture','Temperature','Precipitation (yearly)','Precipitation']
    
    print('Climate data finished: '+ str(rank))
    sys.stdout.flush()
    
    # import crop yield and harvested areas data
    crop_data_fun_dict = {'ray': import_ray_crop_data,
                          'iizumi': import_iizumi_crop_data}

    crop_data_yield = crop_data_fun_dict[y_src](path, crop)
    crop_data_ha = crop_data_fun_dict['ray'](path, crop)
    
    # combine all data into a pandas dataframe
    df_combined = combine_data_to_dataframe(crop_data_yield, crop_data_ha, vars_all, vars_all_names, T_tot, P_tot, mirca_mask[irrig], clim_years)
    
    # save dataframe if run is for all variables
    if not reduced and not incl_climate_params:
        os.chdir(os.path.join(path, 'research/crop_failures/results/combined_out2022'))
        df_combined.to_csv(y_src+'_'+crop+'_'+t_src+'_'+sm_src+'_'+irrig+'_'+transformation+'_'+gs+'.csv')
        
    # select variables to include in the regression
    # change dataframe to numpy array    
    y = df_combined['Crop yield anomaly'].values # target variable (de-trended crop yield)
    X = df_combined[vars_all_names].values # explanatory variables (climate data)
    years_all = df_combined['year'].values
    clim_zones = df_combined['climate_zone'].values
    cell_id = df_combined['cell_id'].values
    ha = np.log10(1+df_combined['harvested_area'].values)
    df_ha_2000 = df_combined.loc[df_combined['year'] == 2000][['cell_id','harvested_area']]

    # define how many temporal splits are made in the training-prediction procedure
    N_folds = 4

    # define plotting setup
    cmap = get_scico_colormap('flipvik', path)
    
    # initialize output variables
    rsq_all = np.zeros((N_zones,N))*np.nan # r-squared
    sample_size_all = np.zeros(N_zones)*np.nan # sample size
    y_dh_all = np.zeros((N,25,25,N_zones))*np.nan # partial dependency (anomaly in hot / dry days)
    y_wc_all = np.zeros((N,25,25,N_zones))*np.nan # partial dependency (anomaly in wet / cold days)
    y_dc_all = np.zeros((N,25,25,N_zones))*np.nan # partial dependency (anomaly in cold / dry days)
    y_wh_all = np.zeros((N,25,25,N_zones))*np.nan # partial dependency (anomaly in hot / wet days)
    
    y_h_all = np.zeros((N,25,N_zones))*np.nan # partial dependency (anomaly in hot days)
    y_d_all = np.zeros((N,25,N_zones))*np.nan # partial dependency (anomaly in dry days)
    y_c_all = np.zeros((N,25,N_zones))*np.nan # partial dependency (anomaly in cold days)
    y_w_all = np.zeros((N,25,N_zones))*np.nan # partial dependency (anomaly in wet days)
            
    clim_zone_ids = np.sort(df_combined['climate_zone'].unique())
        
    # loop across climate zones
    for i in range(0, N_zones):
        if i == 0:
            clim_zone_id = i
        else:
            clim_zone_id = clim_zone_ids[i-1]
        
        # run the function calculate_climate_vs_crop_failures fro each climate bin (i = 1-100) as well as globally (i = 0)
        y_dh, y_wc, y_wh, y_dc, y_h, y_d, y_c, y_w, rsq, sample_size = calculate_climate_vs_crop_failures(path, crop, X, y, ha, cell_id, clim_zones, years_all, df_ha_2000, parameters, clim_zone_id, vars_all_names, N_folds, model_type, cmap = cmap, plot = plot, N = N)
                                                                                           
        y_dh_all[...,i] = y_dh
        y_wc_all[...,i] = y_wc
        y_wh_all[...,i] = y_wh
        y_dc_all[...,i] = y_dc
        
        y_h_all[...,i] = y_h
        y_d_all[...,i] = y_d
        y_c_all[...,i] = y_c
        y_w_all[...,i] = y_w
        
        rsq_all[i,...] = rsq
        sample_size_all[i] = sample_size
        
        print('rank: '+str(rank)+' crop: '+crop+' bin: '+str(i)+' finished.')
        sys.stdout.flush()

    # save the output data
    os.chdir(os.path.join(path, 'research/crop_failures/results/combined_out2022'))

    if reduced:
        os.chdir(os.path.join(path, 'research/crop_failures/results/combined_out_reduced2022'))
    if incl_climate_params:
        os.chdir(os.path.join(path, 'research/crop_failures/results/combined_out_incl_clim2022'))
        df_combined.to_csv(y_src+'_'+crop+'_'+t_src+'_'+sm_src+'_'+irrig+'_'+transformation+'_'+gs+'.csv')

    np.save('anoms_dh_'+crop+'_'+y_src+'_'+gs+'_'+irrig+'_'+transformation+'_'+t_src+'_'+sm_src+'_'+model_type+'.pkl', y_dh_all)
    np.save('anoms_wc_'+crop+'_'+y_src+'_'+gs+'_'+irrig+'_'+transformation+'_'+t_src+'_'+sm_src+'_'+model_type+'.pkl', y_wc_all)
    np.save('anoms_wh_'+crop+'_'+y_src+'_'+gs+'_'+irrig+'_'+transformation+'_'+t_src+'_'+sm_src+'_'+model_type+'.pkl', y_wh_all)
    np.save('anoms_dc_'+crop+'_'+y_src+'_'+gs+'_'+irrig+'_'+transformation+'_'+t_src+'_'+sm_src+'_'+model_type+'.pkl', y_dc_all)
    
    np.save('anoms_h_'+crop+'_'+y_src+'_'+gs+'_'+irrig+'_'+transformation+'_'+t_src+'_'+sm_src+'_'+model_type+'.pkl', y_h_all)
    np.save('anoms_d_'+crop+'_'+y_src+'_'+gs+'_'+irrig+'_'+transformation+'_'+t_src+'_'+sm_src+'_'+model_type+'.pkl', y_d_all)
    np.save('anoms_c_'+crop+'_'+y_src+'_'+gs+'_'+irrig+'_'+transformation+'_'+t_src+'_'+sm_src+'_'+model_type+'.pkl', y_c_all)
    np.save('anoms_w_'+crop+'_'+y_src+'_'+gs+'_'+irrig+'_'+transformation+'_'+t_src+'_'+sm_src+'_'+model_type+'.pkl', y_w_all)
    
    np.save('rsq_'+crop+'_'+y_src+'_'+gs+'_'+irrig+'_'+transformation+'_'+t_src+'_'+sm_src+'_'+model_type+'.pkl', rsq_all)
    np.save('sample_size_'+crop+'_'+y_src+'_'+gs+'_'+irrig+'_'+transformation+'_'+t_src+'_'+sm_src+'_'+model_type+'.pkl', sample_size_all)    
    
    
if __name__== "__main__":
    
    from mpi4py import MPI
    
    run_location = {'cluster': '/scratch/work/heinom2/',
                    'local_d': 'D:/work/',
                    'local_c': 'C:/Users/heinom2/OneDrive - Aalto University/'}
    
    path = run_location['cluster'] # get root path
    
    # initialize MPI run
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    if rank < 4:
        main(path, comm, size, rank, 'temperature', 'soil_moisture_era',   'combined', 'anom',           '90',   100, 'ray',    'XGB', 'optimal', 26, True)
        main(path, comm, size, rank, 'temperature', 'soil_moisture_era',   'combined', 'anom',           '90',   100, 'ray',    'XGB', 'optimal', 1,  False, reduced = True)
        main(path, comm, size, rank, 'temperature', 'soil_moisture_era',   'combined', 'anom',           '90',   100, 'ray',    'XGB', 'optimal', 1,  False, incl_climate_params = True)
        
    elif rank < 8:
        main(path, comm, size, rank, 'temperature', 'soil_moisture_era',   'combined', 'anom',           '90',   100, 'ray',    'RF',  'optimal', 1, False)
        
    else:
        main(path, comm, size, rank, 'temperature', 'soil_moisture_era',   'combined', 'anom',           '90',   100, 'iizumi', 'XGB', 'optimal', 1, False)
        main(path, comm, size, rank, 'temperature', 'soil_moisture_era',   'combined', 'detrended_anom', '90',   100, 'ray',    'XGB', 'optimal', 1, False)
        main(path, comm, size, rank, 'temperature', 'soil_moisture_era',   'combined', 'anom',           'real', 100, 'ray',    'XGB', 'optimal', 1, False)
        main(path, comm, size, rank, 'temperature', 'soil_moisture_gleam', 'combined', 'anom',           '90',   100, 'ray',    'XGB', 'optimal', 1, False)

