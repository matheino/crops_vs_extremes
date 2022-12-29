# -*- coding: utf-8 -*-
import os
import numpy as np

def to_growing_season(data, crop, irrig, path, year_length_t1, gs = 'real'):
    # function creates a boolean array, which, for each grid cell, keeps the
    # value for days that are within the growing season and other days are sets values
    # to nan
    
    import xarray as xr
    
    if crop == 'Soybean':
        crop_str = 'Soybeans'
    else:
        crop_str = crop
    
    # import data about growing season
    os.chdir(path+'data/crop_calendar_isi_mip')
    growing_season_raw = xr.open_dataset(str(crop_str)+'_'+irrig+'_growing_season_dates_v1.25.nc4',decode_times = False)
    harvest_day = np.flipud(growing_season_raw['harvest day'].values-1) # extract day of harvest (as day of year)
    
    
    # import array about growing season lenght 
    if gs == 'real':
        gs_length = np.flipud(growing_season_raw['growing season length'].values) # length of full growing season from AgMIP 
    else:
        gs_length = np.zeros_like(harvest_day) + int(gs) # fixed growing season length (defined in gs variable)
    
    # add the number of days in the year t-1 to harvest day
    # (data from year t-1 is utilized as harvest may occur at the beginning of the year t
    # and hence, the majority of the growing season falls to the previous year (i.e. year t-1))
    harvest_day = harvest_day + year_length_t1
    
    data_new = np.zeros(data.shape).astype(float)*np.nan # define output array (all nans)
    
    # loop across all grid cells
    for lat_i in range(0,harvest_day.shape[0]):
        for lon_i in range(0,harvest_day.shape[1]):
            planting_day_i = harvest_day[lat_i,lon_i]-gs_length[lat_i,lon_i] # calculate planting day for the grid cell in question
            harvest_day_i = harvest_day[lat_i,lon_i] # extract the harvest day for the grid cell in question
            
            # continue to next loop round, if the grid cell has nan for planting or harvest date
            # as well as if the original harvest_day data is below zero
            if np.isnan(planting_day_i) or np.isnan(harvest_day_i) or (harvest_day_i - year_length_t1) < 0:
                continue
            else:
                planting_day_i = planting_day_i.astype(int) # change values to integer here, as integers can't be nan, but indexing doesn't work with float
                harvest_day_i = harvest_day_i.astype(int)
                
                # save values from the original data that are within the growing season into the output array
                data_new[planting_day_i:harvest_day_i, lat_i, lon_i] = data[planting_day_i:harvest_day_i, lat_i, lon_i]
                                
    return data_new
   
    
def get_crop_and_irrig_list(rank,size):
    
    # obtain crop / irrigation set-up for the MPI run in question
    # (based on rank and size of the MPI run)
    crop_list = ['Maize','Rice','Soybean','Wheat']
    
    if rank < 4 and size > 0:
        crop_list = [crop_list[rank]]
        irrig_list = ['ir']
    elif size > 0:
        crop_list = [crop_list[rank-4]]
        irrig_list = ['rf']
    else:
        irrig_list = ['ir','rf']
        
    return crop_list, irrig_list


def plot_table_to_raster(raster_info, df, var, lon_ext = (-180, 180), lat_ext = (-60, 90), clim = None, mask = None, scico = None, label_name = None, title = None, cbar = False, norm = None):
    
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt
    
    
    # check id column name and re-define it, if necessary
    try:
        df['ids']
    except KeyError:
        try:
            df['ids'] = df['fpu_id']
        except KeyError:
            df['ids'] = df['bin_id']
    
    # initialize array to be plotted
    data_raster = np.zeros(raster_info.shape) - 999
    
    # loop through each row in the table, and change the areas defined in raster_info
    # the corresponding value of the table
    for index, row in df.iterrows():            
        data_raster[raster_info == row['ids']] = row[var]

    # change no-value cells to nan
    data_raster[data_raster == -999] = np.nan
    
    # if a mask is defined, change cells outside the mask extent to nan
    if mask is not None:
        data_raster[~mask] = np.nan
    
    # latitude and longitude information in of data_raster array
    lats = np.linspace(89.75,-89.75,360)
    lons = np.linspace(-179.75,179.75,720)

    # initialize figure with information about projection
    ax = plt.axes(projection=ccrs.Robinson())
    ax.coastlines(linewidth = 0.5)
    ax.set_extent([lon_ext[0], lon_ext[1], lat_ext[0], lat_ext[1]], crs=ccrs.PlateCarree())
    ax.outline_patch.set_visible(False)
    
    # define colormap
    if scico is not None:
        cmap = get_scico_colormap(scico, 'C:/Users/heinom2/OneDrive - Aalto University/')
    else:
        cmap = 'viridis'

    # remove frame from the figure
    for spine in ["left", "top", "right","bottom"]:
        ax.spines[spine].set_visible(False)
    
    # two colorscheme options to plot the array: with min and max colorbar limits (clim variable)
    # and explicitly defined colorbar intervals (norm variable)
    if clim is not None and norm is None:
        ax.pcolormesh(lons, lats, data_raster, transform=ccrs.PlateCarree(), vmin=clim[0], vmax=clim[1], cmap = cmap)
        
        if cbar:
            cbar = plt.colorbar(orientation = 'horizontal', fraction=0.046, pad=0.04, extend='max')
            cbar.set_label(label_name, fontsize = 14)
            plt.title(label = title, fontsize = 16)
            plt.tight_layout()

    elif clim is None and norm is not None:
        ax.pcolormesh(lons, lats, data_raster, transform=ccrs.PlateCarree(), norm = norm, cmap = cmap)
        
        if cbar:
            cbar = plt.colorbar(orientation = 'horizontal', fraction=0.046, pad=0.04, extend='max')
            cbar.set_label(label_name, fontsize = 14)
            plt.title(label = title, fontsize = 16)
            plt.tight_layout()
            
    ax.coastlines(linewidth = 0.5)
        
    return plt.gcf()


def get_scico_colormap(scico,path):
    
    # obtain scico colormapping info
    # 'flip' a the start of scico variable indicates that the
    # colorbar is reveresed
    scico_str = scico.replace('flip','')
    scico_np = np.loadtxt(path+'research/crop_failures/scripts/ScientificColourMaps6/ScientificColourMaps6/'+scico_str+'/'+scico_str+'.txt')

    if 'flip' in scico:
        scico_np = np.flip(scico_np,axis = 0)

    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(scico,scico_np)
    
    return cmap




