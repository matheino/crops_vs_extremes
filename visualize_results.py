
import lzma
import pickle
import pandas as pd
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

path = 'C:/Users/heinom2/'
sys.path.insert(0, path+'research/crop_failures/scripts/crop_failures')
from general_functions import import_climate_bin_data, plot_table_to_raster, get_scico_colormap
from climate_bin_analysis_raster_mpi import import_ray_crop_data, import_mirca

def plot_rsq_vs_production_and_irrigation(crops, path, tdata, smdata, gs, y_src, irrig, transformation, model_type, minrsq):
    
    # loop across crop types
    for crop in crops:
        
        # import results about r2 for the modeling set-up in question
        os.chdir(os.path.join(path, 'research/crop_failures/results/combined_out'))
        rsq_all = np.load('rsq_'+crop+'_'+y_src+'_'+gs+'_'+irrig+'_'+transformation+'_'+tdata+'_'+smdata+'_'+model_type+'.pkl.npy')
    
        # average r2 for each climate zone (zero index contains the global results, and is thus esxcluded)
        rsq_mean = np.mean(rsq_all[1:,:],1)*100
        rsq_mean[rsq_mean < 0] = 0
        
        # import cropland mask for year 2000 (mirca)
        mirca_mask = import_mirca(path, crop)
        
        # import cropland mask for year 2000 (ray)
        crop_data_raster = import_ray_crop_data(path, crop)
        yield_mask_ray = ~np.isnan(crop_data_raster['yield'].sel(time = 2000).values)
        yield_mask_combined = np.all([yield_mask_ray, mirca_mask[irrig][...,0]], axis = 0)
        # note that both ray and mirca cropland masks are used here, as the data is masked with both when conducting the the analyses
        
        # import climate bins array and mask with cropland extent
        path_bin_raster = path+'data/earthstat/YieldGapMajorCrops_Geotiff/YieldGapMajorCrops_Geotiff//'+crop+'_yieldgap_geotiff'
        climate_bins = import_climate_bin_data(path_bin_raster, crop+'_binmatrix.tif', mirca_mask[irrig])       
        climate_bins[~yield_mask_combined] = 0
        climate_bins = climate_bins.reshape(-1).astype(int)
        
        # calculate production for each grid cell for year 2000, based on ray data (yield * harvested area)
        ha_np = crop_data_raster['harvested_area'].sel(time = 2000).values
        y_np = crop_data_raster['yield'].sel(time = 2000).values
        
        ha_np[~yield_mask_combined] = np.nan
        y_np[~yield_mask_combined] = np.nan
        
        prod_np = (ha_np * y_np).reshape(-1)
        prod_np[np.isnan(prod_np)] = 0
        
        # production per climate bin
        prod_per_bin = np.bincount(climate_bins, prod_np)[1:] / 10**6
        
        # import rainfed and irrigated harvested area (mirca) and calculate total harvested area
        mirca_rfc, mirca_irc = import_mirca(path, crop, mask = False)
        mirca_rfc[~yield_mask_combined] = 0
        mirca_irc[~yield_mask_combined] = 0        
        combined = mirca_rfc + mirca_irc
        
        # calculate irrigated and combined harvested area per climate bin
        irc_per_bin = np.bincount(climate_bins, mirca_irc.reshape(-1))[1:]
        combined_per_bin = np.bincount(climate_bins, combined.reshape(-1))[1:]
        combined_per_bin[combined_per_bin == 0] = np.nan
        
        # calculate percentage of irrigated harvested area for each climate bin
        irrig_perc_per_bin = irc_per_bin / combined_per_bin * 100
        
        # scatter plots with regression lines about relationship between r2 and production across climate bins
        ax = sns.regplot(prod_per_bin, rsq_mean, color = 'black', scatter_kws={'s': 7})
        ax.set(ylim = [0, 100])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        fig1 = plt.gcf()
        plt.show()
        
        # scatter plots with regression lines about relationship between r2 and irrigation use across climate bins
        ax = sns.regplot(irrig_perc_per_bin, rsq_mean, color = 'black', scatter_kws={'s': 7})
        ax.set(ylim = [0, 100], xlim = [0,100])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        fig2 = plt.gcf()
        plt.show()
        
        # export the figures
        os.chdir(os.path.join(path, 'research/crop_failures/results/figs2021'))
        fig1.savefig(crop+'_'+y_src+'_'+tdata+'_'+smdata+'_'+transformation+'_'+model_type+'_'+irrig+'_rsq_vs_production_scatter.png',bbox_inches='tight')
        fig2.savefig(crop+'_'+y_src+'_'+tdata+'_'+smdata+'_'+transformation+'_'+model_type+'_'+irrig+'_rsq_vs_irrig_scatter.png',bbox_inches='tight')
        
        

def N_per_bin_box_and_maps_fig(crops, path, tdata, smdata, gs, y_src, irrig, transformation, model_type):

    # loop across crop types
    for crop in crops:
        
        # define color limits (as log10 transofmation is used, the color range is actually 100 - 10000)
        clim = (2,4)
        
        # define colormap
        cmap = get_scico_colormap('bilbao', path)
    
        # load sample size data globally and for each climate bin
        os.chdir(os.path.join(path, 'research/crop_failures/results/combined_out'))
        sample_size = np.load('sample_size_'+crop+'_'+y_src+'_'+gs+'_'+irrig+'_'+transformation+'_'+tdata+'_'+smdata+'_'+model_type+'.pkl.npy')
        sample_size_global = sample_size[0]
        sample_size = sample_size[1:]
        
        print('global sample size: ', str(sample_size_global))
        print('sample size: ', crop, ' min ', sample_size.min(), ' max ', sample_size.max())
        
        # as there is a lot of variation in the sample sizes across climate bins, take a log10 transformation on the data
        sample_size = np.log10(sample_size)
        
        # plot data for each climate bin
        fig, ax = plt.subplots()
        plt.imshow(np.flip(sample_size.reshape(10,10).astype(float), axis = 0), clim = clim, cmap = cmap);
        ax.set(xticks = [i-0.5 for i in range(0,11)], 
                xticklabels = [],
                yticks = [i-0.5 for i in range(0,11)], 
                yticklabels = [])
        ax.tick_params(axis = 'x', length=3, direction = 'in', width = 1)
        ax.tick_params(axis = 'y', length=3, direction = 'in', width = 1)
        plt.xticks(rotation=45)
        fig1 = plt.gcf()
        plt.show()
        
        # import cropland mask for year 2000 (mirca)
        mirca_mask = import_mirca(path, crop)
        
        # import cropland mask for year 2000 (ray)
        crop_data_raster = import_ray_crop_data(path, crop)
        yield_mask_ray = ~np.isnan(crop_data_raster['yield'].sel(time = 2000).values)
        yield_mask_combined = np.all([yield_mask_ray, mirca_mask[irrig][:,:,0]], axis = 0)
        # note that both ray and mirca cropland masks are used here, as the data is masked with both when conducting the the analyses
        
        # import climate bins array
        path_bin_raster = path+'data/earthstat/YieldGapMajorCrops_Geotiff/YieldGapMajorCrops_Geotiff//'+crop+'_yieldgap_geotiff'
        climate_bins = import_climate_bin_data(path_bin_raster, crop+'_binmatrix.tif', mirca_mask[irrig])  
        
        # sample size to pandas dataframe format
        df = pd.DataFrame({'sample_size': sample_size, 'bin_id': np.arange(1,101,1)}, columns = ['sample_size', 'bin_id'])
        
        # plot the tabulated values as a map for each climate zone
        plot_table_to_raster(climate_bins, df, 'sample_size', clim = clim, mask = yield_mask_combined, scico = 'bilbao')
        fig2 = plt.gcf()
        plt.show()
        
        # export the figures
        os.chdir(os.path.join(path, 'research/crop_failures/results/figs2021'))
        fig1.savefig(crop+'_'+y_src+'_'+tdata+'_'+smdata+'_'+transformation+'_'+model_type+'_'+irrig+'_sample_size_box.png',bbox_inches='tight')
        fig2.savefig(crop+'_'+y_src+'_'+tdata+'_'+smdata+'_'+transformation+'_'+model_type+'_'+irrig+'_sample_size_map.png',bbox_inches='tight')
        
        # produce and export a colorbar for the plot
        if crop == 'maize':
            img = plt.imshow(sample_size.reshape(10,10).astype(float), clim = clim, cmap = cmap);
            plt.gca().set_visible(False)
    
            cbar = plt.colorbar(img, extend = 'both', orientation='vertical', ticks=[2, 3, 4])
            cbar.ax.set_yticklabels(['100', '1000', '10000'])
            cbar.set_label('Sample size per', fontsize=12)
            plt.savefig('colobar_sample_size.png', dpi = 300, bbox_inches='tight')   


def rsq_box_and_maps_fig(crops, path, tdata, smdata, gs, y_src, irrig, transformation, model_type):

    # loop across crop types
    for crop in crops:
        
        # define colormap
        cmap = get_scico_colormap('flipbamako', path)
        
        # define color intervals
        clim = (0, 50)
        
        # import results about r2 for the modeling set-up in question
        os.chdir(os.path.join(path, 'research/crop_failures/results/combined_out'))
        rsq_all = np.load('rsq_'+crop+'_'+y_src+'_'+gs+'_'+irrig+'_'+transformation+'_'+tdata+'_'+smdata+'_'+model_type+'.pkl.npy')
        
        # multiply by 100 to transform into percentage
        rsq_all = rsq_all*100

        # extract global r2        
        rsq_global = rsq_all[0, :]
        print(crop+' global rsquared:                 '+ str(rsq_global.mean().round(1)))
        print(crop+' global rsquared 97.5 percentile: '+ str(np.percentile(rsq_global, 97.5).round(1)))
        print(crop+' global rsquared  2.5 percentile: '+ str(np.percentile(rsq_global, 2.5).round(1)))
        
        # global mean r2
        rsq_mean = np.mean(rsq_all[1:,:],1)
              
        # plot r2 for each climate bin (heatmap)
        fig, ax = plt.subplots()
        plt.imshow(np.flip(rsq_mean.reshape(10,10), axis = 0), clim = clim, cmap = cmap);
        ax.set(xticks = [i-0.5 for i in range(1,10)], 
                xticklabels = [],
                yticks = [i-0.5 for i in range(1,10)], 
                yticklabels = [])
        ax.tick_params(axis = 'y', length=5.5, direction = 'in', width = 2.5)
        ax.tick_params(axis = 'x', length=5.5, direction = 'in', width = 2.5)
        plt.xticks(rotation=45)
        fig1 = plt.gcf()
        plt.show()
        
        # import cropland mask for year 2000 (mirca)
        mirca_mask = import_mirca(path, crop)
        
        # import cropland mask for year 2000 (ray)
        crop_data_raster = import_ray_crop_data(path, crop)
        yield_mask_ray = ~np.isnan(crop_data_raster['yield'].sel(time = 2000).values)
        yield_mask_combined = np.all([yield_mask_ray, mirca_mask[irrig][:,:,0]], axis = 0)
        # note that both ray and mirca cropland masks are used here, as the data is masked with both when conducting the the analyses
        
        # import climate bins array
        path_bin_raster = path+'data/earthstat/YieldGapMajorCrops_Geotiff/YieldGapMajorCrops_Geotiff//'+crop+'_yieldgap_geotiff'
        climate_bins = import_climate_bin_data(path_bin_raster, crop+'_binmatrix.tif', mirca_mask[irrig])  
        
        # data to pandas dataframe format
        df = pd.DataFrame({'r2': rsq_mean, 'bin_id': np.arange(1,101,1)}, columns = ['r2', 'bin_id'])
        
        # plot the tabulated r2 values as a map for each climate zone
        fig2 = plot_table_to_raster(climate_bins, df, 'r2', clim = clim, mask = yield_mask_combined, scico = 'flipbamako')
        plt.show()
        
        # exoirt the figures
        os.chdir(os.path.join(path, 'research/crop_failures/results/figs2021'))
        fig1.savefig(crop+'_'+y_src+'_'+tdata+'_'+smdata+'_'+transformation+'_'+model_type+'_'+irrig+'_rsq_box.png',bbox_inches='tight')
        fig2.savefig(crop+'_'+y_src+'_'+tdata+'_'+smdata+'_'+transformation+'_'+model_type+'_'+irrig+'_rsq_map.png',bbox_inches='tight')
        
        # create and export a colorbar
        if crop == 'maize':
            img = plt.imshow(rsq_mean.reshape(10,10), clim = clim, cmap = cmap);
            plt.gca().set_visible(False)
    
            cbar = plt.colorbar(img, extend = 'max', orientation='vertical')
            cbar.set_label('% of crop yield variability explained', fontsize=12)
            plt.savefig('colobar_rsquared.png', dpi = 300, bbox_inches='tight')   
     

def partial_dependence_global_2d_fig(crops, path, tdata, smdata, gs, y_src, extr_type, irrig, transformation, model_type):

    # loop across crop types
    for crop in crops:
    
        # define colormap for the plot        
        cmap = get_scico_colormap('flipvik', path)
        
        # import results about r2 for the modeling set-up in question
        os.chdir(os.path.join(path, 'research/crop_failures/results/combined_out'))
        rsq_all = np.load('rsq_'+crop+'_'+y_src+'_'+gs+'_'+irrig+'_'+transformation+'_'+tdata+'_'+smdata+'_'+model_type+'.pkl.npy')
        
        # multiply by 100 to transform into percentage
        rsq_all = rsq_all*100

        # extract global r2        
        rsq_global = rsq_all[0, :]
        print(crop+' global rsquared:                 '+ str(rsq_global.mean().round(1)))
        print(crop+' global rsquared 97.5 percentile: '+ str(np.percentile(rsq_global, 97.5).round(1)))
        print(crop+' global rsquared  2.5 percentile: '+ str(np.percentile(rsq_global, 2.5).round(1)))
        
        # import results from the partial dependence calculations for the model set-up in question
        os.chdir(path+ 'research/crop_failures/results/combined_out')
        anom_all = np.load('anoms_'+extr_type+'_'+crop+'_'+y_src+'_'+gs+'_'+irrig+'_'+transformation+'_'+tdata+'_'+smdata+'_'+model_type+'.pkl.npy')
        
        # extract global results
        anom_global = anom_all[..., 0] * 100
        
        # obtain the anomaly categories for which the partial dependence is calculated (cats_orig) and plotted (cats)
        cats_orig = np.linspace(-3.0,3.0,25)
        # cats = np.linspace(-3.0,3.0,25)
        # cats = np.linspace(-2.5,2.5,21)
        cats = np.linspace(-2.25,2.25,19)

        # create an indexing for the results included in the plot
        cats_index = np.isin(cats_orig, cats)
        
        # calculate the mean global anomaly 
        Z = np.nanmean(anom_global, 0)
        
        # select those anomaly values, which are plotted
        Z = Z[np.ix_(cats_index,cats_index)]
        
        # create a meshgrid for the anomaly categoreis
        XX, YY = np.meshgrid(cats, cats)
    
        # initialize figure and specify its size
        fig1 = plt.figure(1, figsize = (10,10))
        
        # specify the coloring limits and categorization
        levels = np.linspace(-5.0,5.0, num = 21)
        
        # create a contour plot showing the partial dependence of crop yield anomaly to weather extremes
        contour = plt.contourf(XX, YY, Z, levels = levels, cmap = cmap, extend = 'both')
        contour.ax.tick_params(axis = 'y', length=7.5, direction = 'in', width = 5.5)
        contour.ax.tick_params(axis = 'x', length=7.5, direction = 'in', width = 5.5)
        contour.ax.set(xticks = [-2, -1, 0, 1, 2], yticks = [-2, -1, 0, 1, 2], xticklabels = [], yticklabels = [])
        fig1 = plt.gcf()
        plt.show()

        # export figure
        os.chdir(os.path.join(path, 'research/crop_failures/results/figs2021'))
        fig1.savefig(crop+'_'+y_src+'_'+tdata+'_'+smdata+'_'+gs+'_'+transformation+'_'+model_type+'_'+extr_type+'_'+irrig+'_partial_dependency.png',bbox_inches='tight')

        # create and export colorbar
        if crop == 'maize':
            img = plt.contourf(XX, YY, Z, cmap = cmap, levels = levels, extend = 'both')
            plt.gca().set_visible(False)
            
            levels_tick = levels[::2].astype(int)
            
            cbar = plt.colorbar(img, extend = 'max',orientation='horizontal', ticks = levels_tick)
            cbar.ax.set_xticklabels(levels_tick.astype(str), rotation = 45, fontsize = 12)
            cbar.set_label('% yield change', fontsize=12)
            plt.savefig('colobar_partial_dependence.png', dpi = 300, bbox_inches='tight') 
            plt.show()
   
            
def partial_dependency_global_violin_fig(crops, path, tdata, smdata, gs, y_src, irrig, transformation, std_lim, model_type):
    
    os.chdir(path+r'research/crop_failures/results/combined_out')
    
    # loop across crop types
    for crop in crops:
        
        def import_and_filter_anoms(extr_type, crop, y_src, gs, irrig, transformation, std_lim):
            
            # set working folder
            os.chdir(path+ 'research/crop_failures/results/combined_out')
            
            # import partial dependence estimates for the climate scenario in question
            anom_data_both = np.load('anoms_'+extr_type+'_'+crop+'_'+y_src+'_'+gs+'_'+irrig+'_'+transformation+'_'+tdata+'_'+smdata+'_'+model_type+'.pkl.npy')
            anom_data_t = np.load('anoms_'+extr_type[1]+'_'+crop+'_'+y_src+'_'+gs+'_'+irrig+'_'+transformation+'_'+tdata+'_'+smdata+'_'+model_type+'.pkl.npy')
            anom_data_sm = np.load('anoms_'+extr_type[0]+'_'+crop+'_'+y_src+'_'+gs+'_'+irrig+'_'+transformation+'_'+tdata+'_'+smdata+'_'+model_type+'.pkl.npy')
            
            # select the global scenario
            anom_data_both = anom_data_both[..., 0]
            anom_data_t = anom_data_t[..., 0]
            anom_data_sm = anom_data_sm[..., 0]
            
            # anomaly categories for which the partial dependence is calculated
            cats = np.linspace(-3.0,3.0,25)
            XX, YY = np.meshgrid(cats, cats)
            both_bool = np.all([XX == std_lim, YY == std_lim], axis = 0)
            
            # extract the partial dependence estimates specified for the anomaly category in question (defined in std_lim)
            # and reshape to 1-d vector
            anom_t = anom_data_t[..., cats == std_lim].reshape(-1) * 100
            anom_sm = anom_data_sm[..., cats == std_lim].reshape(-1) * 100
            anom_both = anom_data_both[..., both_bool].reshape(-1) * 100
            
            print('Violinplots - Crop: ',crop,' mean anomaly for extr both is in ', extr_type,': ', np.mean(anom_both))
            print('Violinplots - Crop: ',crop,' 97.5% anomaly for extr both is in', extr_type,': ', np.percentile(anom_both, 97.5))
            print('Violinplots - Crop: ',crop,' 2.5% anomaly for extr both is in ', extr_type,': ', np.percentile(anom_both, 2.5))

            print('Violinplots - Crop: ',crop, ' mean anomaly for extr T is in ', extr_type,': ', np.mean(anom_t))
            print('Violinplots - Crop: ',crop, ' 97.5% anomaly for extr T is in ', extr_type,': ', np.percentile(anom_t, 97.5))
            print('Violinplots - Crop: ',crop, ' 2.5% anomaly for extr T is in ', extr_type,': ', np.percentile(anom_t, 2.5))

            print('Violinplots - Crop: ',crop,' mean anomaly for extr SM is in   ', extr_type,': ', np.mean(anom_sm))
            print('Violinplots - Crop: ',crop,' 97.5% anomaly for extr SM is in  ', extr_type,': ', np.percentile(anom_sm, 97.5))
            print('Violinplots - Crop: ',crop,' 2.5% anomaly for extr SM is in', extr_type,': ', np.percentile(anom_sm, 2.5))
            
            return anom_t, anom_sm, anom_both
        
        # extract the partial dependence anomalies for the climate scanario in question
        anom_hot, anom_dry, anom_hotdry = import_and_filter_anoms('dh', crop, y_src, gs, irrig, transformation, std_lim)
        anom_cold, anom_wet, anom_coldwet = import_and_filter_anoms('wc', crop, y_src, gs, irrig, transformation, std_lim)
        
        # combine the parital dependence anomalies to a list for plotting
        list_of_anoms = [anom_hotdry, anom_hot, anom_dry, anom_coldwet, anom_cold, anom_wet]
                
        # set up plot parameters
        labels = ['Hot and\n Dry', 'Hot', 'Dry', 'Cold and\n Wet', 'Cold', 'Wet']

        list_to_plot = list_of_anoms
        list_to_labels = labels
        
        # visualize the results as violin plots
        violin = plt.violinplot(list_to_plot, showmeans = True, showextrema = False)
        ax = plt.gca()        
        for ax1 in violin['bodies']:
            ax1.set_facecolor('#35C3B8')
            ax1.set_edgecolor(None)
            ax1.set_alpha(1)
        # .. more plot parameters
        ax.axhline(0, color = 'black', linestyle = '--', linewidth = 0.5)
        ax.xaxis.tick_top()
        ax.spines['bottom'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis = 'x', length=0, labelsize = 12)
        ax.tick_params(axis = 'y', labelsize = 12)

        ax.set(ylim = [-6.25, 1.05], yticks = np.array([-10, -7.5, -5, -2.5, 0, 5]), xticks = np.arange(1,len(labels)+1))
        
        labels = ax.set_xticklabels(list_to_labels, fontsize = 12)
        
        # .. and more ..
        for label in labels:
            label.set_y(1.10)        
            label.set_verticalalignment('top') 
        fig1 = plt.gcf()
        plt.show()
        
        # export figures
        os.chdir(os.path.join(path, 'research/crop_failures/results/figs2021'))

        fig1.savefig(crop+'_'+y_src+'_'+tdata+'_'+smdata+'_'+gs+'_'+transformation+'_'+model_type+'_'+irrig+'_'+str(std_lim)+'_violins.png',bbox_inches='tight')
        
        
def partial_dependency_local_2d_fig(crops, path, tdata, smdata, gs, y_src, extr_type,  irrig, transformation, std_lim, model_type, minrsq):
    
    # loop across crops
    for crop in crops:

        # define colomap        
        cmap = get_scico_colormap('flipvik', path)
        
        # define color intervals
        clim = (-10,10)
        
        # import results from the partial dependence calculations for the model set-up in question
        os.chdir(path+ 'research/crop_failures/results/combined_out')
        anom_data = np.load('anoms_'+extr_type+'_'+crop+'_'+y_src+'_'+gs+'_'+irrig+'_'+transformation+'_'+tdata+'_'+smdata+'_'+model_type+'.pkl.npy')
    
        # import results about r2 for the modeling set-up in question
        rsq_all = np.load('rsq_'+crop+'_'+y_src+'_'+gs+'_'+irrig+'_'+transformation+'_'+tdata+'_'+smdata+'_'+model_type+'.pkl.npy')

        # extract partial dependence and r2 results for each climate bin
        anom_data = anom_data[..., 1:]
        rsq_bins = np.mean(rsq_all[1:,:],1)
        rsq_bins = np.percentile(rsq_all[1:,:], 10, 1)
        
        # check for which climate bins the r2 threshold applies
        rsq_bool = rsq_bins > minrsq
                        
        # obtain the anomaly categories for which the partial dependence is calculated
        cats = np.linspace(-3.0,3.0,25)
                  
        # create a meshgrid for anomaly categoreis
        XX, YY = np.meshgrid(cats, cats)
                
        anom_data = anom_data.transpose(3,0,1,2) # rotate dimensions to (climate_bins, sampling, X, Y)
                
        anoms_both = np.zeros(100) # initialize array for partial dependence estimates
        
        # ukraine = climate_bins[50:100,400:500][20:45,15:40]
        # ukraine = np.unique(ukraine)[1:].astype(int)
        
        # import the event thresholds used to derive the co-occurring extreme events
        # for each of the logistic regression model created globally as well as for each climate bin
        os.chdir(path+ 'research/crop_failures/results/climate_trend_2021')
        threshold_dict = pickle.load(lzma.open(crop+'_'+extr_type+'_'+smdata+'_'+tdata+'_threshold_v2.pkl.lzma','rb'))
        thresholds = np.array(threshold_dict['both'])[1:]
        
        # loop across climate bins
        for i in range(0,anom_data.shape[0]):
                        
            # extract the partial dependence results for the climate bin in question
            anom_temporary = anom_data[i,...]
            
            # define what theshold to use for the climate anomaly category
            if std_lim == 'dynamic':
                std_lim_temp = thresholds[i]
            else:
                std_lim_temp = std_lim
            
            # from the meshgrid isolate categories of interest, i.e. both extreme types occur simultaneously
            temp = YY == std_lim_temp
            sm = XX == std_lim_temp
            both = np.all([temp, sm], axis = 0)

            # calculate average partial dependence 
            anom_temporary = anom_temporary[...,both].reshape(-1) * 100
              
            # check whether 97.5% of the partial dependence estimates are either larger or smaller than zero
            # if they are, append the average anomaly to the list, otherwise append value zero
            if (np.nansum(anom_temporary > 0) / anom_temporary.shape[0] > 0.975 or np.nansum(anom_temporary < 0) / anom_temporary.shape[0] > 0.975) and rsq_bool[i]:
                val = np.mean(anom_temporary)
            else:
                val = 0
            
            anoms_both[i] = val # save value to list
       
        # plot partial dependency at the specified anomaly category (defined in std_lim or thresholds) for each climate bin (heatmap)
        fig, ax = plt.subplots()
        print(crop + ' ' + extr_type+':  '+str(np.sum(anoms_both<0)))
        plt.imshow(np.flip(np.array(anoms_both).reshape(10,10).astype(float), axis = 0), cmap = cmap, clim = clim);
        ax.set(xticks = [i-0.5 for i in range(0,11)], 
                xticklabels = [],
                yticks = [i-0.5 for i in range(0,11)], 
                yticklabels = [])
        ax.tick_params(axis = 'x', length=3, direction = 'in', width = 1)
        ax.tick_params(axis = 'y', length=3, direction = 'in', width = 1)
        plt.xticks(rotation=45)
        fig1 = plt.gcf()
        plt.show()
        
        # histograms of impacts
        plt.hist(anoms_both)
        plt.title(crop)
        plt.show()
                    
        # import cropland mask for year 2000 (mirca)
        mirca_mask = import_mirca(path, crop)
        
        # import cropland mask for year 2000 (ray)
        crop_data_raster = import_ray_crop_data(path, crop)
        yield_mask_ray = ~np.isnan(crop_data_raster['yield'].sel(time = 2000).values)
        yield_mask_combined = np.all([yield_mask_ray, mirca_mask[irrig][:,:,0]], axis = 0)
        # note that both ray and mirca cropland masks are used here, as the data is masked with both when conducting the the analyses
        
        # import climate bins array
        path_bin_raster = path+'data/earthstat/YieldGapMajorCrops_Geotiff/YieldGapMajorCrops_Geotiff//'+crop+'_yieldgap_geotiff'
        climate_bins = import_climate_bin_data(path_bin_raster, crop+'_binmatrix.tif', mirca_mask[irrig])  
        
        # data to pandas dataframe format
        df = pd.DataFrame({'anoms': np.array(anoms_both), 'bin_id': np.arange(1,101,1)}, columns = ['anoms', 'bin_id'])
        
        # plot the tabulated partial dependence anomaly values as a map for each climate zone
        plot_table_to_raster(climate_bins, df, 'anoms', mask = yield_mask_combined, scico = 'flipvik', clim = clim)
        fig2 = plt.gcf()
        plt.show()
        
        # export figures
        os.chdir(os.path.join(path, 'research/crop_failures/results/figs2021'))
        fig1.savefig(crop+'_'+y_src+'_'+tdata+'_'+smdata+'_'+transformation+'_'+model_type+'_'+extr_type+'_'+irrig+'_box.png', dpi = 300, bbox_inches='tight')
        fig2.savefig(crop+'_'+y_src+'_'+tdata+'_'+smdata+'_'+transformation+'_'+model_type+'_'+extr_type+'_'+irrig+'_map.png', dpi = 300, bbox_inches='tight')
        
        # create and export colorbar
        if crop == 'maize':
            img = plt.imshow(np.array(df['anoms']).reshape(10,10).astype(float), clim = clim, cmap = cmap)
            plt.gca().set_visible(False)
            
            ticks_num = np.array([-10,-8,-6,-4,-2,0,2,4,6,8,10])
            
            cbar = plt.colorbar(img, extend = 'both',orientation='horizontal', ticks = list(ticks_num))
            cbar.ax.set_xticklabels((ticks_num).astype(str), rotation = 45, fontsize = 12)
            cbar.set_label('% yield change', fontsize=12)
            plt.savefig('colobar_local_partial_dependency.png', dpi = 300, bbox_inches='tight') 
            plt.show()
            

def clim_trend_v2(crops, path, gs, extr_type, irrig, sm_src, t_src):
        
    # loop across crops
    for crop in crops:

        # define colomap
        cmap_trend = get_scico_colormap('vik', path)
        
        cmap_thres = get_scico_colormap('fliptokyo', path)
        from matplotlib.colors import BoundaryNorm
        boundaries = [0, 0.375, 0.75, 1.25, 1.75]
        norm_thres = BoundaryNorm(boundaries, cmap_thres.N, clip=True)

        # define color intervals
        clim_trend = (0.8, 1.2)
        clim_thres = (0.25, 1.5)        
        
        os.chdir(path+ 'research/crop_failures/results/climate_trend_2021')
        
        # import the p-value of the logistic regression models created globally as well as for each climate bin
        model_pval_dict = pickle.load(lzma.open(crop+'_'+extr_type+'_'+sm_src+'_'+t_src+'_model_pval_v2.pkl.lzma','rb'))

        # import the coefficient and related p-value for the of the logistic regression models created globally as well as for each climate bin
        coef_dict = pickle.load(lzma.open(crop+'_'+extr_type+'_'+sm_src+'_'+t_src+'_coef_v2.pkl.lzma','rb'))
        coef_pval_dict = pickle.load(lzma.open(crop+'_'+extr_type+'_'+sm_src+'_'+t_src+'_coef_pval_v2.pkl.lzma','rb'))
        coef_pval_bstrp_dict = pickle.load(lzma.open(crop+'_'+extr_type+'_'+sm_src+'_'+t_src+'_coef_bstrp_pval_v2.pkl.lzma','rb'))
        
        # import the intercept and related p-value for the of the logistic regression models created globally as well as for each climate bin
        ipt_dict = pickle.load(lzma.open(crop+'_'+extr_type+'_'+sm_src+'_'+t_src+'_ipt_v2.pkl.lzma','rb'))
        ipt_pval_dict = pickle.load(lzma.open(crop+'_'+extr_type+'_'+sm_src+'_'+t_src+'_ipt_pval_v2.pkl.lzma','rb'))
        
        # import the sample size of the logistic regression models created globally as well as for each climate bin
        sample_size_dict = pickle.load(lzma.open(crop+'_'+extr_type+'_'+sm_src+'_'+t_src+'_sample_size_v2.pkl.lzma','rb'))
        
        # import the event thresholds used to derive the co-occurring extreme events
        # for each of the logistic regression model created globally as well as for each climate bin
        threshold_dict = pickle.load(lzma.open(crop+'_'+extr_type+'_'+sm_src+'_'+t_src+'_threshold_v2.pkl.lzma','rb'))
        
        # loop across co-occurring as well as individually occurring extreme events
        for clim_key in model_pval_dict.keys():
            
            # obtain data from the dictionaries for the extreme scenario in question
            model_pval = np.array(model_pval_dict[clim_key])
            
            coef = np.array(coef_dict[clim_key])
            coef_pval = np.array(coef_pval_dict[clim_key])
            coef_bstrp_pval = np.array(coef_pval_bstrp_dict[clim_key])

            ipt = np.array(ipt_dict[clim_key])
            ipt_pval = np.array(ipt_pval_dict[clim_key])
            
            sample_size = sample_size_dict[clim_key]
            thresholds = np.array(threshold_dict[clim_key])
            
            # initialize list to save values
            val_list = []
            
            # loop across all climate bins
            for i in range(0, len(coef)):
                
                # extract the logistic regression coefficient
                val_temp = np.exp(coef[i])
                
                # if the analysis is conducted at global level, print some of the results
                if i == 0:
                    print(crop+' '+clim_key+' pval model: '+str(model_pval[i]) +' pval coef: '+str(coef_pval[i]) +' pval coef bootstrapped: '+str(coef_bstrp_pval[i]) )
                    print(sample_size[i])
                
                # check whether the model and coefficient p-values are significant at 95% confidence level
                if model_pval[i] < 0.05 and coef_pval[i] < 0.05 and coef_bstrp_pval[i] < 0.05:
                    val = val_temp
                else:
                    val = 1
                    
                # plt.scatter(years,probs)
                # plt.title(val)
                # plt.show()
                
                # append the estimated coefficient to a list
                val_list.append(val)
                
            if len(coef) == 1:
                continue
            
            # transform the python list to a numpy array and select only climate bin -level results (i.e. exclude global results)
            val_np_array = np.array(val_list)[1:]
            
            # plot the climate bin specific trends as a heatmap
            fig1, ax = plt.subplots()
            plt.imshow(np.flip(val_np_array.reshape(10,10).astype(float), axis = 0), clim = clim_trend, cmap = cmap_trend);
            ax.set(xticks = [i-0.5 for i in range(1,10)], 
                    xticklabels = [],
                    yticks = [i-0.5 for i in range(1,10)], 
                    yticklabels = [])
            ax.tick_params(axis = 'y', length=5.5, direction = 'in', width = 2.5)
            ax.tick_params(axis = 'x', length=5.5, direction = 'in', width = 2.5)
            plt.xticks(rotation=45)
            plt.show()
            
            # import cropland mask for year 2000 (mirca)
            mirca_mask = import_mirca(path, crop)
            
            # import cropland mask for year 2000 (ray)
            crop_data_raster = import_ray_crop_data(path, crop)
            yield_mask_ray = ~np.isnan(crop_data_raster['yield'].sel(time = 2000).values)
            yield_mask_combined = np.all([yield_mask_ray, mirca_mask[irrig][:,:,0]], axis = 0)
            # note that both ray and mirca cropland masks are used here, as the data is masked with both when conducting the the analyses
            
            # import climate bins array
            path_bin_raster = path+'data/earthstat/YieldGapMajorCrops_Geotiff/YieldGapMajorCrops_Geotiff//'+crop+'_yieldgap_geotiff'
            climate_bins = import_climate_bin_data(path_bin_raster, crop+'_binmatrix.tif', mirca_mask[irrig])  
            
            # data to pandas dataframe format
            df_trend = pd.DataFrame({'trend': val_np_array, 'bin_id': np.arange(1,101,1)}, columns = ['trend', 'bin_id'])
            
            # plot the tabulated trend values as a map for each climate zone
            fig2 = plot_table_to_raster(climate_bins, df_trend, 'trend', clim = clim_trend, mask = yield_mask_combined, scico = 'vik')
            plt.show()
            
            # export trend figures
            os.chdir(os.path.join(path, 'research/crop_failures/results/figs2021'))
            fig1.savefig('clim_trend_v2_'+crop+'_'+sm_src+'_'+t_src+'_'+extr_type+'_'+irrig+'_'+clim_key+'_'+'15std'+'_box.png', dpi = 300, bbox_inches='tight')
            fig2.savefig('clim_trend_v2_'+crop+'_'+sm_src+'_'+t_src+'_'+extr_type+'_'+irrig+'_'+clim_key+'_'+'15std'+'_map.png', dpi = 300, bbox_inches='tight')
            
            # define and export trend colormap
            if crop == 'maize':
                
                img = plt.imshow(np.array(df_trend['trend']).reshape(10,10).astype(float), cmap = cmap_trend, clim = clim_trend)
                plt.gca().set_visible(False)
        
                cbar = plt.colorbar(img, extend = 'both', orientation = 'horizontal')
                cbar.set_label('Trend', fontsize=12)
                
                os.chdir(os.path.join(path, 'research/crop_failures/results/figs2021'))
        
                plt.savefig('colobar_climate_trend.png', dpi = 300, bbox_inches='tight')
                
                plt.close()
            
            ######### THRESHOLDS #############
            # plot the climate bin specific thresholds as a heatmap
            fig3, ax = plt.subplots()
            plt.imshow(np.flip(thresholds[1:].reshape(10,10).astype(float), axis = 0), norm = norm_thres, cmap = cmap_thres)
            ax.set(xticks = [i-0.5 for i in range(1,10)], 
                    xticklabels = [],
                    yticks = [i-0.5 for i in range(1,10)], 
                    yticklabels = [])
            ax.tick_params(axis = 'y', length=5.5, direction = 'in', width = 2.5)
            ax.tick_params(axis = 'x', length=5.5, direction = 'in', width = 2.5)
            plt.xticks(rotation=45)
            plt.show()

            # put the thresholds data to dataframe format
            df_thresholds = pd.DataFrame({'threshold': thresholds[1:], 'bin_id': np.arange(1,101,1)}, columns = ['threshold', 'bin_id'])

            # plot the tabulated threshold values as a map for each climate zone
            fig4 = plot_table_to_raster(climate_bins, df_thresholds, 'threshold', clim = clim_thres, mask = yield_mask_combined, scico = 'fliptokyo')
            plt.show()
            
            # export figures
            os.chdir(os.path.join(path, 'research/crop_failures/results/figs2021'))
            fig3.savefig('thresholds_'+crop+'_'+sm_src+'_'+t_src+'_'+extr_type+'_'+irrig+'_'+clim_key+'_'+'15std'+'_box.png', dpi = 300, bbox_inches='tight')
            fig4.savefig('thresholds_'+crop+'_'+sm_src+'_'+t_src+'_'+extr_type+'_'+irrig+'_'+clim_key+'_'+'15std'+'_map.png', dpi = 300, bbox_inches='tight')
            
            # define and export threshold colormap
            if crop == 'maize':
                                
                img = plt.imshow(np.flip(thresholds[1:].reshape(10,10).astype(float), axis = 0), norm = norm_thres, cmap = cmap_thres);
                plt.gca().set_visible(False)              
                
                cbar = plt.colorbar(img, orientation = 'horizontal', ticks=[0.375/2, 0.375+(0.75-0.375)/2, 1, 1.5])
                cbar.ax.set_xticklabels([0.25, 0.5, 1, 1.5])
                cbar.ax.tick_params(axis = 'x', length=0, width = 0)
                
                cbar.set_label('co-occurrence threshold', fontsize=12)
                
                os.chdir(os.path.join(path, 'research/crop_failures/results/figs2021'))
        
                plt.savefig('colorbar_coccur_threshold.png', dpi = 300, bbox_inches='tight')
                

def visualize_climate_bins(crops):
    
    import cartopy.crs as ccrs
    
    # loop across all crop types
    for crop in crops:
        
        # import cropland mask for year 2000 (mirca)
        mirca_mask = import_mirca(path, crop)
        
        # import cropland mask for year 2000 (ray)
        crop_data_raster = import_ray_crop_data(path, crop)
        yield_mask_ray = ~np.isnan(crop_data_raster['yield'].sel(time = 2000).values)
        yield_mask_combined = np.all([yield_mask_ray, mirca_mask[irrig][...,0]], axis = 0)
        # note that both ray and mirca cropland masks are used here, as the data is masked with both when conducting the the analyses
        
        # import climate bins array and mask with cropland extent
        path_bin_raster = path+'data/earthstat/YieldGapMajorCrops_Geotiff/YieldGapMajorCrops_Geotiff//'+crop+'_yieldgap_geotiff'
        climate_bins = import_climate_bin_data(path_bin_raster, crop+'_binmatrix.tif', mirca_mask[irrig])       
        
        climate_bins[climate_bins == 0] = np.nan
        climate_bins[~yield_mask_combined] = np.nan
        
        # define coordinates for plotting
        lats = np.linspace(89.75,-89.75,360)
        lons = np.linspace(-179.75,179.75,720)
    
        # initialize figure with information about projection
        ax = plt.axes(projection=ccrs.Robinson())
        ax.coastlines(linewidth = 0.5)
        ax.set_extent([-180, 180, -60, 90], crs=ccrs.PlateCarree())
        ax.outline_patch.set_visible(False)
    
        # create a custom colormap for plotting climate bins
        import matplotlib.colors as col
        def discrete_cmap():
            # define individual colors as hex values
            cpool = [ '#282972', '#3850a2', '#3f68b1', '#45c7f0', '#7dc9a1','#bcd636', '#fed401', '#f26522', '#ed1c24', '#7d1416',
                      '#2c2e7e', '#3c51a3', '#436db5', '#4dc8ee', '#82cba8','#c0d848', '#fed70c', '#f36e21', '#ed1c24', '#89181a',
                      '#2e308e', '#4154a4', '#4973b9', '#57c9ed', '#89cdae','#c4da59', '#feda25', '#f4792a', '#ee2a28', '#932726',
                      '#3b3d98', '#4957a6', '#527dbf', '#60cbed', '#90d0b3','#cadd6a', '#fedc3b', '#f6863e', '#ef3b3b', '#9c393a',
                      '#4e4ea1', '#525da9', '#6189c6', '#6dceec', '#97d3ba','#cee079', '#fee14b', '#f79050', '#f04e4d', '#a54d4c',
                      '#6060ab', '#5d64ad', '#6f95ce', '#87d4ee', '#a1d7c0','#d3e387', '#fee35e', '#f89c61', '#f15f5f', '#b06060',
                      '#7272b5', '#6b6eb3', '#7da4d6', '#87d4ee', '#a9dac6','#d9e696', '#fee673', '#f9a873', '#f27172', '#b87272',
                      '#8585c0', '#8081be', '#8fb2de', '#a6def0', '#b4decd','#dee9a5', '#fee986', '#fab385', '#f48585', '#c28585',
                      '#9898cb', '#9796ca', '#a0bee4', '#b4e3f3', '#bfe3d4','#e1ecb3', '#feec98', '#fbc09a', '#f69899', '#cc9898',
                      '#adacd6', '#aeabd5', '#b0c9e9', '#b4e3f3', '#cbe8db','#e7efc1', '#fff2ac', '#fccbac', '#f8acac', '#d5adac' ]
            
            cpool = np.flip(np.array(cpool).reshape(10,10).T, axis = 1).reshape(-1)
            
            cmap = col.ListedColormap(cpool, 'indexed')
            
            return cmap
        
        # import a custom colormap for plotting te climate bins
        cmap = discrete_cmap()
        
        # plot the climate bins on a global map        
        plt.pcolormesh(lons, lats, climate_bins, transform=ccrs.PlateCarree(), cmap = cmap)
        # cbar = plt.colorbar(orientation = 'horizontal', fraction=0.046, pad=0.04)
        fig1 = plt.gcf()
        plt.show()
        
        # exoirt figure
        os.chdir(os.path.join(path, 'research/crop_failures/results/figs2021'))
        fig1.savefig(crop+'_climate_bins_map.png', dpi = 300, bbox_inches='tight')        
        
        # create and export colormap
        if crop == 'maize':
            colorscale = np.flip(np.arange(1,101,1).reshape(10,10),0)
            plt.imshow(colorscale, cmap = cmap)
            ax = plt.gca()
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.set(xticklabels = [], yticklabels = [])
            ax.tick_params(axis = 'x', length=0, direction = 'in', width = 0)
            ax.tick_params(axis = 'y', length=0, direction = 'in', width = 0)
            fig_cb = plt.gcf()
            fig_cb.savefig('colorbar_climate_bins_map.png', dpi = 300, bbox_inches='tight')


def visualize_quantile_thresholds(crops, path, gs, extr_type, irrig, sm_src, t_src):
    
    from climate_bin_analysis_raster_mpi import obtain_clim_ref
    import cartopy.crs as ccrs
    
    import time
    time_st = time.time()
    # loop across crops
    for crop in crops:
        
        time_elapsed = round(time.time() - time_st,2)
        print(crop+', total time elapsed in the function: '+ str(time_elapsed))
        # import cropland mask for year 2000 (mirca)
        mirca_mask = import_mirca(path, crop)
        
        # import cropland mask for year 2000 (ray)
        crop_data_raster = import_ray_crop_data(path, crop)
        yield_mask_ray = ~np.isnan(crop_data_raster['yield'].sel(time = 2000).values)
        yield_mask_combined = np.all([yield_mask_ray, mirca_mask[irrig][...,0]], axis = 0)
        # note that both ray and mirca cropland masks are used here, as the data is masked with both when conducting the the analyse
        
        # select the thresholds for the climate scenario in question
        if extr_type == 'dh':
            smin = 0.9
            smax = 1.0
        elif extr_type == 'wc':
            smin = 0.0
            smax = 0.1
        
        
        # import the soil moisture and tempereture data for the quantile thesholds in question
        years = np.arange(1981,2009,1) 
        ref_bin_bool_T, bin_min_T, bin_max_T, mask = obtain_clim_ref(t_src, gs, crop, irrig, years, smax, smin, path)
        ref_bin_bool_SM, bin_min_SM, bin_max_SM, mask = obtain_clim_ref(sm_src, gs, crop, irrig, years, smax, smin, path)
        
        # define the colormap and color limits for the figure
        clim_T = (-5, 50)
        clim_SM = (0,1)
        cmap_T =  get_scico_colormap('fliproma', path)
        cmap_SM = get_scico_colormap('flipdevon', path)
        
        import matplotlib.colors as col
        
        clim_T = col.DivergingNorm(vmin=-5, vcenter=0, vmax=50)
        clim_SM = col.DivergingNorm(vmin=0, vcenter=0.5, vmax=1)
        
        def mask_and_plot(data, crop_data, path, gs, extr_type, irrig, sm_src, t_src, clim, cmap, dtype):

            # define coordinates for plotting
            lats = np.linspace(89.75,-89.75,360)
            lons = np.linspace(-179.75,179.75,720)
    
            # initialize figure with information about projection
            ax = plt.axes(projection=ccrs.Robinson())
            ax.coastlines(linewidth = 0.5)
            ax.set_extent([-180, 180, -60, 90], crs=ccrs.PlateCarree())
            ax.outline_patch.set_visible(False)
            
            # mask the data with cropland
            data[~crop_data] = np.nan
            
            # plot the values on a global map        
            plt.pcolormesh(lons, lats, data, transform=ccrs.PlateCarree(), norm = clim, cmap = cmap)
            fig = plt.gcf()
            # plt.colorbar()
            
            # export figure
            os.chdir(os.path.join(path, 'research/crop_failures/results/figs2021'))
            fig.savefig(crop+'_'+sm_src+'_'+t_src+'_'+extr_type+'_'+irrig+'_'+dtype+'_clim_threshold_map.png', dpi = 300, bbox_inches='tight')
            
            plt.show()
            
        # plot the climate category thresholds on a map
        if extr_type == 'dh':
            mask_and_plot(bin_min_T, yield_mask_combined, path, gs, extr_type, irrig, sm_src, t_src, clim_T, cmap_T,'temperature')
            mask_and_plot(bin_min_SM, yield_mask_combined, path, gs, extr_type, irrig, sm_src, t_src, clim_SM, cmap_SM,'soil_moisture')
        
        elif extr_type == 'wc':
            mask_and_plot(bin_max_T, yield_mask_combined, path, gs, extr_type, irrig, sm_src, t_src, clim_T, cmap_T,'temperature')
            mask_and_plot(bin_max_SM, yield_mask_combined, path, gs, extr_type, irrig, sm_src, t_src, clim_SM, cmap_SM, 'soil_moisture')
            
        # define and export colormaps
        if crop == 'maize':
            img = plt.imshow(bin_max_T, cmap = cmap_T, norm = clim_T)
            plt.gca().set_visible(False)
    
            cbar = plt.colorbar(img, extend = 'both',orientation = 'vertical')
            cbar.set_label('Temperature (degrees Celcius)', fontsize=12)
            os.chdir(os.path.join(path, 'research/crop_failures/results/figs2021'))

            plt.savefig('colorbar_T_threshold.png', dpi = 300, bbox_inches='tight')
            plt.show()
            
            img = plt.imshow(bin_max_SM, cmap = cmap_SM, norm = clim_SM)
            plt.gca().set_visible(False)
    
            cbar = plt.colorbar(img,orientation = 'vertical')
            cbar.set_label('Soil moisture deficit', fontsize=12)
            os.chdir(os.path.join(path, 'research/crop_failures/results/figs2021'))

            plt.savefig('colorbar_SM_threshold.png', dpi = 300, bbox_inches='tight')
            plt.show()
            

gs = '90'
irrig = 'combined'
crops = ['soybean','rice']
crops = ['maize','rice','soybean','wheat']
y_src = 'ray'
t_src = 'temperature'

###### Violin plots #######
partial_dependency_global_violin_fig(crops, path, t_src, 'soil_moisture_era', gs, y_src, irrig, 'anom', 1.5, 'XGB')

partial_dependency_global_violin_fig(crops, path, t_src, 'soil_moisture_era', gs, y_src, irrig, 'anom', 1.5, 'RF')
partial_dependency_global_violin_fig(crops, path, t_src, 'soil_moisture_era', gs, y_src, irrig, 'detrended_anom', 1.5, 'XGB')
partial_dependency_global_violin_fig(crops, path, t_src, 'soil_moisture_era', gs, 'iizumi', irrig, 'anom', 1.5, 'XGB')
partial_dependency_global_violin_fig(crops, path, t_src, 'soil_moisture_gleam', gs, y_src, irrig, 'anom', 1.5, 'XGB')
partial_dependency_global_violin_fig(crops, path, t_src, 'soil_moisture_era', 'real', y_src, irrig, 'anom', 1.5, 'XGB')

##### Partial dependence 2D global ######
partial_dependence_global_2d_fig(crops, path, t_src, 'soil_moisture_era', gs, y_src, 'dh', irrig, 'anom', 'XGB')
partial_dependence_global_2d_fig(crops, path, t_src, 'soil_moisture_era', gs, y_src, 'wc', irrig, 'anom', 'XGB')

partial_dependence_global_2d_fig(crops, path, t_src, 'soil_moisture_era', gs, y_src, 'dh', irrig, 'anom', 'RF')
partial_dependence_global_2d_fig(crops, path, t_src, 'soil_moisture_era', gs, y_src, 'wc', irrig, 'anom', 'RF')

partial_dependence_global_2d_fig(crops, path, t_src, 'soil_moisture_era', gs, y_src, 'dh', irrig, 'detrended_anom', 'XGB')
partial_dependence_global_2d_fig(crops, path, t_src, 'soil_moisture_era', gs, y_src, 'wc', irrig, 'detrended_anom', 'XGB')

partial_dependence_global_2d_fig(crops, path, t_src, 'soil_moisture_era', gs, 'iizumi', 'dh', irrig, 'anom', 'XGB')
partial_dependence_global_2d_fig(crops, path, t_src, 'soil_moisture_era', gs, 'iizumi', 'wc', irrig, 'anom', 'XGB')

partial_dependence_global_2d_fig(crops, path, t_src, 'soil_moisture_gleam', gs, y_src, 'dh', irrig, 'anom', 'XGB')
partial_dependence_global_2d_fig(crops, path, t_src, 'soil_moisture_gleam', gs, y_src, 'wc', irrig, 'anom', 'XGB')

partial_dependence_global_2d_fig(crops, path, t_src, 'soil_moisture_era', 'real', y_src, 'dh', irrig, 'anom', 'XGB')
partial_dependence_global_2d_fig(crops, path, t_src, 'soil_moisture_era', 'real', y_src, 'wc', irrig, 'anom', 'XGB')

##### R-squared - local ######
rsq_box_and_maps_fig(crops, path, t_src, 'soil_moisture_era', gs, y_src, irrig, 'anom', 'XGB')

##### Partial dependency 2D local ######
partial_dependency_local_2d_fig(crops, path,  t_src, 'soil_moisture_era', gs, y_src, 'dh', irrig, 'anom', 1.5, 'XGB', -100)
partial_dependency_local_2d_fig(crops, path,  t_src, 'soil_moisture_era', gs, y_src, 'wc', irrig, 'anom', 1.5, 'XGB', -100)

##### Climate trend #####
clim_trend_v2(crops, path, gs, 'dh', irrig, 'soil_moisture_era', t_src)
clim_trend_v2(crops, path, gs, 'wc', irrig, 'soil_moisture_era', t_src)

clim_trend_v2(crops, path, gs, 'dh', irrig, 'soil_moisture_gleam', t_src)
clim_trend_v2(crops, path, gs, 'wc', irrig, 'soil_moisture_gleam', t_src)

###### Supplementary figures ######
visualize_climate_bins(crops)

N_per_bin_box_and_maps_fig(crops, path, t_src, 'soil_moisture_era', gs, y_src, irrig, 'anom', 'XGB')

plot_rsq_vs_production_and_irrigation(crops, path, t_src, 'soil_moisture_era', gs, y_src, irrig, 'anom', 'XGB', minrsq = -100)

visualize_quantile_thresholds(crops, path, gs, 'dh', irrig, 'soil_moisture_era', t_src)
visualize_quantile_thresholds(crops, path, gs, 'wc', irrig, 'soil_moisture_era', t_src)


