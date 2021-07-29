# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 09:36:25 2020

@author: heinom2
"""

# -*- coding: utf-8 -*-

import lzma
import pickle
import pandas as pd
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm           

#sns.set()
#sns.reset_orig()

def clim_trend_v2(path, rank, t_src, sm_src, irrig, gs, plot_all, only_log_odds):

    sys.path.insert(0, path+'research/crop_failures/scripts/crop_failures')
    from general_functions import import_climate_bin_data
    from climate_bin_analysis_raster_mpi import import_ray_crop_data, number_of_days_vs_ref, import_mirca
    
    # select crop and climate scenario based on rank of MPI run
    crop_list = ['maize','rice','soybean','wheat']
    clim_list = ['dh','wc']
            
    crop = crop_list[rank % 4]
    clim = clim_list[rank // 4]

    years = np.arange(1981,2010,1) # define time interval used here

    # obtain mirca cropland mask
    irrig_mask = import_mirca(path, crop, mask = True)    
    mirca_bool = irrig_mask[irrig]
    
    # import Ray crop data
    crop_data = import_ray_crop_data(path, crop)
    
    # de-trended crop yield and harvested areas data as numpy array
    crop_data_dtnd_values = np.copy(crop_data['detrended_yield'].values)
    crop_data_ha_values = np.copy(crop_data['harvested_area'].values)

    # find cells with numeric yield and harvested areas values for all years
    none_is_nan_yield = np.all(~np.isnan(crop_data_dtnd_values), axis = 2)
    none_is_nan_ha = np.all(~np.isnan(crop_data_ha_values), axis = 2)
    
    # create a boolean array that has True in cells with crop land according to Ray and MIRCA
    crop_bool = np.all([none_is_nan_yield, none_is_nan_ha, mirca_bool[:,:,0]], axis = 0)
        
    # import climate bin data
    path_bin_raster = os.path.join(path, 'data/earthstat/YieldGapMajorCrops_Geotiff/YieldGapMajorCrops_Geotiff/'+crop+'_yieldgap_geotiff')
    climate_bins = import_climate_bin_data(path_bin_raster, crop+'_binmatrix.tif', mirca_bool)

    # select the cimate scenario (dry-hot or wet-cold)       
    if clim == 'dh':
        smin = 0.9
        smax = 1.0
    elif clim == 'wc':
        smin = 0.0
        smax = 0.1
    
    # import annual anomaly in days in the climate scenario in question
    T_extr = number_of_days_vs_ref(path, t_src, gs, crop, years, irrig, 'anom', perc_min = smin, perc_max = smax)
    SM_extr = number_of_days_vs_ref(path, sm_src, gs, crop, years, irrig, 'anom', perc_min = smin, perc_max = smax)
    
    # Ray harvested areas for year 2000
    ha_2000 = crop_data['harvested_area'].sel(time = 2000).values
    
    T_bool = np.all(~np.isnan(T_extr), axis = 2)
    SM_bool = np.all(~np.isnan(SM_extr), axis = 2)
    
    # mask out cells without climate data and crop data in MIRCA and Ray crop data
    data_bool = np.all([T_bool, SM_bool, crop_bool], axis = 0)

    # check how large percentage of crop land has climate data
    print(crop +' ' +t_src+' '+sm_src+' '+clim + ' temperature (ha-%) - ' + str(np.nansum(T_bool * ha_2000 * crop_bool) / np.nansum(ha_2000 * crop_bool)))
    print(crop +' ' +t_src+' '+sm_src+' '+clim + ' temperature (cell-%) - ' + str(np.nansum(T_bool * crop_bool) / np.nansum(crop_bool)))
    print(crop +' ' +t_src+' '+sm_src+' '+clim + ' soil moisture (ha-%) - ' + str(np.nansum(SM_bool * ha_2000 * crop_bool) / np.nansum(ha_2000 * crop_bool)))
    print(crop +' ' +t_src+' '+sm_src+' '+clim + ' soil moisture (cell-%) - ' + str(np.nansum(SM_bool * crop_bool) / np.nansum(crop_bool)))
    print(crop +' ' +t_src+' '+sm_src+' '+clim + ' both (ha-%) - ' + str(np.nansum(data_bool * ha_2000) / np.nansum(ha_2000 * crop_bool)))
    print(crop +' ' +t_src+' '+sm_src+' '+clim + ' both (cell-%) - ' + str(np.nansum(data_bool * crop_bool) / np.nansum(crop_bool)))
    sys.stdout.flush()
    
    # initialize dictionaries to be exported
    model_pval_dict = {}
    coef_dict = {}
    coef_pval_dict = {}
    coef_pval_bstrp_dict = {}
    
    ipt_dict = {}
    ipt_pval_dict = {}
    
    sample_size_dict = {}
    threshold_dict = {}

    std_thresholds = [1.5, 1, 0.5, 0.25]
    
    T_bool_list = []
    SM_bool_list = []
    both_bool_list = []
    
    for threshold in std_thresholds:
        # create a boolean array of events above the scenario in question
        
        T_bool_temp = T_extr > threshold
        SM_bool_temp = SM_extr > threshold
        both_bool_temp = np.all([T_bool_temp, SM_bool_temp], axis = 0)
        
        T_bool_list.append(T_bool_temp)
        SM_bool_list.append(SM_bool_temp)
        both_bool_list.append(both_bool_temp)
        
    anom_list = [both_bool_list, T_bool_list, SM_bool_list]
    
    # loop across climate variables
    for anom, dtype in zip(anom_list,['both', 'T', 'SM']):

        # initialize variables
        model_pval = []
        coef = []
        coef_pval = []
        coef_bstrp_pval = []
        ipt = []
        ipt_pval = []
        sample_size = []
        threshold = []
        
        print(anom[1].shape)
        
        # loop across climate zones
        for i in range(0,101):
            
            if i != 0 and dtype != 'both':
                continue
            
            def filter_and_tabulate_clim_events(std_temp, data_bool, climate_bins, years, dtype):
            
                std_temp[data_bool == False, ...] = np.nan
                
                if i != 0:
                    std_temp[climate_bins != i,...] = np.nan
                
                # tabulate the data and filter nans
                std_temp = std_temp.reshape(360*720,-1)
                
                std_not_nan = np.all(~np.isnan(std_temp),axis = 1)
                std_temp = std_temp[std_not_nan]
                
                # create a variable to identify each spatial cell
                cell_id = np.arange(0,360*720,1)[std_not_nan]
                
                # set numpy array to pandas dataframe, each column is a single year
                df = pd.DataFrame(std_temp, columns = years.astype(float))
                df['cell_id'] = cell_id
                
                # format pandas dataframe to long format, where each row is a single observation
                df_long =  pd.melt(df, id_vars = 'cell_id', var_name = 'year', value_name = dtype).sort_values('year')
                
                # change data format from 
                X = df_long['year'].values.astype(int)
                
                X2 = sm.add_constant(X)
                
                y = df_long[dtype].astype(bool)
            
                return y, X2, std_temp, df_long, df
            
            
            def check_log_odds_linearity(df_long):
                # check linearity in log-odds
                q = 5
                
                df_long['year_quantile'] = pd.qcut(df_long['year'], q = q, labels = False)
                p_hat = df_long.groupby('year_quantile')[dtype].mean().values
                log_odds = np.log(p_hat / (1-p_hat))
                
                years_per_cat = df_long.groupby('year')['year_quantile'].mean()
                
                return np.arange(q)+1, log_odds, q, years_per_cat
            
            # loop across different extreme threshold scenarios and ensure that
            # we obtain a sufficient amount of observations to run the logistic regression
            # (here defined so that for each climate bin at least a third of years have
            # observations of the extreme type in question)
            for std_temp, threshold_temp  in zip(anom, std_thresholds):
                
                std_temp = std_temp.astype(float)
                std_temp = std_temp.copy()
                
                # filter and tabulate the raster array
                y, X2, std_temp, df_long, df = filter_and_tabulate_clim_events(std_temp, data_bool, climate_bins, years, dtype)
                
                log_odds_ids, log_odds, q, years_per_cat = check_log_odds_linearity(df_long)
                
                # check that all temporal quintiles have data
                if np.all(~np.isinf(log_odds)) and np.all(~np.isnan(log_odds)):
                    
                    m, b = np.polyfit(log_odds_ids, log_odds, 1)
                    plt.scatter(log_odds_ids, log_odds, color = [0.5,0.5,0.5], s = 20)
                    plt.plot(np.arange(0,6) + 0.5, (np.arange(0,6)+0.5)*m+b)
                    ax = plt.gca()
                    ax.set(xlim = [0.5,5.5], xticks = [1,2,3,4,5], xticklabels = [])
                    ax.spines['right'].set_visible(False)
                    ax.spines['top'].set_visible(False)
                    fig_log_odds = plt.gcf()
                    plt.show()
                    if only_log_odds:
                        break
                    
                    est = sm.Logit(y, X2).fit(disp = 0)
                    threshold.append(threshold_temp)
                    
                    est_list = []
                    
                    for N in range(100):
                        rnd_idx = np.random.choice(y.shape[0], y.shape[0], replace = True)
                        y_temp = y[rnd_idx]
                        X2_temp = X2[rnd_idx,:]
                        
                        est_temp = sm.Logit(y_temp, X2_temp).fit(disp = 0)
                        est_list.append(est_temp)
                    
                    break

                else:
                    print('failure to fit for bin ' +dtype+ ' '+ str(i) + ' clim: '+clim +' , crop: '+crop+ ', '+sm_src+ ' '+ str(threshold_temp))
            
            # function to calculate the sigmoid function
            def sigmoid_array(z):                                        
                return 1 / (1 + np.exp(-z))
            
            # if global analysis, plot results
            if only_log_odds and i == 0:
                # print(years_per_cat)
                os.chdir(path+'research/crop_failures/results/climate_trend_2021')
                fig_log_odds.savefig(crop+'_'+clim+'_'+sm_src+'_'+t_src+'_'+dtype+'_'+'15std'+'_log_odds_linearity.png', dpi = 300, bbox_inches='tight')
                plt.close()
                continue
            elif only_log_odds:
                continue
            
            # evaluate modeled probability for each year
            # probs = est.predict(sm.add_constant(years))
            probs = sigmoid_array(years * est.params['x1'] + est.params['const'])
            
            def plot_clim_trend(years, df_long, probs, est, dtype, i = False, est_list = False):
                
                freq = df_long.groupby('year')[dtype].mean()
                plt.plot(years, freq.values, color = [0.9,0.9,0.9])
                
                for est_temp in est_list:
                    probs_temp = sigmoid_array(years * est_temp.params['x1'] + est_temp.params['const'])
                    plt.plot(years, probs_temp, c = [0.8,0.8,0.8])
                
                plt.plot(years, freq.rolling(5, center = True).mean(), color = [0.3,0.3,0.8])
                plt.plot(years, probs, c = 'black')
                
                ax = plt.gca()
                
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                
                if dtype == 'both':
                    ylim = 0.102
                else:
                    ylim = 0.2502
                
                if not i:
                    ax.set(xlim = [1980.5, 2009.5], ylim = [0, ylim], xticklabels = [], yticklabels = [])
                
                return plt.gcf()
            
            
            # if global analysis, plot results
            if i == 0:
                fig = plot_clim_trend(years, df_long, probs, est, dtype, i, est_list = est_list)
                os.chdir(path+'research/crop_failures/results/climate_trend_2021')
                fig.savefig(crop+'_'+clim+'_'+sm_src+'_'+t_src+'_'+dtype+'_'+'15std'+'_clim_trend_logistic.png', dpi = 300, bbox_inches='tight')
                plt.close()
                
                fig_log_odds.savefig(crop+'_'+clim+'_'+sm_src+'_'+t_src+'_'+dtype+'_'+'15std'+'_log_odds_linearity.png', dpi = 300, bbox_inches='tight')
                plt.close()
            
            # for checking what the data looks like, plot climate bin specific results
            if plot_all and dtype == 'both':
                plot_clim_trend(years, df_long, probs, est, dtype, i, est_list = est_list)
                plt.title(crop +' ' + clim+' ' +str(i) +'    ' + str(df.shape[0]) + '  ' + str(est.llr_pvalue))
                plt.show()
                
            coef_bstrp = np.array([est_temp.params['x1'] for est_temp in est_list])
            coef_pval_bstrp_np = (1 - np.max([np.sum(coef_bstrp < 0), np.sum(coef_bstrp > 0)]) / coef_bstrp.shape[0]) * 2
            
            print('coef t-test p-value: ' + str(est.pvalues['x1']))
            print('coef bootstrapped p-value: ' + str(coef_pval_bstrp_np))
            
            # save model results to lists
            model_pval.append(est.llr_pvalue)
            coef.append(est.params['x1'])
            coef_pval.append(est.pvalues['x1'])
            coef_bstrp_pval.append(coef_pval_bstrp_np)
            ipt.append(est.params['const'])
            ipt_pval.append(est.pvalues['const'])
            sample_size.append(df_long.shape[0])
            
        if only_log_odds:
            break
            
        # save model results to dictionaries
        model_pval_dict[dtype] = model_pval
        coef_dict[dtype] = coef
        coef_pval_dict[dtype] = coef_pval
        coef_pval_bstrp_dict[dtype] = coef_bstrp_pval


        ipt_dict[dtype] = ipt
        ipt_pval_dict[dtype] = ipt_pval
        sample_size_dict[dtype] = sample_size
        
        threshold_dict[dtype] = threshold
            
    # export results as dictionaries
    os.chdir(path+'research/crop_failures/results/climate_trend_2021')
    pickle.dump(model_pval_dict, lzma.open(crop+'_'+clim+'_'+sm_src+'_'+t_src+'_model_pval_v2.pkl.lzma', 'wb'))
    pickle.dump(coef_dict, lzma.open(crop+'_'+clim+'_'+sm_src+'_'+t_src+'_coef_v2.pkl.lzma', 'wb'))
    pickle.dump(coef_pval_dict, lzma.open(crop+'_'+clim+'_'+sm_src+'_'+t_src+'_coef_pval_v2.pkl.lzma', 'wb'))
    pickle.dump(coef_pval_bstrp_dict, lzma.open(crop+'_'+clim+'_'+sm_src+'_'+t_src+'_coef_bstrp_pval_v2.pkl.lzma', 'wb'))

    pickle.dump(ipt_dict, lzma.open(crop+'_'+clim+'_'+sm_src+'_'+t_src+'_ipt_v2.pkl.lzma', 'wb'))
    pickle.dump(ipt_pval_dict, lzma.open(crop+'_'+clim+'_'+sm_src+'_'+t_src+'_ipt_pval_v2.pkl.lzma', 'wb'))
    pickle.dump(sample_size_dict, lzma.open(crop+'_'+clim+'_'+sm_src+'_'+t_src+'_sample_size_v2.pkl.lzma', 'wb'))
    
    pickle.dump(threshold_dict, lzma.open(crop+'_'+clim+'_'+sm_src+'_'+t_src+'_threshold_v2.pkl.lzma', 'wb'))



if __name__== "__main__":
    
    
    run_location = {'cluster': '/scratch/work/heinom2/',
                    'local_d': 'D:/work/',
                    'local_c': 'C:/Users/heinom2/'}
    
    run_MPI = False

    t_src = 'temperature'
    gs = '90' 
    irrig = 'combined'
    
    if run_MPI:

        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()
        
        path = run_location['cluster']
        sm_src = 'soil_moisture_era'
        clim_trend_v2(path, rank, t_src, sm_src, irrig, gs, False)
            
        sm_src = 'soil_moisture_gleam'
        clim_trend_v2(path, rank, t_src, sm_src, irrig, gs, False)
    
    elif not run_MPI:
        
        path = run_location['local_c']
        
        for rank in range(0, 8):
            sm_src = 'soil_moisture_era'
            clim_trend_v2(path, rank, t_src, sm_src, irrig, gs, True, only_log_odds = False)
            
            sm_src = 'soil_moisture_gleam'
            clim_trend_v2(path, rank, t_src, sm_src, irrig, gs, True, only_log_odds = False)



