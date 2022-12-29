# -*- coding: utf-8 -*-

import lzma
import pickle
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm           

def clim_trend_v2(path, rank, t_src, sm_src, irrig, gs, plot_all, only_log_odds):
 
    # select crop and climate scenario based on rank of MPI run
    crop_list = ['maize','rice','soybean','wheat']
    clim_list = ['dh','wc']
            
    crop = crop_list[rank % 4]
    clim = clim_list[rank // 4]
    
    # import dataframe with climate data
    y_src = 'ray'
    os.chdir(path+ 'research/crop_failures/results/combined_out2022')
    climate_data_df = pd.read_csv(y_src+'_'+crop+'_'+t_src+'_'+sm_src+'_'+irrig+'_anom_'+gs+'.csv')

    threshold = 1.5
    
    years = np.arange(1981,2010,1)
    
    # select the cimate scenario (dry-hot or wet-cold)        
    if clim == 'dh':
        columns = ['cell_id','year','climate_zone', 'Hot days','Dry days']
        df_long = climate_data_df[columns].copy()
        df_long['T'] = df_long['Hot days'] > threshold
        df_long['SM'] = df_long['Dry days'] > threshold
        
        
    elif clim == 'wc':
        columns = ['cell_id','year','climate_zone', 'Cold days','Wet days']
        df_long = climate_data_df[columns].copy()
        df_long['T'] = df_long['Cold days'] > threshold
        df_long['SM'] = df_long['Wet days'] > threshold
    
    df_long['both'] = np.all([df_long['T'], df_long['SM']], axis = 0)
    
    clim_zone_ids = np.sort(df_long['climate_zone'].unique())

    # initialize dictionaries to be exported
    model_pval_dict = {}
    coef_dict = {}
    coef_pval_dict = {}
    coef_pval_bstrp_dict = {}
    
    ipt_dict = {}
    ipt_pval_dict = {}
    
    sample_size_dict = {}
    
    # loop across climate variables
    for dtype in ['both', 'T', 'SM']:

        # initialize variables
        model_pval = []
        coef = []
        coef_pval = []
        coef_bstrp_pval = []
        ipt = []
        ipt_pval = []
        sample_size = []
    
        # loop across climate zones
        for i in range(0, 26):
            
            # check  whether analysis is global or climate zone specific:
            # calculate only trends in co-occurrin events if for climate zones
            if i > 0 and dtype != 'both':
                continue
            
            # extract inspected variables from the dataframe
            y = df_long[dtype].astype(bool)
            X = df_long['year'].values.astype(int)
            X2 = sm.add_constant(X)   
            
            # isolate climate zone specific data
            if i > 0:
                clim_zone_id = clim_zone_ids[i-1]
                y = y[df_long['climate_zone'] == clim_zone_id]
                X = X[df_long['climate_zone'] == clim_zone_id]
                X2 = X2[df_long['climate_zone'] == clim_zone_id, ...]
            
            df_Xy = pd.DataFrame({'year': X, dtype: y}, columns = ['year',dtype])
            
            # calculate log odds for each temporal quintile
            def check_log_odds_linearity(df_long):
                # check linearity in log-odds
                q = 5
                
                df_long['year_quantile'] = pd.qcut(df_long['year'], q = q, labels = False)
                p_hat = df_long.groupby('year_quantile')[dtype].mean().values
                log_odds = np.log(p_hat / (1-p_hat))
                
                years_per_cat = df_long.groupby('year')['year_quantile'].mean()
                
                return np.arange(q)+1, log_odds, q, years_per_cat
        
            log_odds_ids, log_odds, q, years_per_cat = check_log_odds_linearity(df_Xy)
                
            # check whether all temporal quintiles have data
            log_odds_inf = np.any(np.isinf(log_odds))

            # plot log odds values
            m, b = np.polyfit(log_odds_ids, log_odds, 1)
            plt.scatter(log_odds_ids, log_odds, color = [0.5,0.5,0.5], s = 20)
            plt.plot(np.arange(0,6) + 0.5, (np.arange(0,6)+0.5)*m+b)
            ax = plt.gca()
            ax.set(xlim = [0.5,5.5], xticks = [1,2,3,4,5], xticklabels = [])
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            fig_log_odds = plt.gcf()
            plt.close()
                    
            est = sm.Logit(y, X2).fit(disp = 0)
                    
            est_list = []
                    
            # bootstrap observation 100 times and calculate the logistic regression for each bootsrap sample
            for N in range(100):
                rnd_idx = np.random.choice(y.shape[0], y.shape[0], replace = True)
                y_temp = y.iloc[rnd_idx]
                X2_temp = X2[rnd_idx,...]
                
                est_temp = sm.Logit(y_temp, X2_temp).fit(disp = 0)
                est_list.append(est_temp)
                        
            # function to calculate the sigmoid function
            def sigmoid_array(z):                                        
                return 1 / (1 + np.exp(-z))
                        
            # evaluate modeled probability for each year
            # probs = est.predict(sm.add_constant(years))
            probs = sigmoid_array(years * est.params['x1'] + est.params['const'])
            
            # if global analysis, plot results and export figures
            def plot_clim_trend(years, df_Xy, probs, est, dtype, i = False, est_list = False):
                
                freq = df_Xy.groupby('year')[dtype].mean()
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
            
            
            if i == 0:
                fig = plot_clim_trend(years, df_Xy, probs, est, dtype, i, est_list = est_list)
                os.chdir(path+'research/crop_failures/results/climate_trend2022')
                fig.savefig(crop+'_'+clim+'_'+sm_src+'_'+t_src+'_'+dtype+'_'+'15std'+'_clim_trend_logistic.png', dpi = 300, bbox_inches='tight')
                plt.close()
                
                fig_log_odds.savefig(crop+'_'+clim+'_'+sm_src+'_'+t_src+'_'+dtype+'_'+'15std'+'_log_odds_linearity.png', dpi = 300, bbox_inches='tight')
                plt.close()
            
            # for checking what the data looks like, plot climate bin specific results
            if (plot_all or log_odds_inf) and dtype == 'both':
                plot_clim_trend(years, df_Xy, probs, est, dtype, i, est_list = est_list)
                plt.title(crop +' ' + clim+' ' +str(i) +'    ' +  str(est.llr_pvalue))
                plt.show()
                
            coef_bstrp = np.array([est_temp.params['x1'] for est_temp in est_list])
            coef_pval_bstrp_np = (1 - np.max([np.sum(coef_bstrp < 0), np.sum(coef_bstrp > 0)]) / coef_bstrp.shape[0]) * 2
            
            # print('coef t-test p-value: ' + str(est.pvalues['x1']))
            # print('coef bootstrapped p-value: ' + str(coef_pval_bstrp_np))
            
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
                    
    # export results as dictionaries
    os.chdir(path+'research/crop_failures/results/climate_trend2022')
    pickle.dump(model_pval_dict, lzma.open(crop+'_'+clim+'_'+sm_src+'_'+t_src+'_model_pval_v2.pkl.lzma', 'wb'))
    pickle.dump(coef_dict, lzma.open(crop+'_'+clim+'_'+sm_src+'_'+t_src+'_coef_v2.pkl.lzma', 'wb'))
    pickle.dump(coef_pval_dict, lzma.open(crop+'_'+clim+'_'+sm_src+'_'+t_src+'_coef_pval_v2.pkl.lzma', 'wb'))
    pickle.dump(coef_pval_bstrp_dict, lzma.open(crop+'_'+clim+'_'+sm_src+'_'+t_src+'_coef_bstrp_pval_v2.pkl.lzma', 'wb'))

    pickle.dump(ipt_dict, lzma.open(crop+'_'+clim+'_'+sm_src+'_'+t_src+'_ipt_v2.pkl.lzma', 'wb'))
    pickle.dump(ipt_pval_dict, lzma.open(crop+'_'+clim+'_'+sm_src+'_'+t_src+'_ipt_pval_v2.pkl.lzma', 'wb'))
    pickle.dump(sample_size_dict, lzma.open(crop+'_'+clim+'_'+sm_src+'_'+t_src+'_sample_size_v2.pkl.lzma', 'wb'))


if __name__== "__main__":
    
    
    run_location = {'cluster': '/scratch/work/heinom2/',
                    'local_d': 'D:/work/',
                    'local_c': 'C:/Users/heinom2/OneDrive - Aalto University/'}
    
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
            clim_trend_v2(path, rank, t_src, sm_src, irrig, gs, False, only_log_odds = False)
            
            sm_src = 'soil_moisture_gleam'
            clim_trend_v2(path, rank, t_src, sm_src, irrig, gs, False, only_log_odds = False)



