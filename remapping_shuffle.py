#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 09:27:12 2024

@author: dhruv
"""

import numpy as np 
import pandas as pd 
import nwbmatic as ntm
import scipy.io
import pynapple as nap 
import os, sys
import matplotlib.pyplot as plt 
import seaborn as sns
import warnings
from scipy.stats import mannwhitneyu, wilcoxon
from functions_DM import *

#%% 

warnings.filterwarnings("ignore")

# data_directory = '/media/dhruv/Expansion/Processed'
data_directory = '/media/dhruv/Expansion/Processed/CA3'
# data_directory = '/media/adrien/Expansion/Processed'
# datasets = np.genfromtxt(os.path.join(data_directory,'remapping_DM.list'), delimiter = '\n', dtype = str, comments = '#')
# datasets = np.genfromtxt(os.path.join(data_directory,'remapping_2sq.list'), delimiter = '\n', dtype = str, comments = '#')
datasets = np.genfromtxt(os.path.join(data_directory,'remapping_CA3.list'), delimiter = '\n', dtype = str, comments = '#')

shuffs = []

pvcorr_wt_shu = []
pvcorr_ko_shu = []

pvsess_wt = []
pvsess_ko = []

env_stability_wt_shu = []
env_stability_ko_shu = []

sess_wt = []
sess_ko = []

for s in datasets:
    print(s)
    name = s.split('-')[0]
       
    path = os.path.join(data_directory, s)
    
    if name == 'B2618' or name == 'B2627' or name == 'B2628' or name == 'B3805' or name == 'B3813':
        isWT = 0
    else: isWT = 1 

    sp2 = np.load(os.path.join(path, 'spikedata_0.55.npz'), allow_pickle = True)
    time_support = nap.IntervalSet(sp2['start'], sp2['end'])
    tsd = nap.Tsd(t=sp2['t'], d=sp2['index'], time_support = time_support)
    spikes = tsd.to_tsgroup()
    spikes.set_info(group = sp2['group'], location = sp2['location'], celltype = sp2['celltype'], tr2pk = sp2['tr2pk'])
    
    data = ntm.load_session(path, 'neurosuite')
    epochs = data.epochs
    position = data.position
    
#%% Rotate position 

    rot_pos = []
        
    xypos = np.array(position[['x', 'z']])
    
    if name == 'B2625' or name == 'B3800':
        rad = 0.6
    elif name == 'B2618' :
        rad = 0.95
    elif s == 'B2627-240528' or s == 'B2627-240530' or name == 'B3804' or name == 'B3807' or name == 'B3813':
        rad = 0.05
    elif name == 'B3811':
        rad = 0.1
    else: rad = 0    
       
    for i in range(len(xypos)):
        newx, newy = rotate_via_numpy(xypos[i], rad)
        rot_pos.append((newx, newy))
        
    rot_pos = nap.TsdFrame(t = position.index.values, d = rot_pos, columns = ['x', 'z'])
    
    w1 = nap.IntervalSet(start = epochs['wake'][0]['start'], end = epochs['wake'][0]['end'])
    w2 = nap.IntervalSet(start = epochs['wake'][1]['start'], end = epochs['wake'][1]['end'])
    
#%% Get cells with wake rate more than 0.5Hz
        
    spikes_by_celltype = spikes.getby_category('celltype')
    
    if 'pyr' in spikes._metadata['celltype'].values:
        pyr = spikes_by_celltype['pyr']
    
    keep = []
    
    for i in pyr.index:
        if pyr.restrict(nap.IntervalSet(epochs['wake'][0]))._metadata['rate'][i] > 0.5:
            keep.append(i)

    pyr2 = pyr[keep]
    # print(len(pyr2))
    
#%% Compute speed during wake 

    if len(pyr2) > 2:
        
        speedbinsize = np.diff(rot_pos.index.values)[0]
        
        time_bins = np.arange(rot_pos.index[0], rot_pos.index[-1] + speedbinsize, speedbinsize)
        index = np.digitize(rot_pos.index.values, time_bins)
        tmp = rot_pos.as_dataframe().groupby(index).mean()
        tmp.index = time_bins[np.unique(index)-1]+(speedbinsize)/2
        distance = np.sqrt(np.power(np.diff(tmp['x']), 2) + np.power(np.diff(tmp['z']), 2)) * 100 #in cm
        speed = pd.Series(index = tmp.index.values[0:-1]+ speedbinsize/2, data = distance/speedbinsize) # in cm/s
        speed2 = speed.rolling(window = 25, win_type='gaussian', center=True, min_periods=1).mean(std=10) #Smooth over 200ms 
        speed2 = nap.Tsd(speed2)
             
        moving_ep = nap.IntervalSet(speed2.threshold(2).time_support) #Epochs in which speed is > 2 cm/s
        
        ep1 = moving_ep.intersect(epochs['wake'].loc[[0]])
        ep2 = moving_ep.intersect(epochs['wake'].loc[[1]])
        
#%% Odd-eve epoch correlation

        # oddep = ep1[1::2]
        # evenep = ep1[::2]
        
        # pf1, binsxy = nap.compute_2d_tuning_curves(group = pyr2, features = rot_pos[['x', 'z']], ep = oddep, nb_bins=24)  
        # pxx1 = occupancy_prob(rot_pos, oddep, nb_bins=24, norm = True)
        # SI1 = nap.compute_2d_mutual_info(pf1, rot_pos[['x', 'z']], ep = oddep)
        
        # pf2, binsxy = nap.compute_2d_tuning_curves(group = pyr2, features = rot_pos[['x', 'z']], ep = evenep, nb_bins=24)  
        # pxx2 = occupancy_prob(rot_pos, evenep, nb_bins=24, norm = True)  
        # SI2 = nap.compute_2d_mutual_info(pf2, rot_pos[['x', 'z']], ep = evenep)
        
        # for i in pyr2.keys(): 
            
        #     pf1[i][np.isnan(pf1[i])] = 0
        #     pf1[i] = scipy.ndimage.gaussian_filter(pf1[i], 1.5, mode = 'nearest')
        #     masked_array = np.ma.masked_where(pxx1 == 0, pf1[i]) #should work fine without it 
        #     pf1[i] = masked_array
            
        #     pf2[i][np.isnan(pf2[i])] = 0
        #     pf2[i] = scipy.ndimage.gaussian_filter(pf2[i], 1.5, mode = 'nearest')
        #     masked_array = np.ma.masked_where(pxx2 == 0, pf2[i]) #should work fine without it 
        #     pf2[i] = masked_array
                
        # keep = []
        # oddeven = []
        
        # for k in pyr2:
        
        #     good = np.logical_and(np.isfinite(pf1[k].flatten()), np.isfinite(pf2[k].flatten()))
        #     corr, p = scipy.stats.pearsonr(pf1[k].flatten()[good], pf2[k].flatten()[good]) 
        #     oddeven.append(corr)
                   
        #     if corr > 0.65: 
        #         keep.append(k)
                    
        # pyr3 = pyr2[keep]
        pyr3 = pyr2 
    
#%% Reverse the positions
    
        if len(pyr3) > 0: 
            revpos1 = pd.DataFrame()
        
            for i in range(len(ep1)):
                pos = rot_pos[['x', 'z']].restrict(ep1[i]).as_dataframe()
                pos[:] = pos[::-1]
                revpos1 = pd.concat([revpos1, pos])
             
            revpos1 = nap.TsdFrame(revpos1)
            
            revpos2 = pd.DataFrame()
        
            for i in range(len(ep2)):
                pos = rot_pos[['x', 'z']].restrict(ep2[i]).as_dataframe()
                pos[:] = pos[::-1]
                revpos2 = pd.concat([revpos2, pos])
             
            revpos2 = nap.TsdFrame(revpos2)
            
    
#%% Compute shuffles for spatial info
       
    # for i in range(100): 
    #     spk = shuffleByCircularSpikes(pyr2, ep1)
    #     pf, binsxy = nap.compute_2d_tuning_curves(group = spk, features = rot_pos[['x', 'z']], ep = ep1, nb_bins=24)
    #     SI = nap.compute_2d_mutual_info(pf, rot_pos[['x', 'z']], ep = ep1)
    #     shuffs.extend(SI['SI'].tolist())
    
    #     pf, binsxy = nap.compute_2d_tuning_curves(group = pyr2, features = revpos[['x', 'z']], ep = ep1, nb_bins=24)
    #     SI = nap.compute_2d_mutual_info(pf, rot_pos[['x', 'z']], ep = ep1)
    #     shuffs.extend(SI['SI'].tolist())
        
    # pf, binsxy = nap.compute_2d_tuning_curves(group = pyr2, features = revpos1, ep = ep1, nb_bins=24)
    # SI = nap.compute_2d_mutual_info(pf, revpos1, ep = ep1)
    # shuffs.extend(SI['SI'].tolist())
        
#%% Compute shuffle for stability 

            oddep = ep1[1::2]
            evenep = ep1[::2]
                       
            pf1, binsxy = nap.compute_2d_tuning_curves(group = pyr2, features = revpos1[['x', 'z']], ep = oddep, nb_bins=24)  
            pxx1 = occupancy_prob(rot_pos, oddep, nb_bins=24, norm = True)
            # SI1 = nap.compute_2d_mutual_info(pf1, revpos1[['x', 'z']], ep = oddep)
            
            pf2, binsxy = nap.compute_2d_tuning_curves(group = pyr2, features = revpos1[['x', 'z']], ep = evenep, nb_bins=24)  
            pxx2 = occupancy_prob(rot_pos, evenep, nb_bins=24, norm = True)  
            # SI2 = nap.compute_2d_mutual_info(pf2, revpos1[['x', 'z']], ep = evenep)
            
            for i in pyr3.keys(): 
                
                pf1[i][np.isnan(pf1[i])] = 0
                pf1[i] = scipy.ndimage.gaussian_filter(pf1[i], 1.5, mode = 'nearest')
                masked_array = np.ma.masked_where(pxx1 == 0, pf1[i]) #should work fine without it 
                pf1[i] = masked_array
                
                pf2[i][np.isnan(pf2[i])] = 0
                pf2[i] = scipy.ndimage.gaussian_filter(pf2[i], 1.5, mode = 'nearest')
                masked_array = np.ma.masked_where(pxx2 == 0, pf2[i]) #should work fine without it 
                pf2[i] = masked_array
                   
            for k in pyr3:
            
                good = np.logical_and(np.isfinite(pf1[k].flatten()), np.isfinite(pf2[k].flatten()))
                corr, p = scipy.stats.pearsonr(pf1[k].flatten()[good], pf2[k].flatten()[good]) 
                shuffs.append(corr)
                
        
#%%        

# bins = np.linspace(-0.4, 1, 30) ### CA1 
bins = np.linspace(-0.2, 0.7, 30) ### CA3
xcenters = np.mean(np.vstack([bins[:-1], bins[1:]]), axis = 0)
counts, _ = np.histogram(shuffs, bins)
prop = counts/sum(counts)

plt.figure()
plt.title('Shuffle Distribution')
plt.stairs(prop, bins, color = 'k' , linewidth = 2)
plt.axvline(np.percentile(shuffs,95), color = 'silver', linestyle = '--')
plt.xlabel('Odd-Even Correlation (R)')
plt.ylabel('% cells')
plt.gca().set_box_aspect(1)

        

#%% Use reversed position to quantify remapping 

    
        
            # placefields1, binsxy1 = nap.compute_2d_tuning_curves(group = pyr3, 
            #                                                     features = revpos1[['x', 'z']], 
            #                                                     ep = ep1, 
            #                                                     nb_bins=24)  
                   
            # px1 = occupancy_prob(revpos1, ep1, nb_bins=24, norm = True)
            
            
            # placefields2, binsxy2 = nap.compute_2d_tuning_curves(group = pyr3, 
            #                                                     features = revpos2[['x', 'z']], 
            #                                                     ep = ep2, 
            #                                                     nb_bins=24)
                    
            # px2 = occupancy_prob(revpos2, ep2, nb_bins=24, norm = True)
           
            
            # for i in pyr3.keys(): 
                
            #     placefields1[i][np.isnan(placefields1[i])] = 0
            #     placefields1[i] = scipy.ndimage.gaussian_filter(placefields1[i], 1.5, mode = 'nearest')
            #     masked_array = np.ma.masked_where(px1 == 0, placefields1[i]) #should work fine without it 
            #     placefields1[i] = masked_array
                
            #     placefields2[i][np.isnan(placefields2[i])] = 0
            #     placefields2[i] = scipy.ndimage.gaussian_filter(placefields2[i], 1.5, mode = 'nearest')
            #     masked_array = np.ma.masked_where(px2 == 0, placefields2[i]) #should work fine without it 
            #     placefields2[i] = masked_array
                
            # normpf1 = {}
            # for i in placefields1.keys():
            #     normpf1[i] = placefields1[i] / np.max(placefields1[i])
                
            # normpf2 = {}
            # for i in placefields2.keys():
            #     normpf2[i] = placefields2[i] / np.max(placefields2[i])
                
            # pvcorr = compute_PVcorrs(normpf1, normpf2, pyr3.index)
            
            # if isWT == 1:
            #     pvcorr_wt_shu.append(pvcorr)
            #     pvsess_wt.append(name)
            # else: 
            #     pvcorr_ko_shu.append(pvcorr)
            #     pvsess_ko.append(name)
        
            # for k in pyr3:
            #     good = np.logical_and(np.isfinite(placefields1[k].flatten()), np.isfinite(placefields2[k].flatten()))
            #     corr, p = scipy.stats.pearsonr(placefields1[k].flatten()[good], placefields2[k].flatten()[good]) 
            
            #     if isWT == 1:
            #         env_stability_wt_shu.append(corr)
            #         sess_wt.append(s)
            #     else: 
            #         env_stability_ko_shu.append(corr)
            #         sess_ko.append(s)

#%% Organize remapping data
    
# wt = np.array(['WT' for x in range(len(env_stability_wt_shu))])
# ko = np.array(['KO' for x in range(len(env_stability_ko_shu))])

# genotype = np.hstack([wt, ko])

# sinfos = []
# sinfos.extend(env_stability_wt_shu)
# sinfos.extend(env_stability_ko_shu)

# sinfos2 = []
# sinfos2.extend(sess_wt)
# sinfos2.extend(sess_ko)


# allinfos = pd.DataFrame(data = [sinfos, genotype, sinfos2], index = ['corr', 'type', 'sess']).T

# color_labels = allinfos['sess'].unique()
# rgb_values = sns.color_palette("viridis", len(color_labels))
# color_map = dict(zip(color_labels, rgb_values))


#%% 

# plt.figure()
# plt.title('Remapping: A v/s B')
# sns.set_style('white')
# palette = ['royalblue', 'indianred']
# ax = sns.violinplot( x = allinfos['type'], y=allinfos['corr'].astype(float) , data = allinfos, dodge=False,
#                     palette = palette,cut = 2,
#                     scale="width", inner=None)
# ax.tick_params(bottom=True, left=True)
# xlim = ax.get_xlim()
# ylim = ax.get_ylim()
# for violin in ax.collections:
#     x0, y0, width, height = violin.get_paths()[0].get_extents().bounds
#     violin.set_clip_path(plt.Rectangle((x0, y0), width / 2, height, transform=ax.transData))
# sns.boxplot(x = allinfos['type'], y=allinfos['corr'].astype(float) , data = allinfos, saturation=1, showfliers=False,
#             width=0.3, boxprops={'zorder': 3, 'facecolor': 'none'}, ax=ax)
# old_len_collections = len(ax.collections)
# sns.stripplot(x = allinfos['type'], y = allinfos['corr'].astype(float), data = allinfos, 
#               hue = allinfos['sess'], palette=sns.color_palette('inferno_r', len(allinfos['sess'].unique())), 
#                                                                 dodge=False, ax=ax, legend=False)
# # sns.swarmplot(x = durdf['genotype'], y = durdf['dur'].astype(float), data = durdf, color = 'k', dodge=False, ax=ax)
# for dots in ax.collections[old_len_collections:]:
#     dots.set_offsets(dots.get_offsets())
# ax.set_xlim(xlim)
# ax.set_ylim(ylim)
# plt.ylabel('Correlation (R)')
# plt.axhline(0, linestyle = '--', color = 'silver')
# ax.set_box_aspect(1)

#%% Organize PV corr data 

# wt1 = np.array(['WT' for x in range(len(pvcorr_wt_shu))])
# ko1 = np.array(['KO' for x in range(len(pvcorr_ko_shu))])

# genotype = np.hstack([wt1, ko1])

# sinfos1 = []
# sinfos1.extend(pvcorr_wt_shu)
# sinfos1.extend(pvcorr_ko_shu)

# allinfos3 = pd.DataFrame(data = [sinfos1, genotype], index = ['PVCorr', 'genotype']).T

#%% Plotting PV corr

# plt.figure()
# plt.title('Population vector correlation')
# sns.set_style('white')
# palette = ['royalblue', 'indianred'] 
# ax = sns.violinplot( x = allinfos3['genotype'], y=allinfos3['PVCorr'].astype(float) , data = allinfos3, dodge=False,
#                     palette = palette,cut = 2,
#                     scale="width", inner=None)
# ax.tick_params(bottom=True, left=True)
# xlim = ax.get_xlim()
# ylim = ax.get_ylim()
# for violin in ax.collections:
#     x0, y0, width, height = violin.get_paths()[0].get_extents().bounds
#     violin.set_clip_path(plt.Rectangle((x0, y0), width / 2, height, transform=ax.transData))
# sns.boxplot(x = allinfos3['genotype'], y=allinfos3['PVCorr'].astype(float) , data = allinfos3, saturation=1, showfliers=False,
#             width=0.3, boxprops={'zorder': 3, 'facecolor': 'none'}, ax=ax)
# old_len_collections = len(ax.collections)
# sns.stripplot(x = allinfos3['genotype'], y = allinfos3['PVCorr'].astype(float), data = allinfos3, color = 'k', dodge=False, ax=ax, alpha = 0.2)
# # sns.swarmplot(x = wakedf['type'], y = wakedf['rate'].astype(float), data = wakedf, color = 'k', dodge=False, ax=ax)
# for dots in ax.collections[old_len_collections:]:
#     dots.set_offsets(dots.get_offsets())
# ax.set_xlim(xlim)
# ax.set_ylim(ylim)
# plt.ylabel('Correlation (R)')
# plt.axhline(0, linestyle = '--', color = 'silver')
# ax.set_box_aspect(1)

#%% Stats 

# ### Remapping
# z_wt, p_wt = wilcoxon(np.array(env_stability_wt_shu)-0)
# z_ko, p_ko = wilcoxon(np.array(env_stability_ko_shu)-0)

# ### PV corr
# z_pvcorr_wt, p_pvcorr_wt = wilcoxon(np.array(pvcorr_wt_shu)-0)
# z_pvcorr_ko, p_pvcorr_ko = wilcoxon(np.array(pvcorr_ko_shu)-0)
            
#%% SI shuffle values 
    

#%% Stability shuffle values 

# 95th percentile odd-even = 0.53 (for data up to B3807)
# 99th percentile odd-even = 0.65
