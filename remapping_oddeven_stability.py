#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 11:58:50 2025

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

stability_corr_wt1 = []
stability_corr_ko1 = []

stability_corr_wt2 = []
stability_corr_ko2 = []


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
        
#%% Odd-even epoch correlation
        
        oddep1 = ep1[1::2]
        evenep1 = ep1[::2]
        
        oddep2 = ep2[1::2]
        evenep2 = ep2[::2]
        
        pf1, binsxy = nap.compute_2d_tuning_curves(group = pyr2, features = rot_pos[['x', 'z']], ep = oddep1, nb_bins=24)  
        pxx1 = occupancy_prob(rot_pos, oddep1, nb_bins=24, norm = True)
        SI1 = nap.compute_2d_mutual_info(pf1, rot_pos[['x', 'z']], ep = oddep1)
        
        pf2, binsxy = nap.compute_2d_tuning_curves(group = pyr2, features = rot_pos[['x', 'z']], ep = evenep1, nb_bins=24)  
        pxx2 = occupancy_prob(rot_pos, evenep1, nb_bins=24, norm = True)  
        SI2 = nap.compute_2d_mutual_info(pf2, rot_pos[['x', 'z']], ep = evenep1)
        
        keep = []
        oddeven = []
        
        for i in pyr2.keys(): 
            
            pf1[i][np.isnan(pf1[i])] = 0
            pf1[i] = scipy.ndimage.gaussian_filter(pf1[i], 1.5, mode = 'nearest')
            masked_array = np.ma.masked_where(pxx1 == 0, pf1[i]) #should work fine without it 
            pf1[i] = masked_array
            
            pf2[i][np.isnan(pf2[i])] = 0
            pf2[i] = scipy.ndimage.gaussian_filter(pf2[i], 1.5, mode = 'nearest')
            masked_array = np.ma.masked_where(pxx2 == 0, pf2[i]) #should work fine without it 
            pf2[i] = masked_array
        
        for k in pyr2:
        
            good = np.logical_and(np.isfinite(pf1[k].flatten()), np.isfinite(pf2[k].flatten()))
            corr, p = scipy.stats.pearsonr(pf1[k].flatten()[good], pf2[k].flatten()[good]) 
            oddeven.append(corr)
                     
            # if corr > 0.53: ### CA1
            if corr > 0.54: ### CA3 
                keep.append(k)
                    
        pyr3 = pyr2[keep]

#%% 
      
        
        if len(pyr3) > 0:     
        #     print('yes')
    
            placefields1, binsxy1 = nap.compute_2d_tuning_curves(group = pyr3, 
                                                                features = rot_pos[['x', 'z']], 
                                                                ep = oddep1, 
                                                                nb_bins=24)  
                   
            px1 = occupancy_prob(rot_pos, oddep1, nb_bins=24, norm = True)
            spatialinfo_env1 = nap.compute_2d_mutual_info(placefields1, rot_pos[['x', 'z']], ep = oddep1)
            
            placefields2, binsxy2 = nap.compute_2d_tuning_curves(group = pyr3, 
                                                                features = rot_pos[['x', 'z']], 
                                                                ep = evenep1, 
                                                                nb_bins=24)
                    
            px2 = occupancy_prob(rot_pos, evenep1, nb_bins=24, norm = True)
            spatialinfo_env2 = nap.compute_2d_mutual_info(placefields2, rot_pos[['x', 'z']], ep = evenep1)   
          
            placefields3, binsxy3 = nap.compute_2d_tuning_curves(group = pyr3, 
                                                                features = rot_pos[['x', 'z']], 
                                                                ep = oddep2, 
                                                                nb_bins=24)  
                   
            px3 = occupancy_prob(rot_pos, oddep2, nb_bins=24, norm = True)
            spatialinfo_env3 = nap.compute_2d_mutual_info(placefields3, rot_pos[['x', 'z']], ep = oddep2)
            
            placefields4, binsxy4 = nap.compute_2d_tuning_curves(group = pyr3, 
                                                                features = rot_pos[['x', 'z']], 
                                                                ep = evenep2, 
                                                                nb_bins=24)
                    
            px4 = occupancy_prob(rot_pos, evenep2, nb_bins=24, norm = True)
            spatialinfo_env4 = nap.compute_2d_mutual_info(placefields2, rot_pos[['x', 'z']], ep = evenep2)
                        
            
            for i in pyr3.keys(): 
                
                placefields1[i][np.isnan(placefields1[i])] = 0
                placefields1[i] = scipy.ndimage.gaussian_filter(placefields1[i], 1.5, mode = 'nearest')
                masked_array = np.ma.masked_where(px1 == 0, placefields1[i]) #should work fine without it 
                placefields1[i] = masked_array
                
                placefields2[i][np.isnan(placefields2[i])] = 0
                placefields2[i] = scipy.ndimage.gaussian_filter(placefields2[i], 1.5, mode = 'nearest')
                masked_array = np.ma.masked_where(px2 == 0, placefields2[i]) #should work fine without it 
                placefields2[i] = masked_array
                
                placefields3[i][np.isnan(placefields3[i])] = 0
                placefields3[i] = scipy.ndimage.gaussian_filter(placefields3[i], 1.5, mode = 'nearest')
                masked_array = np.ma.masked_where(px3 == 0, placefields3[i]) #should work fine without it 
                placefields3[i] = masked_array
                
                placefields4[i][np.isnan(placefields4[i])] = 0
                placefields4[i] = scipy.ndimage.gaussian_filter(placefields4[i], 1.5, mode = 'nearest')
                masked_array = np.ma.masked_where(px2 == 0, placefields4[i]) #should work fine without it 
                placefields4[i] = masked_array
                
           
                
            for k in pyr3:
                
                good = np.logical_and(np.isfinite(placefields1[k].flatten()), np.isfinite(placefields2[k].flatten()))
                corr, p = scipy.stats.pearsonr(placefields1[k].flatten()[good], placefields2[k].flatten()[good]) 
                               
                if isWT == 1:
                    stability_corr_wt1.append(corr)
                else: stability_corr_ko1.append(corr)
                
                good = np.logical_and(np.isfinite(placefields3[k].flatten()), np.isfinite(placefields4[k].flatten()))
                corr, p = scipy.stats.pearsonr(placefields3[k].flatten()[good], placefields4[k].flatten()[good]) 
                               
                if isWT == 1:
                    stability_corr_wt2.append(corr)
                else: stability_corr_ko2.append(corr)
            
#%% 

wt = np.array(['WT' for x in range(len(stability_corr_wt1))])
ko = np.array(['KO' for x in range(len(stability_corr_ko1))])

genotype = np.hstack([wt, ko])

sinfos = []
sinfos.extend(stability_corr_wt1)
sinfos.extend(stability_corr_ko1)

allinfos = pd.DataFrame(data = [sinfos, genotype], index = ['corr', 'type']).T

wt = np.array(['WT' for x in range(len(stability_corr_wt2))])
ko = np.array(['KO' for x in range(len(stability_corr_ko2))])

genotype2 = np.hstack([wt, ko])

sinfos2 = []
sinfos2.extend(stability_corr_wt2)
sinfos2.extend(stability_corr_ko2)

allinfos2 = pd.DataFrame(data = [sinfos2, genotype2], index = ['corr', 'type']).T


#%% 

plt.figure()
plt.suptitle('Odd v/s Even Epoch Correlation')

# plt.subplot(121)
plt.title('Arena A')
sns.set_style('white')
palette = ['royalblue', 'indianred']
ax = sns.violinplot( x = allinfos['type'], y=allinfos['corr'].astype(float) , data = allinfos, dodge=False,
                    palette = palette,cut = 2,
                    scale="width", inner=None)
ax.tick_params(bottom=True, left=True)
xlim = ax.get_xlim()
ylim = ax.get_ylim()
for violin in ax.collections:
    x0, y0, width, height = violin.get_paths()[0].get_extents().bounds
    violin.set_clip_path(plt.Rectangle((x0, y0), width / 2, height, transform=ax.transData))
sns.boxplot(x = allinfos['type'], y=allinfos['corr'].astype(float) , data = allinfos, saturation=1, showfliers=False,
            width=0.3, boxprops={'zorder': 3, 'facecolor': 'none'}, ax=ax)
old_len_collections = len(ax.collections)
sns.stripplot(x = allinfos['type'], y = allinfos['corr'].astype(float), data = allinfos, color = 'k', dodge=False, ax=ax, legend=False)
# sns.swarmplot(x = durdf['genotype'], y = durdf['dur'].astype(float), data = durdf, color = 'k', dodge=False, ax=ax)
for dots in ax.collections[old_len_collections:]:
    dots.set_offsets(dots.get_offsets())
ax.set_xlim(xlim)
ax.set_ylim(ylim)
plt.ylabel('Correlation (R)')
ax.set_box_aspect(1)

# plt.subplot(122)
# plt.title('Arena B')
# sns.set_style('white')
# palette = ['royalblue', 'indianred']
# ax = sns.violinplot( x = allinfos2['type'], y=allinfos2['corr'].astype(float) , data = allinfos2, dodge=False,
#                     palette = palette,cut = 2,
#                     scale="width", inner=None)
# ax.tick_params(bottom=True, left=True)
# xlim = ax.get_xlim()
# ylim = ax.get_ylim()
# for violin in ax.collections:
#     x0, y0, width, height = violin.get_paths()[0].get_extents().bounds
#     violin.set_clip_path(plt.Rectangle((x0, y0), width / 2, height, transform=ax.transData))
# sns.boxplot(x = allinfos2['type'], y=allinfos2['corr'].astype(float) , data = allinfos2, saturation=1, showfliers=False,
#             width=0.3, boxprops={'zorder': 3, 'facecolor': 'none'}, ax=ax)
# old_len_collections = len(ax.collections)
# sns.stripplot(x = allinfos2['type'], y = allinfos2['corr'].astype(float), data = allinfos2, color = 'k', dodge=False, ax=ax, legend=False)
# # sns.swarmplot(x = durdf['genotype'], y = durdf['dur'].astype(float), data = durdf, color = 'k', dodge=False, ax=ax)
# for dots in ax.collections[old_len_collections:]:
#     dots.set_offsets(dots.get_offsets())
# ax.set_xlim(xlim)
# ax.set_ylim(ylim)
# plt.ylabel('Correlation (R)')
# ax.set_box_aspect(1)



#%% 

t1, p1 = mannwhitneyu(stability_corr_wt1, stability_corr_ko1)
t2, p2 = mannwhitneyu(stability_corr_wt2, stability_corr_ko2)

