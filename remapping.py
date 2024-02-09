#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 15:17:52 2023

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
from scipy.stats import mannwhitneyu
from matplotlib.backends.backend_pdf import PdfPages    

#%% #%% 

def multipage(filename, figs=None, dpi=200):
    pp = PdfPages(filename)
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()
    
    
#%% 

data_directory = '/media/dhruv/Expansion/Processed'
datasets = np.genfromtxt(os.path.join(data_directory,'remapping_DM.list'), delimiter = '\n', dtype = str, comments = '#')

env_stability = []

halfsession1_corr = []
halfsession2_corr = []

for s in datasets:
    print(s)
    name = s.split('-')[0]
       
    path = os.path.join(data_directory, s)
    
    if name == 'B2618':
        isWT = 0
    else: isWT = 1 

    sp2 = np.load(os.path.join(path, 'spikedata.npz'), allow_pickle = True)
    time_support = nap.IntervalSet(sp2['start'], sp2['end'])
    tsd = nap.Tsd(t=sp2['t'], d=sp2['index'], time_support = time_support)
    spikes = tsd.to_tsgroup()
    spikes.set_info(group = sp2['group'], location = sp2['location'], celltype = sp2['celltype'], tr2pk = sp2['tr2pk'])
    
    data = ntm.load_session(path, 'neurosuite')
    epochs = data.epochs
    position = data.position
    
   #%% Get cells with wake rate more then 0.5Hz
        
    spikes_by_celltype = spikes.getby_category('celltype')
    
    if 'pyr' in spikes._metadata['celltype'].values:
        pyr = spikes_by_celltype['pyr']
    
    keep = []
    
    for i in pyr.index:
        if pyr.restrict(epochs['wake'].loc[[0]])._metadata['rate'][i] > 0.5:
            keep.append(i)

    pyr2 = pyr[keep]
    
#%% Compute speed during wake 

    if len(pyr2) > 2:
        
        speedbinsize = np.diff(position.index.values)[0]
        
        time_bins = np.arange(position.index[0], position.index[-1] + speedbinsize, speedbinsize)
        index = np.digitize(position.index.values, time_bins)
        tmp = position.as_dataframe().groupby(index).mean()
        tmp.index = time_bins[np.unique(index)-1]+(speedbinsize)/2
        distance = np.sqrt(np.power(np.diff(tmp['x']), 2) + np.power(np.diff(tmp['z']), 2)) * 100 #in cm
        speed = nap.Tsd(t = tmp.index.values[0:-1]+ speedbinsize/2, d = distance/speedbinsize) # in cm/s
     
        moving_ep = nap.IntervalSet(speed.threshold(2).time_support) #Epochs in which speed is > 2 cm/s
        ep1 = moving_ep.intersect(epochs['wake'].loc[[0]])
        ep2 = moving_ep.intersect(epochs['wake'].loc[[1]])
           
#%% 
    
    # w1 = nap.IntervalSet(start = epochs['wake'].loc[0]['start'], end = epochs['wake'].loc[0]['end'])
    # w2 = nap.IntervalSet(start = epochs['wake'].loc[1]['start'], end = epochs['wake'].loc[1]['end'])
                                     
    placefields1, binsxy1 = nap.compute_2d_tuning_curves(group = pyr2, 
                                                       feature = position[['x', 'z']], 
                                                       ep = ep1, 
                                                       nb_bins=20)      
    
    placefields2, binsxy2 = nap.compute_2d_tuning_curves(group = pyr2, 
                                                       feature = position[['x', 'z']], 
                                                       ep = ep2, 
                                                       nb_bins=20)
    
    for i in pyr2.keys(): 
        placefields1[i][np.isnan(placefields1[i])] = 0
        placefields1[i] = scipy.ndimage.gaussian_filter(placefields1[i], 1)
        
        placefields2[i][np.isnan(placefields2[i])] = 0
        placefields2[i] = scipy.ndimage.gaussian_filter(placefields2[i], 1)

#%% Plot tracking 

    # plt.figure()
    # plt.suptitle(s)
    # plt.subplot(121)
    # plt.plot(position['x'].restrict(w1), position['z'].restrict(w1))
    # plt.subplot(122)
    # plt.plot(position['x'].restrict(w2), position['z'].restrict(w2))
    
#%% Plot remapping 

    plt.figure()
    plt.suptitle(s + ' Wake1')
    for i,n in enumerate(pyr2):
        plt.subplot(9,8,n+1)
        # plt.title(spikes._metadata['celltype'][i])
        plt.imshow(placefields1[n], extent=(binsxy1[1][0],binsxy1[1][-1],binsxy1[0][0],binsxy1[0][-1]), cmap = 'jet')        
        plt.colorbar()

    plt.figure()
    plt.suptitle(s + ' Wake2')
    for i,n in enumerate(pyr2):
        plt.subplot(9,8,n+1)
        # plt.title(spikes._metadata['celltype'][i])
        plt.imshow(placefields2[n], extent=(binsxy2[1][0],binsxy2[1][-1],binsxy2[0][0],binsxy2[0][-1]), cmap = 'jet')        
        plt.colorbar()
    
#%% Split both wake wpochs into halves 

    center1 = position.restrict(epochs['wake'].loc[[0]]).time_support.get_intervals_center()
    center2 = position.restrict(epochs['wake'].loc[[1]]).time_support.get_intervals_center()
    
    halves1 = nap.IntervalSet( start = [position.restrict(epochs['wake'].loc[[0]]).time_support.start[0], center1.t[0]],
                              end = [center1.t[0], position.restrict(epochs['wake'].loc[[0]]).time_support.end[0]])

    halves2 = nap.IntervalSet( start = [position.restrict(epochs['wake'].loc[[1]]).time_support.start[0], center2.t[0]],
                              end = [center2.t[0], position.restrict(epochs['wake'].loc[[1]]).time_support.end[0]])

    
    ep_wake1 = halves1.intersect(moving_ep)
    ep_wake2 = halves2.intersect(moving_ep)
    
    half1_wake1 = ep_wake1.loc[0:len(ep_wake1)/2]
    half2_wake1 = ep_wake1.loc[(len(ep_wake1)/2)+1:]
    
    half1_wake2 = ep_wake2.loc[0:len(ep_wake2)/2]
    half2_wake2 = ep_wake2.loc[(len(ep_wake2)/2)+1:]
        
    pf1, binsxy = nap.compute_2d_tuning_curves(group = pyr2, feature = position[['x', 'z']], ep = half1_wake1, nb_bins=20)  
    pf2, binsxy = nap.compute_2d_tuning_curves(group = pyr2, feature = position[['x', 'z']], ep = half2_wake1, nb_bins=20)  

    pf3, binsxy = nap.compute_2d_tuning_curves(group = pyr2, feature = position[['x', 'z']], ep = half1_wake2, nb_bins=20)  
    pf4, binsxy = nap.compute_2d_tuning_curves(group = pyr2, feature = position[['x', 'z']], ep = half2_wake2, nb_bins=20)  

        
#%% Quantify spatial maps between 2 environments 

    for k in pyr2:
        
        ###Between 2 environments
        good = np.logical_and(np.isfinite(placefields1[k].flatten()), np.isfinite(placefields2[k].flatten()))
        corr, p = scipy.stats.pearsonr(placefields1[k].flatten()[good], placefields2[k].flatten()[good]) 
        
        env_stability.append(corr)
        
        ###Between 2 halves of first wake 
        good2 = np.logical_and(np.isfinite(pf1[k].flatten()), np.isfinite(pf2[k].flatten()))
        corr2, p2 = scipy.stats.pearsonr(pf1[k].flatten()[good2], pf2[k].flatten()[good2]) 
        
        halfsession1_corr.append(corr2)
        
        ###Between 2 halves of second wake 
        good3 = np.logical_and(np.isfinite(pf3[k].flatten()), np.isfinite(pf4[k].flatten()))
        corr3, p3 = scipy.stats.pearsonr(pf3[k].flatten()[good3], pf4[k].flatten()[good3]) 
        
        halfsession2_corr.append(corr3)
        
        
        
#%% Organize and plot environment stability data 
    
env = np.array(['A v/s B' for x in range(len(env_stability))])
h1 = np.array(['A1 v/s B1' for x in range(len(halfsession1_corr))])
h2 = np.array(['A2 v/s B2' for x in range(len(halfsession2_corr))])

corrtype = np.hstack([env, h1, h2])

sinfos = []
sinfos.extend(env_stability)
sinfos.extend(halfsession1_corr)
sinfos.extend(halfsession2_corr)

allinfos = pd.DataFrame(data = [sinfos, corrtype], index = ['corr', 'type']).T

plt.figure()
plt.title('Remapping Quantification')
sns.set_style('white')
palette = ['orangered', 'coral', 'darksalmon'] 
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
sns.stripplot(x = allinfos['type'], y = allinfos['corr'].astype(float), data = allinfos, color = 'k', dodge=False, ax=ax)
# sns.swarmplot(x = wakedf['type'], y = wakedf['rate'].astype(float), data = wakedf, color = 'k', dodge=False, ax=ax)
for dots in ax.collections[old_len_collections:]:
    dots.set_offsets(dots.get_offsets())
ax.set_xlim(xlim)
ax.set_ylim(ylim)
plt.ylabel('Spatial Map Correlation')
ax.set_box_aspect(1)
    
# t, p = mannwhitneyu(env_stability_wt, env_stability_ko)
    
        
#%% Plot Example cells 

# examples = [5,8,10,11]

# for n in examples:
#     plt.figure()
#     peakfreq = max(placefields1[n].max(), placefields2[n].max()) 
#     pf1 = placefields1[n] / peakfreq
#     pf2 = placefields2[n] / peakfreq
    
    
#     plt.subplot(1,2,1)
#     plt.imshow(pf1.T, cmap = 'viridis', aspect = 'auto', origin = 'lower', vmin = 0, vmax = 1)   
#     plt.tight_layout()
#     plt.gca().set_box_aspect(1)
#     plt.subplot(1,2,2)
#     plt.imshow(pf2.T, cmap = 'viridis', aspect = 'auto', origin = 'lower', vmin = 0, vmax = 1)   
#     # plt.colorbar()
#     plt.gca().set_box_aspect(1)
#     plt.tight_layout()
    
    
# plt.figure()
# plt.subplot(121)
# plt.plot(position['x'].restrict(w1), position['z'].restrict(w1), color = 'grey')
# spk_pos1 = spikes[examples[0]].value_from(position.restrict(w1))
# plt.plot(spk_pos1['x'], spk_pos1['z'], 'o', color = 'r', markersize = 5, alpha = 0.5)
# plt.gca().set_box_aspect(1)
# plt.subplot(122)
# plt.plot(position['x'].restrict(w2), position['z'].restrict(w2), color = 'grey')
# spk_pos2 = spikes[examples[0]].value_from(position.restrict(w2))
# plt.plot(spk_pos2['x'], spk_pos2['z'], 'o', color = 'r', markersize = 5, alpha = 0.5)
# plt.gca().set_box_aspect(1)    
    
    