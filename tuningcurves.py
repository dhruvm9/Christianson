#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 15:44:48 2023

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
from scipy.stats import mannwhitneyu, pearsonr
from matplotlib.backends.backend_pdf import PdfPages    

#%% 

def multipage(filename, figs=None, dpi=200):
    pp = PdfPages(filename)
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()
    
def rotate_via_numpy(xy, radians):
    """xy is a tuple or array """
    x, y = xy
    c, s = np.cos(radians), np.sin(radians)
    j = np.array([[c, s], [-s, c]])
    m = np.dot(j, [x, y])

    return float(m.T[0]), float(m.T[1])
    
#%% 

data_directory = '/media/dhruv/Expansion/Processed'
datasets = np.genfromtxt(os.path.join(data_directory,'dataset_DM.list'), delimiter = '\n', dtype = str, comments = '#')

allspatialinfo_wt = []
allspatialinfo_ko = []

placefield_stability_wt = []
placefield_stability_ko = []

for s in datasets:
    print(s)
    name = s.split('-')[0]
       
    path = os.path.join(data_directory, s)
    
    if name == 'B2613' or name == 'B2618':
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

#%% Rotate position 

    rot_pos = []
        
    xypos = np.array(position[['x', 'z']])
      
    for i in range(len(xypos)):
        newx, newy = rotate_via_numpy(xypos[i], 1.05)
        rot_pos.append((newx, newy))
        
    rot_pos = nap.TsdFrame(t = position.index.values, d = rot_pos, columns = ['x', 'z'])
                                     
#%% Get cells with wake rate more then 0.5Hz
        
    spikes_by_celltype = spikes.getby_category('celltype')
    
    if 'pyr' in spikes._metadata['celltype'].values:
        pyr = spikes_by_celltype['pyr']
    
    keep = []
    
    for i in pyr.index:
        if pyr.restrict(nap.IntervalSet(epochs['wake'].loc[[0]]))._metadata['rate'][i] > 0.5:
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
        speed = nap.Tsd(t = tmp.index.values[0:-1]+ speedbinsize/2, d = distance/speedbinsize) # in cm/s
     
        moving_ep = nap.IntervalSet(speed.threshold(2).time_support) #Epochs in which speed is > 2 cm/s
        ep = moving_ep.intersect(epochs['wake'].loc[[0]])
        
#%% 
    
    placefields, binsxy = nap.compute_2d_tuning_curves(group = pyr2, features = rot_pos, ep = ep, nb_bins=20)                                               
    
    spatialinfo = nap.compute_2d_mutual_info(placefields, rot_pos, ep = ep)
    
    if isWT == 1:
        allspatialinfo_wt.extend(spatialinfo['SI'].tolist())
    else: allspatialinfo_ko.extend(spatialinfo['SI'].tolist())

    for i in pyr2.keys(): 
        placefields[i][np.isnan(placefields[i])] = 0
        placefields[i] = scipy.ndimage.gaussian_filter(placefields[i], 1)

    
    
#%%  Place field stability 

    center = rot_pos.restrict(nap.IntervalSet(epochs['wake'].loc[[0]])).time_support.get_intervals_center()
    
    halves = nap.IntervalSet( start = [rot_pos.restrict(nap.IntervalSet(epochs['wake'].loc[[0]])).time_support.start[0], center.t[0]],
                              end = [center.t[0], rot_pos.restrict(nap.IntervalSet(epochs['wake'].loc[[0]])).time_support.end[0]])

    ep2 = halves.intersect(moving_ep)
    
    half1 = ep2.loc[0:len(ep2)/2]
    half2 = ep2.loc[(len(ep2)/2)+1:]
    
    pf1, binsxy = nap.compute_2d_tuning_curves(group = pyr2, features = rot_pos, ep = half1, nb_bins=20)  
    pf2, binsxy = nap.compute_2d_tuning_curves(group = pyr2, features = rot_pos, ep = half2, nb_bins=20)  
    
    for k in pyr2:
        good = np.logical_and(np.isfinite(pf1[k].flatten()), np.isfinite(pf2[k].flatten()))
        corr, p = scipy.stats.pearsonr(pf1[k].flatten()[good], pf2[k].flatten()[good]) 
        
        if isWT == 1:
            placefield_stability_wt.append(corr)
        else: placefield_stability_ko.append(corr)
        
    
#%% Plot stability
    
### All cells
    
    
    # for i,n in enumerate(pyr2):
    #     plt.figure()
    #     good = np.logical_and(np.isfinite(pf1[k].flatten()), np.isfinite(pf2[k].flatten()))
    #     corr, _ = scipy.stats.pearsonr(pf1[n].flatten()[good], pf2[n].flatten()[good]) 
    #     plt.suptitle('R = '  + str(round(corr, 2)))
    #     plt.subplot(121)
    #     plt.imshow(pf1[n], extent=(binsxy[1][0],binsxy[1][-1],binsxy[0][0],binsxy[0][-1]), cmap = 'jet')        
    #     plt.colorbar()
    #     plt.subplot(122)
    #     plt.imshow(pf2[n], extent=(binsxy[1][0],binsxy[1][-1],binsxy[0][0],binsxy[0][-1]), cmap = 'jet')        
    #     plt.colorbar()


### For examples 

    # norm = max(np.nanmax(pf1[pyr2.index[7]]), np.nanmax(pf2[pyr2.index[7]]))
    # plt.figure()
    # plt.subplot(121)
    # plt.imshow(pf1[pyr2.index[7]] / norm, extent=(binsxy[1][0],binsxy[1][-1],binsxy[0][0],binsxy[0][-1]), cmap = 'jet') 
    # plt.subplot(122)
    # plt.imshow(pf2[pyr2.index[7]] / norm, extent=(binsxy[1][0],binsxy[1][-1],binsxy[0][0],binsxy[0][-1]), cmap = 'jet')        
    
    
#%% Plot tracking 

    # if name != 'B2618':
        plt.figure()
        plt.subplot(121)
        plt.plot(rot_pos['x'], rot_pos['z'], color = 'grey')
        spk_pos1 = pyr2[pyr2.index[10]].value_from(rot_pos)
        plt.plot(spk_pos1['x'], spk_pos1['z'], 'o', color = 'r', markersize = 5, alpha = 0.5)
        plt.gca().set_box_aspect(1)
        plt.subplot(122)
        plt.title('SI = '  + str(round(spatialinfo['SI'].tolist()[10], 2)))
        plt.imshow(placefields[pyr2.index[10]].T / placefields[pyr2.index[10]].max() , origin = 'lower', cmap = 'viridis') 
        plt.colorbar(label = 'Norm. Rate')
        plt.gca().set_box_aspect(1)
                

#%% Plot tuning curves 
    
    # if name != 'B2618':
    # if isWT != 0:
    #     plt.figure()
    #     plt.suptitle(s)
    #     # for n in range(len(spikes)):
    #     for i,n in enumerate(pyr2):
    #         plt.subplot(9,8,n+1)
    #         # plt.title(spikes._metadata['celltype'][n])
    #         plt.title('SI = '  + str(round(spatialinfo['SI'].tolist()[i], 2)))
    #         plt.imshow(placefields[n], extent=(binsxy[1][0],binsxy[1][-1],binsxy[0][0],binsxy[0][-1]), cmap = 'jet')        
    #         plt.colorbar()
        
    # multipage(data_directory + '/' + 'Allcells.pdf', dpi=250)
    
#%% Organize spatial information data 

wt = np.array(['WT' for x in range(len(allspatialinfo_wt))])
ko = np.array(['KO' for x in range(len(allspatialinfo_ko))])

genotype = np.hstack([wt, ko])

sinfos = []
sinfos.extend(allspatialinfo_wt)
sinfos.extend(allspatialinfo_ko)

allinfos = pd.DataFrame(data = [sinfos, genotype], index = ['SI', 'genotype']).T

#%% Plotting SI

plt.figure()
sns.set_style('white')
palette = ['royalblue', 'indianred'] 
ax = sns.violinplot( x = allinfos['genotype'], y=allinfos['SI'].astype(float) , data = allinfos, dodge=False,
                    palette = palette,cut = 2,
                    scale="width", inner=None)
ax.tick_params(bottom=True, left=True)
xlim = ax.get_xlim()
ylim = ax.get_ylim()
for violin in ax.collections:
    x0, y0, width, height = violin.get_paths()[0].get_extents().bounds
    violin.set_clip_path(plt.Rectangle((x0, y0), width / 2, height, transform=ax.transData))
sns.boxplot(x = allinfos['genotype'], y=allinfos['SI'].astype(float) , data = allinfos, saturation=1, showfliers=False,
            width=0.3, boxprops={'zorder': 3, 'facecolor': 'none'}, ax=ax)
old_len_collections = len(ax.collections)
sns.stripplot(x = allinfos['genotype'], y = allinfos['SI'].astype(float), data = allinfos, color = 'k', dodge=False, ax=ax, alpha = 0.2)
# sns.swarmplot(x = wakedf['type'], y = wakedf['rate'].astype(float), data = wakedf, color = 'k', dodge=False, ax=ax)
for dots in ax.collections[old_len_collections:]:
    dots.set_offsets(dots.get_offsets())
ax.set_xlim(xlim)
ax.set_ylim(ylim)
plt.ylabel('Spatial Information (bits per spike)')
ax.set_box_aspect(1)

#%% Stats

t, p = mannwhitneyu(allspatialinfo_wt, allspatialinfo_ko)
t2, p2 = mannwhitneyu(placefield_stability_wt, placefield_stability_ko)

#%% Organize and plot Placefield Stability data 

wt = np.array(['WT' for x in range(len(placefield_stability_wt))])
ko = np.array(['KO' for x in range(len(placefield_stability_ko))])

genotype = np.hstack([wt, ko])

sinfos = []
sinfos.extend(placefield_stability_wt)
sinfos.extend(placefield_stability_ko)

allinfos = pd.DataFrame(data = [sinfos, genotype], index = ['Corr', 'genotype']).T

plt.figure()
sns.set_style('white')
palette = ['royalblue', 'indianred'] 
ax = sns.violinplot( x = allinfos['genotype'], y=allinfos['Corr'].astype(float) , data = allinfos, dodge=False,
                    palette = palette,cut = 2,
                    scale="width", inner=None)
ax.tick_params(bottom=True, left=True)
xlim = ax.get_xlim()
ylim = ax.get_ylim()
for violin in ax.collections:
    x0, y0, width, height = violin.get_paths()[0].get_extents().bounds
    violin.set_clip_path(plt.Rectangle((x0, y0), width / 2, height, transform=ax.transData))
sns.boxplot(x = allinfos['genotype'], y=allinfos['Corr'].astype(float) , data = allinfos, saturation=1, showfliers=False,
            width=0.3, boxprops={'zorder': 3, 'facecolor': 'none'}, ax=ax)
old_len_collections = len(ax.collections)
sns.stripplot(x = allinfos['genotype'], y = allinfos['Corr'].astype(float), data = allinfos, color = 'k', dodge=False, ax=ax, alpha = 0.2)
# sns.swarmplot(x = wakedf['type'], y = wakedf['rate'].astype(float), data = wakedf, color = 'k', dodge=False, ax=ax)
for dots in ax.collections[old_len_collections:]:
    dots.set_offsets(dots.get_offsets())
ax.set_xlim(xlim)
ax.set_ylim(ylim)
plt.ylabel('Half session correlation (R)')
ax.set_box_aspect(1)


