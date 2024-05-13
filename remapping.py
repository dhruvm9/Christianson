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
from scipy.stats import mannwhitneyu, wilcoxon
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

def occupancy_prob(position, ep, nb_bins=24, norm = False):
    pos= position[['x','z']]
    position_tsd = pos.restrict(ep)
    xpos = position_tsd[:,0]
    ypos = position_tsd[:,1]
    xbins = np.linspace(xpos.min(), xpos.max()+1e-6, nb_bins+1)
    ybins = np.linspace(ypos.min(), ypos.max()+1e-6, nb_bins+1) 
    occupancy, _, _ = np.histogram2d(ypos, xpos, [ybins,xbins])
    
    if norm is True:
        occupancy = occupancy/sum(occupancy)
        
    masked_array = np.ma.masked_where(occupancy == 0, occupancy) 
    masked_array = np.flipud(masked_array)
    return masked_array

def sparsity(rate_map, px):
    '''
    Compute sparsity of a rate map, The sparsity  measure is an adaptation
    to space. The adaptation measures the fraction of the environment  in which
    a cell is  active. A sparsity of, 0.1 means that the place field of the
    cell occupies 1/10 of the area the subject traverses [2]_

    Parameters
    ----------
    rate_map : normalized numpy.ndarray
        A firing rate map, any number of dimensions.

    Returns
    -------
    out : float
        sparsity

    References
    ----------
    .. [2] Skaggs, W. E., McNaughton, B. L., Wilson, M., & Barnes, C. (1996).
       Theta phase precession in hippocampal neuronal populations and the
       compression of temporal sequences. Hippocampus, 6, 149-172.
    '''
    tmp_rate_map = rate_map.copy()
    tmp_rate_map[np.isnan(tmp_rate_map)] = 0
    avg_rate = np.sum(np.ravel(tmp_rate_map * px))
    avg_sqr_rate = np.sum(np.ravel(tmp_rate_map**2 * px))
    return avg_rate**2 / avg_sqr_rate

    
#%% 

data_directory = '/media/dhruv/Expansion/Processed'
# data_directory = '/media/adrien/Expansion/Processed'
datasets = np.genfromtxt(os.path.join(data_directory,'remapping_DM.list'), delimiter = '\n', dtype = str, comments = '#')

env_stability_wt = []
env_stability_ko = []

halfsession1_corr_wt = []
halfsession1_corr_ko = []

halfsession2_corr_wt = []
halfsession2_corr_ko = []

SI1_wt = []
SI1_ko = []

SI2_wt = []
SI2_ko = []

SI3_wt = []
SI3_ko = []

SI4_wt = []
SI4_ko = []

allspatialinfo_env1_wt = []
allspatialinfo_env2_wt = []

allspatialinfo_env1_ko = []
allspatialinfo_env2_ko = []

for s in datasets:
    print(s)
    name = s.split('-')[0]
       
    path = os.path.join(data_directory, s)
    
    if name == 'B2618':
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
    
    if isWT == 1:
        rad = 0.6
    else: rad = 1
    
    for i in range(len(xypos)):
        newx, newy = rotate_via_numpy(xypos[i], rad)
        rot_pos.append((newx, newy))
        
    rot_pos = nap.TsdFrame(t = position.index.values, d = rot_pos, columns = ['x', 'z'])
    
#%% Get cells with wake rate more then 0.5Hz
        
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
        speed = nap.Tsd(t = tmp.index.values[0:-1]+ speedbinsize/2, d = distance/speedbinsize) # in cm/s
     
        moving_ep = nap.IntervalSet(speed.threshold(2).time_support) #Epochs in which speed is > 2 cm/s
        ep1 = moving_ep.intersect(epochs['wake'].loc[[0]])
        ep2 = moving_ep.intersect(epochs['wake'].loc[[1]])
           
#%% Compute place fields in the 2 arenas
    
    w1 = nap.IntervalSet(start = epochs['wake'][0]['start'], end = epochs['wake'][0]['end'])
    w2 = nap.IntervalSet(start = epochs['wake'][1]['start'], end = epochs['wake'][1]['end'])
                                     
    placefields1, binsxy1 = nap.compute_2d_tuning_curves(group = pyr2, 
                                                       features = rot_pos[['x', 'z']], 
                                                       ep = ep1, 
                                                       nb_bins=24)      
    px1 = occupancy_prob(rot_pos, ep1, nb_bins=24)
    
        
    
    
    placefields2, binsxy2 = nap.compute_2d_tuning_curves(group = pyr2, 
                                                       features = rot_pos[['x', 'z']], 
                                                       ep = ep2, 
                                                       nb_bins=24)
    px2 = occupancy_prob(rot_pos, ep2, nb_bins=24)
    
    SI_1 = nap.compute_2d_mutual_info(placefields1, rot_pos[['x', 'z']], ep = ep1)
    SI_2 = nap.compute_2d_mutual_info(placefields2, rot_pos[['x', 'z']], ep = ep2)

    spatialinfo_env1 = nap.compute_2d_mutual_info(placefields1, rot_pos[['x', 'z']], ep = ep1)
    spatialinfo_env2 = nap.compute_2d_mutual_info(placefields2, rot_pos[['x', 'z']], ep = ep2)

    if isWT == 1:
        allspatialinfo_env1_wt.extend(spatialinfo_env1['SI'].tolist())
        allspatialinfo_env2_wt.extend(spatialinfo_env2['SI'].tolist())
    else: 
        allspatialinfo_env1_ko.extend(spatialinfo_env1['SI'].tolist())
        allspatialinfo_env2_ko.extend(spatialinfo_env2['SI'].tolist())
    
    for i in pyr2.keys(): 
        placefields1[i][np.isnan(placefields1[i])] = 0
        placefields1[i] = scipy.ndimage.gaussian_filter(placefields1[i], 1.5, mode = 'nearest')
        masked_array = np.ma.masked_where(px1 == 0, placefields1[i]) #should work fine without it 
        placefields1[i] = masked_array
        
        placefields2[i][np.isnan(placefields2[i])] = 0
        placefields2[i] = scipy.ndimage.gaussian_filter(placefields2[i], 1.5, mode = 'nearest')
        masked_array = np.ma.masked_where(px2 == 0, placefields2[i]) #should work fine without it 
        placefields2[i] = masked_array
        
    

#%% Plot tracking 

    # plt.figure()
    # plt.suptitle(s)
    # plt.subplot(121)
    # plt.plot(rot_pos['x'].restrict(w1), rot_pos['z'].restrict(w1))
    # plt.subplot(122)
    # plt.plot(rot_pos['x'].restrict(w2), rot_pos['z'].restrict(w2))
    
#%% Plot remapping 
    
    # ref = pyr2.keys()
    # nrows = int(np.sqrt(len(ref)))
    # ncols = int(len(ref)/nrows)+1

    # plt.figure()
    # plt.suptitle(s + ' Wake1')
    # for i,n in enumerate(pyr2):
    #     plt.subplot(nrows, ncols, i+1)
    #     # plt.title(spikes._metadata['celltype'][i])
    #     plt.imshow(placefields1[n], extent=(binsxy1[1][0],binsxy1[1][-1],binsxy1[0][0],binsxy1[0][-1]), cmap = 'jet')        
    #     plt.colorbar()

    # plt.figure()
    # plt.suptitle(s + ' Wake2')
    # for i,n in enumerate(pyr2):
    #     plt.subplot(nrows, ncols, i+1)
    #     # plt.title(spikes._metadata['celltype'][i])
    #     plt.imshow(placefields2[n], extent=(binsxy2[1][0],binsxy2[1][-1],binsxy2[0][0],binsxy2[0][-1]), cmap = 'jet')        
    #     plt.colorbar()
    
    
###EXAMPLES 
    # for i,n in enumerate(pyr2):
    #     plt.figure()
    #     good = np.logical_and(np.isfinite(placefields1[n].flatten()), np.isfinite(placefields2[n].flatten()))
    #     corr, _ = scipy.stats.pearsonr(placefields1[n].flatten()[good], placefields2[n].flatten()[good]) 
    #     plt.suptitle('R = '  + str(round(corr, 2)))
    #     plt.subplot(121)
    #     plt.imshow(placefields1[n], extent=(binsxy1[1][0],binsxy1[1][-1],binsxy1[0][0],binsxy1[0][-1]), cmap = 'jet')        
    #     plt.colorbar()
    #     plt.subplot(122)
    #     plt.imshow(placefields2[n], extent=(binsxy2[1][0],binsxy2[1][-1],binsxy2[0][0],binsxy2[0][-1]), cmap = 'jet')        
    #     plt.colorbar()

    
#%% Split both wake epochs into halves 

    center1 = rot_pos.restrict(nap.IntervalSet(epochs['wake'][0])).time_support.get_intervals_center()
    center2 = rot_pos.restrict(nap.IntervalSet(epochs['wake'][1])).time_support.get_intervals_center()
    
    halves1 = nap.IntervalSet(start = [rot_pos.restrict(nap.IntervalSet(epochs['wake'].loc[[0]])).time_support.start[0], center1.t[0]],
                              end = [center1.t[0], rot_pos.restrict(nap.IntervalSet(epochs['wake'].loc[[0]])).time_support.end[0]])

    halves2 = nap.IntervalSet(start = [rot_pos.restrict(nap.IntervalSet(epochs['wake'].loc[[1]])).time_support.start[0], center2.t[0]],
                              end = [center2.t[0], rot_pos.restrict(nap.IntervalSet(epochs['wake'].loc[[1]])).time_support.end[0]])

    
    ep_wake1 = halves1.intersect(moving_ep)
    ep_wake2 = halves2.intersect(moving_ep)
        
    half1_wake1 = ep_wake1[0:len(ep_wake1)//2]
    half2_wake1 = ep_wake1[(len(ep_wake1)//2)+1:]
    
    half1_wake2 = ep_wake2[0:len(ep_wake2)//2]
    half2_wake2 = ep_wake2[(len(ep_wake2)//2)+1:]
        
    pf1, binsxy = nap.compute_2d_tuning_curves(group = pyr2, features = rot_pos[['x', 'z']], ep = half1_wake1, nb_bins=24)  
    px1 = occupancy_prob(rot_pos, half1_wake1, nb_bins=24)
    spatialinfo1 = nap.compute_2d_mutual_info(pf1, rot_pos[['x', 'z']], ep = half1_wake1)
    
    norm_px1 = occupancy_prob(rot_pos, half1_wake1, nb_bins=24, norm=True)
    norm_px2 = occupancy_prob(rot_pos, half1_wake2, nb_bins=24, norm=True)
    norm_px3 = occupancy_prob(rot_pos, half2_wake1, nb_bins=24, norm=True)
    norm_px4 = occupancy_prob(rot_pos, half2_wake2, nb_bins=24, norm=True)
    
    sp1 = []
    for k in pyr2:
        tmp = sparsity(pf1[k], px1)
        sp1.append(tmp)
   
    
    pf2, binsxy = nap.compute_2d_tuning_curves(group = pyr2, features = rot_pos[['x', 'z']], ep = half2_wake1, nb_bins=24)  
    px2 = occupancy_prob(rot_pos, half2_wake1, nb_bins=24)
    spatialinfo2 = nap.compute_2d_mutual_info(pf2, rot_pos[['x', 'z']], ep = half2_wake1)

    pf3, binsxy = nap.compute_2d_tuning_curves(group = pyr2, features = rot_pos[['x', 'z']], ep = half1_wake2, nb_bins=24)  
    px3 = occupancy_prob(rot_pos, half1_wake2, nb_bins=24)
    spatialinfo3 = nap.compute_2d_mutual_info(pf3, rot_pos[['x', 'z']], ep = half1_wake2)
        
    pf4, binsxy = nap.compute_2d_tuning_curves(group = pyr2, features = rot_pos[['x', 'z']], ep = half2_wake2, nb_bins=24)  
    px4 = occupancy_prob(rot_pos, half2_wake2, nb_bins=24)
    spatialinfo4 = nap.compute_2d_mutual_info(pf4, rot_pos[['x', 'z']], ep = half2_wake2)
    
    for i in pyr2.keys(): 
        pf1[i][np.isnan(pf1[i])] = 0
        pf1[i] = scipy.ndimage.gaussian_filter(pf1[i], 1.5, mode = 'nearest')
        masked_array = np.ma.masked_where(px1 == 0, pf1[i]) #should work fine without it 
        pf1[i] = masked_array
        
        pf2[i][np.isnan(pf2[i])] = 0
        pf2[i] = scipy.ndimage.gaussian_filter(pf2[i], 1.5, mode = 'nearest')
        masked_array = np.ma.masked_where(px2 == 0, pf2[i]) #should work fine without it 
        pf2[i] = masked_array
        
        pf3[i][np.isnan(pf3[i])] = 0
        pf3[i] = scipy.ndimage.gaussian_filter(pf3[i], 1.5, mode = 'nearest')
        masked_array = np.ma.masked_where(px3 == 0, pf3[i]) #should work fine without it 
        pf3[i] = masked_array
        
        pf4[i][np.isnan(pf4[i])] = 0
        pf4[i] = scipy.ndimage.gaussian_filter(pf4[i], 1.5, mode = 'nearest')
        masked_array = np.ma.masked_where(px4 == 0, pf4[i]) #should work fine without it 
        pf4[i] = masked_array
                
    
    if isWT == 1:
        SI1_wt.extend(spatialinfo1.values)
        SI2_wt.extend(spatialinfo2.values)
        SI3_wt.extend(spatialinfo3.values)
        SI4_wt.extend(spatialinfo4.values)
    else: 
        SI1_ko.extend(spatialinfo1.values)
        SI2_ko.extend(spatialinfo2.values)
        SI3_ko.extend(spatialinfo3.values)
        SI4_ko.extend(spatialinfo4.values)
        
        
   
        
#%% Quantify spatial maps between 2 environments 

    for k in pyr2:
        
        ###Between 2 environments
        good = np.logical_and(np.isfinite(placefields1[k].flatten()), np.isfinite(placefields2[k].flatten()))
        corr, p = scipy.stats.pearsonr(placefields1[k].flatten()[good], placefields2[k].flatten()[good]) 
        
        if isWT == 1:
            env_stability_wt.append(corr)
        else: 
            env_stability_ko.append(corr)
        
        ###Between 2 halves of first wake 
        good2 = np.logical_and(np.isfinite(pf1[k].flatten()), np.isfinite(pf2[k].flatten()))
        corr2, p2 = scipy.stats.pearsonr(pf1[k].flatten()[good2], pf2[k].flatten()[good2]) 
        
        if isWT == 1:
            halfsession1_corr_wt.append(corr2)
        else:
            halfsession1_corr_ko.append(corr2)
        
        ###Between 2 halves of second wake 
        good3 = np.logical_and(np.isfinite(pf3[k].flatten()), np.isfinite(pf4[k].flatten()))
        corr3, p3 = scipy.stats.pearsonr(pf3[k].flatten()[good3], pf4[k].flatten()[good3]) 
        
        if isWT == 1:
            halfsession2_corr_wt.append(corr3)
        else: 
            halfsession2_corr_ko.append(corr3)
        
### PLOT EXAMPLES ARENA 1 

    # for i,n in enumerate(pyr2):
    #     plt.figure()
    #     good = np.logical_and(np.isfinite(pf1[n].flatten()), np.isfinite(pf2[n].flatten()))
    #     corr, _ = scipy.stats.pearsonr(pf1[n].flatten()[good], pf2[n].flatten()[good]) 
    #     plt.suptitle('R = '  + str(round(corr, 2)))
    #     plt.subplot(121)
    #     plt.title(round(spatialinfo1['SI'][n],2))
    #     plt.imshow(pf1[n], extent=(binsxy[1][0],binsxy[1][-1],binsxy[0][0],binsxy[0][-1]), cmap = 'jet')        
    #     plt.colorbar()
    #     plt.subplot(122)
    #     plt.title(round(spatialinfo2['SI'][n],2))
    #     plt.imshow(pf2[n], extent=(binsxy[1][0],binsxy[1][-1],binsxy[0][0],binsxy[0][-1]), cmap = 'jet')        
    #     plt.colorbar()
        
        
### PLOT EXAMPLES ARENA 2
    
    # for i,n in enumerate(pyr2):
    #     plt.figure()
    #     good = np.logical_and(np.isfinite(pf3[n].flatten()), np.isfinite(pf4[n].flatten()))
    #     corr, _ = scipy.stats.pearsonr(pf3[n].flatten()[good], pf4[n].flatten()[good]) 
    #     plt.suptitle('R = '  + str(round(corr, 2)))
    #     plt.subplot(121)
    #     plt.title(round(spatialinfo3['SI'][n],2))
    #     plt.imshow(pf3[n], extent=(binsxy[1][0],binsxy[1][-1],binsxy[0][0],binsxy[0][-1]), cmap = 'jet')        
    #     plt.colorbar()
    #     plt.subplot(122)
    #     plt.title(round(spatialinfo4['SI'][n],2))
    #     plt.imshow(pf4[n], extent=(binsxy[1][0],binsxy[1][-1],binsxy[0][0],binsxy[0][-1]), cmap = 'jet')        
    #     plt.colorbar()
        
        
        
#%% Organize and plot environment stability data 
    
###Across 2 envs

wt = np.array(['WT' for x in range(len(env_stability_wt))])
ko = np.array(['KO' for x in range(len(env_stability_ko))])

genotype = np.hstack([wt, ko])

sinfos = []
sinfos.extend(env_stability_wt)
sinfos.extend(env_stability_ko)

allinfos = pd.DataFrame(data = [sinfos, genotype], index = ['corr', 'type']).T

###Half-session corr: 1st env

wt = np.array(['WT' for x in range(len(halfsession1_corr_wt))])
ko = np.array(['KO' for x in range(len(halfsession1_corr_ko))])

genotype2 = np.hstack([wt, ko])

sinfos2 = []
sinfos2.extend(halfsession1_corr_wt)
sinfos2.extend(halfsession1_corr_ko)

allinfos2 = pd.DataFrame(data = [sinfos2, genotype2], index = ['corr', 'type']).T

###Half-session corr: 2nd env 

wt = np.array(['WT' for x in range(len(halfsession2_corr_wt))])
ko = np.array(['KO' for x in range(len(halfsession2_corr_ko))])

genotype3 = np.hstack([wt, ko])

sinfos3 = []
sinfos3.extend(halfsession2_corr_wt)
sinfos3.extend(halfsession2_corr_ko)

allinfos3 = pd.DataFrame(data = [sinfos3, genotype3], index = ['corr', 'type']).T

#%% 

plt.figure()
plt.suptitle('Remapping')

plt.subplot(131)
plt.title('A v/s B')
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
sns.stripplot(x = allinfos['type'], y = allinfos['corr'].astype(float), data = allinfos, color = 'k', dodge=False, ax=ax)
# sns.swarmplot(x = durdf['genotype'], y = durdf['dur'].astype(float), data = durdf, color = 'k', dodge=False, ax=ax)
for dots in ax.collections[old_len_collections:]:
    dots.set_offsets(dots.get_offsets())
ax.set_xlim(xlim)
ax.set_ylim(ylim)
plt.ylabel('Correlation (R)')
plt.axhline(0, linestyle = '--', color = 'silver')
ax.set_box_aspect(1)

plt.subplot(132)
plt.title('A1 v/s A2')
sns.set_style('white')
palette = ['royalblue', 'indianred']
ax = sns.violinplot( x = allinfos2['type'], y=allinfos2['corr'].astype(float) , data = allinfos2, dodge=False,
                    palette = palette,cut = 2,
                    scale="width", inner=None)
ax.tick_params(bottom=True, left=True)
xlim = ax.get_xlim()
ylim = ax.get_ylim()
for violin in ax.collections:
    x0, y0, width, height = violin.get_paths()[0].get_extents().bounds
    violin.set_clip_path(plt.Rectangle((x0, y0), width / 2, height, transform=ax.transData))
sns.boxplot(x = allinfos2['type'], y=allinfos2['corr'].astype(float) , data = allinfos2, saturation=1, showfliers=False,
            width=0.3, boxprops={'zorder': 3, 'facecolor': 'none'}, ax=ax)
old_len_collections = len(ax.collections)
sns.stripplot(x = allinfos2['type'], y = allinfos2['corr'].astype(float), data = allinfos2, color = 'k', dodge=False, ax=ax)
# sns.swarmplot(x = durdf['genotype'], y = durdf['dur'].astype(float), data = durdf, color = 'k', dodge=False, ax=ax)
for dots in ax.collections[old_len_collections:]:
    dots.set_offsets(dots.get_offsets())
ax.set_xlim(xlim)
ax.set_ylim(ylim)
plt.ylabel('Correlation (R)')
ax.set_box_aspect(1)

plt.subplot(133)
plt.title('B1 v/s B2')
sns.set_style('white')
palette = ['royalblue', 'indianred']
ax = sns.violinplot( x = allinfos3['type'], y=allinfos3['corr'].astype(float) , data = allinfos3, dodge=False,
                    palette = palette,cut = 2,
                    scale="width", inner=None)
ax.tick_params(bottom=True, left=True)
xlim = ax.get_xlim()
ylim = ax.get_ylim()
for violin in ax.collections:
    x0, y0, width, height = violin.get_paths()[0].get_extents().bounds
    violin.set_clip_path(plt.Rectangle((x0, y0), width / 2, height, transform=ax.transData))
sns.boxplot(x = allinfos3['type'], y=allinfos3['corr'].astype(float) , data = allinfos3, saturation=1, showfliers=False,
            width=0.3, boxprops={'zorder': 3, 'facecolor': 'none'}, ax=ax)
old_len_collections = len(ax.collections)
sns.stripplot(x = allinfos3['type'], y = allinfos3['corr'].astype(float), data = allinfos3, color = 'k', dodge=False, ax=ax)
# sns.swarmplot(x = durdf['genotype'], y = durdf['dur'].astype(float), data = durdf, color = 'k', dodge=False, ax=ax)
for dots in ax.collections[old_len_collections:]:
    dots.set_offsets(dots.get_offsets())
ax.set_xlim(xlim)
ax.set_ylim(ylim)
plt.ylabel('Correlation (R)')
ax.set_box_aspect(1)


#%% Stats for remapping 

t, p = mannwhitneyu(env_stability_wt, env_stability_ko)    

z_wt, p_wt = wilcoxon(np.array(env_stability_wt)-0)
z_ko, p_ko = wilcoxon(np.array(env_stability_ko)-0)

t2, p2 = mannwhitneyu(halfsession1_corr_wt, halfsession1_corr_ko)
t3, p3 = mannwhitneyu(halfsession2_corr_wt, halfsession2_corr_ko)
        
#%% Plot Example cells 

# examples = [4,5,8,10]

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
# plt.plot(rot_pos['x'].restrict(ep1), rot_pos['z'].restrict(ep1), color = 'grey')
# spk_pos1 = pyr2[examples[1]].value_from(rot_pos.restrict(ep1))
# plt.plot(spk_pos1['x'], spk_pos1['z'], 'o', color = 'r', markersize = 5, alpha = 0.5)
# plt.gca().set_box_aspect(1)
# plt.subplot(122)
# plt.plot(rot_pos['x'].restrict(ep2), rot_pos['z'].restrict(ep2), color = 'grey')
# spk_pos2 = pyr2[examples[1]].value_from(rot_pos.restrict(ep2))
# plt.plot(spk_pos2['x'], spk_pos2['z'], 'o', color = 'r', markersize = 5, alpha = 0.5)
# plt.gca().set_box_aspect(1)    
    

# plt.figure()
# for i,n in enumerate(spikes):
#     plt.subplot(4,5,i+1)
#     plt.plot(rot_pos['x'].restrict(ep2), rot_pos['z'].restrict(ep2), color = 'grey')    
#     spk_pos1 = spikes[i].value_from(rot_pos.restrict(ep2))    
#     plt.plot(spk_pos1['x'], spk_pos1['z'], 'o', color = 'r', markersize = 0.32, alpha = 0.5)
#     plt.gca().set_box_aspect(1)
   

#%% Organize spatial information data 

wt1 = np.array(['WT' for x in range(len(allspatialinfo_env1_wt))])
ko1 = np.array(['KO' for x in range(len(allspatialinfo_env1_ko))])

genotype = np.hstack([wt1, ko1])

sinfos1 = []
sinfos1.extend(allspatialinfo_env1_wt)
sinfos1.extend(allspatialinfo_env1_ko)

allinfos1 = pd.DataFrame(data = [sinfos1, genotype], index = ['SI', 'genotype']).T

wt2 = np.array(['WT' for x in range(len(allspatialinfo_env2_wt))])
ko2 = np.array(['KO' for x in range(len(allspatialinfo_env2_ko))])

genotype = np.hstack([wt2, ko2])

sinfos2 = []
sinfos2.extend(allspatialinfo_env2_wt)
sinfos2.extend(allspatialinfo_env2_ko)

allinfos2 = pd.DataFrame(data = [sinfos2, genotype], index = ['SI', 'genotype']).T


#%% Plotting SI

plt.figure()

plt.subplot(121)
plt.title('Square arena')
sns.set_style('white')
palette = ['royalblue', 'indianred'] 
ax = sns.violinplot( x = allinfos1['genotype'], y=allinfos1['SI'].astype(float) , data = allinfos1, dodge=False,
                    palette = palette,cut = 2,
                    scale="width", inner=None)
ax.tick_params(bottom=True, left=True)
xlim = ax.get_xlim()
ylim = ax.get_ylim()
for violin in ax.collections:
    x0, y0, width, height = violin.get_paths()[0].get_extents().bounds
    violin.set_clip_path(plt.Rectangle((x0, y0), width / 2, height, transform=ax.transData))
sns.boxplot(x = allinfos1['genotype'], y=allinfos1['SI'].astype(float) , data = allinfos1, saturation=1, showfliers=False,
            width=0.3, boxprops={'zorder': 3, 'facecolor': 'none'}, ax=ax)
old_len_collections = len(ax.collections)
sns.stripplot(x = allinfos1['genotype'], y = allinfos1['SI'].astype(float), data = allinfos1, color = 'k', dodge=False, ax=ax, alpha = 0.2)
# sns.swarmplot(x = wakedf['type'], y = wakedf['rate'].astype(float), data = wakedf, color = 'k', dodge=False, ax=ax)
for dots in ax.collections[old_len_collections:]:
    dots.set_offsets(dots.get_offsets())
ax.set_xlim(xlim)
ax.set_ylim(ylim)
plt.ylabel('Spatial Information (bits per spike)')
ax.set_box_aspect(1)

plt.subplot(122)
plt.title('Circular arena')
sns.set_style('white')
palette = ['royalblue', 'indianred'] 
ax = sns.violinplot( x = allinfos2['genotype'], y=allinfos2['SI'].astype(float) , data = allinfos2, dodge=False,
                    palette = palette,cut = 2,
                    scale="width", inner=None)
ax.tick_params(bottom=True, left=True)
xlim = ax.get_xlim()
ylim = ax.get_ylim()
for violin in ax.collections:
    x0, y0, width, height = violin.get_paths()[0].get_extents().bounds
    violin.set_clip_path(plt.Rectangle((x0, y0), width / 2, height, transform=ax.transData))
sns.boxplot(x = allinfos2['genotype'], y=allinfos2['SI'].astype(float) , data = allinfos2, saturation=1, showfliers=False,
            width=0.3, boxprops={'zorder': 3, 'facecolor': 'none'}, ax=ax)
old_len_collections = len(ax.collections)
sns.stripplot(x = allinfos2['genotype'], y = allinfos2['SI'].astype(float), data = allinfos2, color = 'k', dodge=False, ax=ax, alpha = 0.2)
# sns.swarmplot(x = wakedf['type'], y = wakedf['rate'].astype(float), data = wakedf, color = 'k', dodge=False, ax=ax)
for dots in ax.collections[old_len_collections:]:
    dots.set_offsets(dots.get_offsets())
ax.set_xlim(xlim)
ax.set_ylim(ylim)
plt.ylabel('Spatial Information (bits per spike)')
ax.set_box_aspect(1)


#%% Stats

t_env1, p_env1 = mannwhitneyu(allspatialinfo_env1_wt, allspatialinfo_env1_ko)
t_env2, p_env2 = mannwhitneyu(allspatialinfo_env2_wt, allspatialinfo_env2_ko)    