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
from scipy.stats import mannwhitneyu
from matplotlib.backends.backend_pdf import PdfPages    

#%% 

def multipage(filename, figs=None, dpi=200):
    pp = PdfPages(filename)
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()
    
    
#%% 

data_directory = '/media/dhruv/Expansion/Processed'
datasets = np.genfromtxt(os.path.join(data_directory,'dataset_DM.list'), delimiter = '\n', dtype = str, comments = '#')

allspatialinfo_wt = []
allspatialinfo_ko = []

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
                                     
    placefields, binsxy = nap.compute_2d_tuning_curves(group = spikes, 
                                                       feature = position[['x', 'z']], 
                                                       ep = epochs['wake'].loc[0] , 
                                                       nb_bins=20)                                               
    
    spatialinfo = nap.compute_2d_mutual_info(placefields, position[['x', 'z']], ep = epochs['wake'])
    
    if isWT == 1:
        allspatialinfo_wt.extend(spatialinfo['SI'].tolist())
    else: allspatialinfo_ko.extend(spatialinfo['SI'].tolist())

    for i in spikes.keys(): 
        placefields[i][np.isnan(placefields[i])] = 0
        placefields[i] = scipy.ndimage.gaussian_filter(placefields[i], 1)

#%% Plot tracking 

    # if name != 'B2618':
        # plt.figure()
        # plt.title(s)
        # plt.plot(position['x'], position['z'])

#%% Plot tuning curves 
    
    # if name != 'B2618':
    #     plt.figure()
    #     plt.suptitle(s)
    #     for n in range(len(spikes)):
    #         plt.subplot(9,8,n+1)
    #         plt.title(spikes._metadata['celltype'][n])
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

#%% Plotting 

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