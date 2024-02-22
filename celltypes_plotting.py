#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 16:00:04 2023

@author: dhruv
"""

import numpy as np 
import pandas as pd
import pynapple as nap
import seaborn as sns
import os, sys
import matplotlib.pyplot as plt 
from scipy.stats import mannwhitneyu

#%% 

data_directory = '/media/dhruv/Expansion/Processed'
datasets = np.genfromtxt(os.path.join(data_directory,'dataset_DM.list'), delimiter = '\n', dtype = str, comments = '#')

isWT = []

t2p = []
rates = []
celltype = []
genotype = []

cellratios_wt = []
cellratios_ko = []

for s in datasets[1:]:
    print(s)
    name = s.split('-')[0]
        
    if name == 'B2613' or name == 'B2618':
        isWT = 0
    else: isWT = 1
    
    path = os.path.join(data_directory, s)

#%% Load classified spikes 

    # sp2 = np.load(os.path.join(path, 'spikedata.npz'), allow_pickle = True)
    sp2 = np.load(os.path.join(path, 'spikedata_0.55.npz'), allow_pickle = True)
    time_support = nap.IntervalSet(sp2['start'], sp2['end'])
    tsd = nap.Tsd(t=sp2['t'], d=sp2['index'], time_support = time_support)
    spikes = tsd.to_tsgroup()
    spikes.set_info(group = sp2['group'], location = sp2['location'], celltype = sp2['celltype'], tr2pk = sp2['tr2pk'])
    
    rates.extend(spikes._metadata['rate'].values)
    t2p.extend(spikes._metadata['tr2pk'].values)
    celltype.extend(spikes._metadata['celltype'].values)
    
    if name == 'B2613' or name == 'B2618':
        genotype.extend(np.zeros_like(spikes._metadata['rate'].values, dtype = 'bool'))
    else: genotype.extend(np.ones_like(spikes._metadata['rate'].values, dtype ='bool'))

#%% Ratio of FS versus broad waveform cells
            
    # if 'fs' in spikes.getby_category('celltype').keys():
    #     fscells = spikes.getby_category('celltype')['fs']
            
    #     bwf = 0 
    #     for i in spikes._metadata['tr2pk']: 
    #         if i > 0.38:
    #             bwf += 1
    
    #     fs2bwf = len(fscells) / bwf
        
    #     if isWT == 1: 
    #         cellratios_wt.append(fs2bwf)
        
    #     else: cellratios_ko.append(fs2bwf)

#%% Sorting the ratio of FS to broad waveform cells

# wt = np.array(['WT' for x in range(len(cellratios_wt))])
# ko = np.array(['KO' for x in range(len(cellratios_ko))])    

# genotype = np.hstack([wt, ko])


# sess_ratios = []
# sess_ratios.extend(cellratios_wt)
# sess_ratios.extend(cellratios_ko)

# summ = pd.DataFrame(data = [sess_ratios, genotype], index = ['cellratio', 'genotype']).T
    
#%% Plotting cell ratios by session 

# plt.figure()
# plt.title('Ratio of FS to broad waveform cells')
# sns.set_style('white')
# palette = ['royalblue', 'indianred']
# ax = sns.violinplot( x = summ['genotype'], y = summ['cellratio'].astype(float) , data = summ, dodge = False,
#                     palette = palette,cut = 2,
#                     scale = "width", inner = None)
# ax.tick_params(bottom=True, left=True)
# xlim = ax.get_xlim()
# ylim = ax.get_ylim()
# for violin in ax.collections:
#     x0, y0, width, height = violin.get_paths()[0].get_extents().bounds
#     violin.set_clip_path(plt.Rectangle((x0, y0), width / 2, height, transform=ax.transData))
# sns.boxplot(x = summ['genotype'], y = summ['cellratio'] , data = summ, saturation = 1, showfliers = False,
#             width = 0.3, boxprops = {'zorder': 3, 'facecolor': 'none'}, ax = ax)
# old_len_collections = len(ax.collections)
# sns.swarmplot(x = summ['genotype'], y=summ['cellratio'], data = summ, color = 'k', dodge = False, ax = ax)
# # sns.stripplot(x = wakedf['type'], y = wakedf['rate'].astype(float), data = wakedf, color = 'k', dodge=False, ax=ax)
# for dots in ax.collections[old_len_collections:]:
#     dots.set_offsets(dots.get_offsets())
# ax.set_xlim(xlim)
# ax.set_ylim(ylim)
# plt.ylabel('Ratio of FS to broad waveform cells')
# ax.set_box_aspect(1)

# t, p = mannwhitneyu(cellratios_wt, cellratios_ko)

#%% Sorting by genotype and plotting

t2p_wt = np.array(t2p)[genotype]
rates_wt = np.array(rates)[genotype]

pyr = [i for i, x in enumerate(np.array(celltype)[genotype]) if x == 'pyr']
fs = [i for i, x in enumerate(np.array(celltype)[genotype]) if x == 'fs']
oth = [i for i, x in enumerate(np.array(celltype)[genotype]) if x == 'other']

plt.figure()
# plt.suptitle('Cell Type Classification')

# plt.subplot(121)
# plt.title('WT')
plt.scatter([t2p_wt[i] for i in pyr], [rates_wt[i] for i in pyr] , marker = 'o', color = 'royalblue', label = 'WT EX', zorder = 3)        
plt.scatter([t2p_wt[i] for i in fs], [rates_wt[i] for i in fs] , marker = 'o', color = 'indianred', label = 'WT FS', zorder = 3)        
plt.scatter([t2p_wt[i] for i in oth], [rates_wt[i] for i in oth] , marker = 'o',  color = 'silver', label = 'WT Unclassified', zorder = 3)       

# plt.xlim([0, 1.5])
# plt.ylim([0, 50])
# plt.axhline(10, linestyle = '--', color = 'k')
# plt.axvline(0.38, linestyle = '--', color = 'k')
# plt.axvline(0.55, linestyle = '--', color = 'k')
# plt.xlabel('Trough to peak (ms)')
# plt.ylabel('Firing rate (Hz)')   
# plt.legend(loc = 'upper right') 

t2p_ko = np.array(t2p)[np.invert(genotype)]
rates_ko = np.array(rates)[np.invert(genotype)]

pyr = [i for i, x in enumerate(np.array(celltype)[np.invert(genotype)]) if x == 'pyr']
fs = [i for i, x in enumerate(np.array(celltype)[np.invert(genotype)]) if x == 'fs']
oth = [i for i, x in enumerate(np.array(celltype)[np.invert(genotype)]) if x == 'other']

# plt.subplot(122)
# plt.title('KO')
plt.scatter([t2p_ko[i] for i in pyr], [rates_ko[i] for i in pyr] , marker = 'x', color = 'lightsteelblue', label = 'KO EX')        
plt.scatter([t2p_ko[i] for i in fs], [rates_ko[i] for i in fs] ,  marker = 'x', color = 'rosybrown', label = 'KO FS')        
plt.scatter([t2p_ko[i] for i in oth], [rates_ko[i] for i in oth] , marker =  'x',  color = 'silver', label = 'KO Unclassified')        

plt.xlim([0, 1.5])
plt.ylim([0, 50])
plt.axhline(10, linestyle = '--', color = 'k')
# plt.axvline(0.38, linestyle = '--', color = 'k')
plt.axvline(0.55, linestyle = '--', color = 'k')
plt.xlabel('Trough to peak (ms)')
plt.ylabel('Firing rate (Hz)')   
plt.legend(loc = 'upper right')

