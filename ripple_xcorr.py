#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 15:01:09 2024

@author: dhruv
"""

import numpy as np 
import pandas as pd
import nwbmatic as ntm
import pynapple as nap 
import pickle
import scipy.io
import os, sys
import seaborn as sns
import matplotlib.pyplot as plt 
import warnings
from scipy.stats import mannwhitneyu

#%% 

warnings.filterwarnings("ignore")

data_directory = '/media/dhruv/Expansion/Processed'
datasets = np.genfromtxt(os.path.join(data_directory,'dataset_DM.list'), delimiter = '\n', dtype = str, comments = '#')

all_xc_pyr_wt = pd.DataFrame()
all_xc_fs_wt = pd.DataFrame()

all_xc_pyr_ko = pd.DataFrame()
all_xc_fs_ko = pd.DataFrame()

min_pyr_wt = []
max_pyr_wt = []
dur_pyr_wt = [] 
min_pyr_ko = []
max_pyr_ko = []
dur_pyr_ko = [] 

min_fs_wt = []
max_fs_wt = []
dur_fs_wt = [] 
min_fs_ko = []
max_fs_ko = []
dur_fs_ko = [] 

lor_pyr_wt = []
lor_pyr_ko = []
lor_fs_wt = []
lor_fs_ko = []

for s in datasets:
    print(s)
    name = s.split('-')[0]
       
    path = os.path.join(data_directory, s)
    
    if name == 'B2613' or name == 'B2618' or name == 'B2627' or name == 'B2628' or name == 'B3805' or name == 'B3813':
        isWT = 0
    else: isWT = 1 

    # sp2 = np.load(os.path.join(path, 'spikedata.npz'), allow_pickle = True)
    sp2 = np.load(os.path.join(path, 'spikedata_0.55.npz'), allow_pickle = True)
    time_support = nap.IntervalSet(sp2['start'], sp2['end'])
    tsd = nap.Tsd(t=sp2['t'], d=sp2['index'], time_support = time_support)
    spikes = tsd.to_tsgroup()
    spikes.set_info(group = sp2['group'], location = sp2['location'], celltype = sp2['celltype'], tr2pk = sp2['tr2pk'])
    
    data = ntm.load_session(path, 'neurosuite')
    epochs = data.epochs
    position = data.position
    
    file = os.path.join(path, s +'.sws.evt')
    sws_ep = nap.IntervalSet(data.read_neuroscope_intervals(name = 'SWS', path2file = file))
    
    file = os.path.join(path, s +'.evt.py.rip')
    rip_ep = data.read_neuroscope_intervals(name = 'rip', path2file = file)
    
    with open(os.path.join(path, 'riptsd.pickle'), 'rb') as pickle_file:
        rip_tsd = pickle.load(pickle_file)
        
    # with open(os.path.join(path, 'riptrough.pickle'), 'rb') as pickle_file:
    #     rip_trough = pickle.load(pickle_file)
          
#%% Ripple cross corrs

    spikes_by_celltype = spikes.getby_category('celltype')
    rip = nap.Ts(rip_tsd.index.values)
    
    if 'pyr' in spikes._metadata['celltype'].values:
        pyr = spikes_by_celltype['pyr']
        
    keep = []    
    for i in pyr.index:
        if pyr.restrict(nap.IntervalSet(epochs['wake'][0]))._metadata['rate'][i] > 0.5:
            keep.append(i)

    pyr2 = pyr[keep]
        
    if len(pyr2) > 0:
        
        xc_pyr = nap.compute_eventcorrelogram(pyr2, rip, binsize = 0.005, windowsize = 0.2 , ep = nap.IntervalSet(sws_ep), norm = True)
        # xc_pyr = nap.compute_eventcorrelogram(pyr, nap.Ts(np.array(rip_trough)), binsize = 0.0005, windowsize = 0.1 , ep = nap.IntervalSet(sws_ep), norm = True)
        
        minp_pyr = xc_pyr.mean(axis=1)[0:0.105].idxmin()
        minr_pyr = xc_pyr.mean(axis=1)[0:0.105].min()
        maxp_pyr = xc_pyr.mean(axis=1)[minp_pyr:].idxmax()
        durp_pyr = maxp_pyr - minp_pyr
        
        # plt.figure()
        # plt.title(s + '_PYR')
        # plt.plot(xc_pyr.mean(axis=1), color = 'k')
        # plt.axvline(minp_pyr)
        # plt.axvline(maxp_pyr)
    
        if isWT == 1:
            all_xc_pyr_wt = pd.concat([all_xc_pyr_wt, xc_pyr], axis = 1)
            
        else: all_xc_pyr_ko = pd.concat([all_xc_pyr_ko, xc_pyr], axis = 1)
                   
    
    if 'fs' in spikes._metadata['celltype'].values:
        fs = spikes_by_celltype['fs']
        
        xc_fs = nap.compute_eventcorrelogram(fs, rip, binsize = 0.005, windowsize = 0.2 , ep = nap.IntervalSet(sws_ep), norm = True)
        # xc_fs = nap.compute_eventcorrelogram(fs, nap.Ts(np.array(rip_trough)), binsize = 0.0005, windowsize = 0.1 , ep = nap.IntervalSet(sws_ep), norm = True)
        all_xc_pyr_wt.mean(axis=1)
        minp_fs = xc_fs.mean(axis=1)[0:0.105].idxmin()
        minr_fs = xc_fs.mean(axis=1)[0:0.105].min()
        maxp_fs = xc_fs.mean(axis=1)[minp_fs:].idxmax()
        durp_fs = maxp_fs - minp_fs
        
        # plt.figure()
        # plt.title(s + '_FS')
        # plt.plot(xc_fs.mean(axis=1), color = 'k')
        # plt.axvline(minp_fs)
        # plt.axvline(maxp_fs)
        
        if isWT == 1:
            all_xc_fs_wt = pd.concat([all_xc_fs_wt, xc_fs], axis = 1)
            min_pyr_wt.append(minp_pyr)
            max_pyr_wt.append(maxp_pyr)
            dur_pyr_wt.append(durp_pyr)
            min_fs_wt.append(minp_fs)
            max_fs_wt.append(maxp_fs)
            dur_fs_wt.append(durp_fs)
            lor_pyr_wt.append(minr_pyr)
            lor_fs_wt.append(minr_fs)
        else: 
            all_xc_fs_ko = pd.concat([all_xc_fs_ko, xc_fs], axis = 1)
            min_pyr_ko.append(minp_pyr)
            max_pyr_ko.append(maxp_pyr)
            dur_pyr_ko.append(durp_pyr)
            min_fs_ko.append(minp_fs)
            max_fs_ko.append(maxp_fs)
            dur_fs_ko.append(durp_fs)
            lor_pyr_ko.append(minr_pyr)
            lor_fs_ko.append(minr_fs)
           
    else: fs = []
    
    del pyr, fs

                    
        
#%% Plotting 


    # plt.figure()
    # plt.suptitle(s)
    # for n in range(len(spikes)):
    #     plt.subplot(9,8,n+1)
    #     plt.title(spikes._metadata['celltype'][n])
    #     plt.plot(xc[n])

plt.figure()
plt.tight_layout()
plt.suptitle('Ripple onset Cross-correlogram')       

plt.subplot(121)
plt.title('PYR')
plt.xlabel('Time from SWR (s)')
plt.ylabel('norm. rate')
plt.plot(all_xc_pyr_wt.mean(axis=1), color = 'lightsteelblue', label = 'WT')
err = all_xc_pyr_wt.sem(axis=1)
plt.fill_between(all_xc_pyr_wt.index.values, all_xc_pyr_wt.mean(axis=1) - err, all_xc_pyr_wt.mean(axis=1) + err, alpha = 0.2, color = 'lightsteelblue') 

plt.plot(all_xc_pyr_ko.mean(axis=1), color = 'lightcoral', label = 'KO')
err = all_xc_pyr_ko.sem(axis=1)
plt.fill_between(all_xc_pyr_ko.index.values, all_xc_pyr_ko.mean(axis=1) - err, all_xc_pyr_ko.mean(axis=1) + err, alpha = 0.2, color = 'lightcoral') 
plt.legend(loc = 'upper right')
# plt.xticks([-0.1, 0, 0.1])
plt.yticks([1,5])
plt.gca().set_box_aspect(1)

        
plt.subplot(122)
plt.title('FS')
plt.xlabel('Time from SWR (s)')
# plt.ylabel('norm. rate')
plt.plot(all_xc_fs_wt.mean(axis=1), color = 'royalblue', label = 'WT')
err = all_xc_fs_wt.sem(axis=1)
plt.fill_between(all_xc_fs_wt.index.values, all_xc_fs_wt.mean(axis=1) - err, all_xc_fs_wt.mean(axis=1) + err, alpha = 0.2, color = 'royalblue') 

plt.plot(all_xc_fs_ko.mean(axis=1), color = 'indianred', label = 'KO')
err = all_xc_fs_ko.sem(axis=1)
plt.fill_between(all_xc_fs_ko.index.values, all_xc_fs_ko.mean(axis=1) - err, all_xc_fs_ko.mean(axis=1) + err, alpha = 0.2, color = 'indianred') 
plt.legend(loc = 'upper right')
# plt.xticks([-0.1, 0, 0.1])
plt.yticks([1,4.5])
plt.gca().set_box_aspect(1)


#%% Organize min, max, dur 

###Min

wt1 = np.array(['pyr_WT' for x in range(len(min_pyr_wt))])
ko1 = np.array(['pyr_KO' for x in range(len(min_pyr_ko))])
wt2 = np.array(['fs_WT' for x in range(len(min_fs_wt))])
ko2 = np.array(['fs_KO' for x in range(len(min_fs_ko))])

genotype = np.hstack([wt1, ko1, wt2, ko2])

sinfos = []
sinfos.extend(min_pyr_wt)
sinfos.extend(min_pyr_ko)
sinfos.extend(min_fs_wt)
sinfos.extend(min_fs_ko)

allinfos = pd.DataFrame(data = [sinfos, genotype], index = ['corr', 'type']).T

###Max

wt1 = np.array(['pyr_WT' for x in range(len(max_pyr_wt))])
ko1 = np.array(['pyr_KO' for x in range(len(max_pyr_ko))])
wt2 = np.array(['fs_WT' for x in range(len(max_fs_wt))])
ko2 = np.array(['fs_KO' for x in range(len(max_fs_ko))])

genotype = np.hstack([wt1, ko1, wt2, ko2])

sinfos = []
sinfos.extend(max_pyr_wt)
sinfos.extend(max_pyr_ko)
sinfos.extend(max_fs_wt)
sinfos.extend(max_fs_ko)

allinfos2 = pd.DataFrame(data = [sinfos, genotype], index = ['corr', 'type']).T

# ###Dur

wt1 = np.array(['pyr_WT' for x in range(len(dur_pyr_wt))])
ko1 = np.array(['pyr_KO' for x in range(len(dur_pyr_ko))])
wt2 = np.array(['fs_WT' for x in range(len(dur_fs_wt))])
ko2 = np.array(['fs_KO' for x in range(len(dur_fs_ko))])

genotype = np.hstack([wt1, ko1, wt2, ko2])

sinfos = []
sinfos.extend(dur_pyr_wt)
sinfos.extend(dur_pyr_ko)
sinfos.extend(dur_fs_wt)
sinfos.extend(dur_fs_ko)

allinfos3 = pd.DataFrame(data = [sinfos, genotype], index = ['corr', 'type']).T

###Rate at min

wt1 = np.array(['pyr_WT' for x in range(len(lor_pyr_wt))])
ko1 = np.array(['pyr_KO' for x in range(len(lor_pyr_ko))])
wt2 = np.array(['fs_WT' for x in range(len(lor_fs_wt))])
ko2 = np.array(['fs_KO' for x in range(len(lor_fs_ko))])

genotype = np.hstack([wt1, ko1, wt2, ko2])

sinfos = []
sinfos.extend(lor_pyr_wt)
sinfos.extend(lor_pyr_ko)
sinfos.extend(lor_fs_wt)
sinfos.extend(lor_fs_ko)

allinfos4 = pd.DataFrame(data = [sinfos, genotype], index = ['corr', 'type']).T

#%%

tmin_pyr, pmin_pyr = mannwhitneyu(min_pyr_wt, min_pyr_ko)
tmin_fs, pmin_fs = mannwhitneyu(min_fs_wt, min_fs_ko)

tmax_pyr, pmax_pyr = mannwhitneyu(max_pyr_wt, max_pyr_ko)
tmax_fs, pmax_fs = mannwhitneyu(max_fs_wt, max_fs_ko)

td_pyr, pd_pyr = mannwhitneyu(dur_pyr_wt, dur_pyr_ko)
td_fs, pd_fs = mannwhitneyu(dur_fs_wt, dur_fs_ko)

tr_pyr, pr_pyr = mannwhitneyu(lor_pyr_wt, lor_pyr_ko)
tr_fs, pr_fs = mannwhitneyu(lor_fs_wt, lor_fs_ko)


#%% 

plt.figure()
plt.title('Time of min rate')
sns.set_style('white')
palette = ['royalblue', 'lightsteelblue', 'indianred', 'lightcoral']
ax = sns.violinplot( x = allinfos['type'], y=allinfos['corr'].astype(float) , data = allinfos, dodge=False,
                    palette = palette,cut = 2,
                    scale="width", inner=None)
ax.tick_params(bottom=True, left=True)
xlim = ax.get_xlim()
ylim = ax.get_ylim()
for violin in ax.collections:
    x0, y0, width, height = violin.get_paths()[0].get_extents().bounds
    violin.set_clip_path(plt.Rectangle((x0, y0), width / 2, height, transform=ax.transData))
sns.boxplot(x = allinfos['type'], y=allinfos['corr'] , data = allinfos, saturation=1, showfliers=False,
            width=0.3, boxprops={'zorder': 3, 'facecolor': 'none'}, ax=ax)
sns.stripplot(x = allinfos['type'], y = allinfos['corr'].astype(float), data = allinfos, color = 'k', dodge=False, ax=ax, alpha = 0.2)
plt.ylabel('Time (s)')
ax.set_box_aspect(1)

plt.figure()
plt.title('Time of onset')
sns.set_style('white')
palette = ['royalblue', 'lightsteelblue', 'indianred', 'lightcoral']
ax = sns.violinplot( x = allinfos2['type'], y=allinfos2['corr'].astype(float) , data = allinfos2, dodge=False,
                    palette = palette,cut = 2,
                    scale="width", inner=None)
ax.tick_params(bottom=True, left=True)
xlim = ax.get_xlim()
ylim = ax.get_ylim()
for violin in ax.collections:
    x0, y0, width, height = violin.get_paths()[0].get_extents().bounds
    violin.set_clip_path(plt.Rectangle((x0, y0), width / 2, height, transform=ax.transData))
sns.boxplot(x = allinfos2['type'], y=allinfos2['corr'] , data = allinfos2, saturation=1, showfliers=False,
            width=0.3, boxprops={'zorder': 3, 'facecolor': 'none'}, ax=ax)
sns.stripplot(x = allinfos2['type'], y = allinfos2['corr'].astype(float), data = allinfos2, color = 'k', dodge=False, ax=ax, alpha = 0.2)
plt.ylabel('Time (s)')
ax.set_box_aspect(1)

plt.figure()
plt.title('Duration of recovery')
sns.set_style('white')
palette = ['royalblue', 'lightsteelblue', 'indianred', 'lightcoral']
ax = sns.violinplot( x = allinfos3['type'], y=allinfos3['corr'].astype(float) , data = allinfos3, dodge=False,
                    palette = palette,cut = 2,
                    scale="width", inner=None)
ax.tick_params(bottom=True, left=True)
xlim = ax.get_xlim()
ylim = ax.get_ylim()
for violin in ax.collections:
    x0, y0, width, height = violin.get_paths()[0].get_extents().bounds
    violin.set_clip_path(plt.Rectangle((x0, y0), width / 2, height, transform=ax.transData))
sns.boxplot(x = allinfos3['type'], y=allinfos3['corr'] , data = allinfos3, saturation=1, showfliers=False,
            width=0.3, boxprops={'zorder': 3, 'facecolor': 'none'}, ax=ax)
sns.stripplot(x = allinfos3['type'], y = allinfos3['corr'].astype(float), data = allinfos3, color = 'k', dodge=False, ax=ax, alpha = 0.2)
plt.ylabel('Time (s)')
ax.set_box_aspect(1)

plt.figure()
plt.title('Rate at minima')
sns.set_style('white')
palette = ['royalblue', 'lightsteelblue', 'indianred', 'lightcoral']
ax = sns.violinplot( x = allinfos4['type'], y=allinfos4['corr'].astype(float) , data = allinfos3, dodge=False,
                    palette = palette,cut = 2,
                    scale="width", inner=None)
ax.tick_params(bottom=True, left=True)
xlim = ax.get_xlim()
ylim = ax.get_ylim()
for violin in ax.collections:
    x0, y0, width, height = violin.get_paths()[0].get_extents().bounds
    violin.set_clip_path(plt.Rectangle((x0, y0), width / 2, height, transform=ax.transData))
sns.boxplot(x = allinfos4['type'], y=allinfos4['corr'] , data = allinfos4, saturation=1, showfliers=False,
            width=0.3, boxprops={'zorder': 3, 'facecolor': 'none'}, ax=ax)
sns.stripplot(x = allinfos4['type'], y = allinfos4['corr'].astype(float), data = allinfos4, color = 'k', dodge=False, ax=ax, alpha = 0.2)
plt.ylabel('Norm rate')
ax.set_box_aspect(1)
