#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 16:17:35 2023

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
from scipy.stats import mannwhitneyu

#%% 

data_directory = '/media/dhruv/Expansion/Processed'
datasets = np.genfromtxt(os.path.join(data_directory,'dataset_DM.list'), delimiter = '\n', dtype = str, comments = '#')
ripplechannels = np.genfromtxt(os.path.join(data_directory,'ripplechannel.list'), delimiter = '\n', dtype = str, comments = '#')

Fs = 1250

sess_sync_wt = []
sess_sync_ko = []

sess_sync_pyr_wt = []
sess_sync_fs_wt = []

sess_sync_pyr_ko = []
sess_sync_fs_ko = []

sync_all_wt = []
sync_all_ko = []

sync_all_pyr_wt = []
sync_all_fs_wt = []

sync_all_pyr_ko = []
sync_all_fs_ko = []

for r,s in enumerate(datasets):
    print(s)
    name = s.split('-')[0]
    path = os.path.join(data_directory, s)
    
    data = ntm.load_session(path, 'neurosuite')
    data.load_neurosuite_xml(path)
    epochs = data.epochs
    
    if name == 'B2613' or name == 'B2618':
        isWT = 0
    else: isWT = 1 
    
    lfp = nap.load_eeg(path + '/' + s + '.eeg', channel = int(ripplechannels[r]), n_channels = 32, frequency = Fs)
    
    file = os.path.join(path, s +'.sws.evt')
    sws_ep = data.read_neuroscope_intervals(name = 'SWS', path2file = file)
    
    file = os.path.join(path, s +'.rem.evt')
    rem_ep = data.read_neuroscope_intervals(name = 'REM', path2file = file)
    
    file = os.path.join(path, s +'.evt.py.rip')
    rip_ep = data.read_neuroscope_intervals(name = 'rip', path2file = file)
    
    with open(os.path.join(path, 'riptsd.pickle'), 'rb') as pickle_file:
        rip_tsd = pickle.load(pickle_file)
    
#%% Load classified spikes 

    sp2 = np.load(os.path.join(path, 'spikedata.npz'), allow_pickle = True)
    time_support = nap.IntervalSet(sp2['start'], sp2['end'])
    tsd = nap.Tsd(t=sp2['t'], d=sp2['index'], time_support = time_support)
    spikes = tsd.to_tsgroup()
    spikes.set_info(group = sp2['group'], location = sp2['location'], celltype = sp2['celltype'], tr2pk = sp2['tr2pk'])
    
#%% Population synchrony during ripples 

    ripdur = rip_ep['end'] - rip_ep['start']        

    allsync = []
    allsync_pyr = []
    allsync_fs = []
    
    spikes_by_celltype = spikes.getby_category('celltype')
    if 'pyr' in spikes._metadata['celltype'].values:
        pyr = spikes_by_celltype['pyr']
       
    if 'fs' in spikes._metadata['celltype'].values:
        fs = spikes_by_celltype['fs']
           
    if len(fs) > 0 and len(pyr) > 0:

        for i in range(len(rip_ep)):
            rates = spikes.count(ripdur[i], nap.IntervalSet(start = rip_ep['start'][i], end = rip_ep['end'][i])) 
            r_pyr = pyr.count(ripdur[i], nap.IntervalSet(start = rip_ep['start'][i], end = rip_ep['end'][i])) 
            r_fs = fs.count(ripdur[i], nap.IntervalSet(start = rip_ep['start'][i], end = rip_ep['end'][i])) 
                  
            synchrony = rates.d.astype(bool).sum(axis=1)/len(spikes)
            s_pyr = r_pyr.d.astype(bool).sum(axis=1)/len(pyr)
            s_fs = r_fs.d.astype(bool).sum(axis=1)/len(fs)
            
            allsync.append(synchrony)
            allsync_pyr.append(s_pyr)
            allsync_fs.append(s_fs)
    
        allsync = np.concatenate(allsync)
        allsync_pyr = np.concatenate(allsync_pyr)
        allsync_fs = np.concatenate(allsync_fs)

#%% Synchrony per session by genotype    

        if isWT == 1:
            sync_all_wt.extend(allsync)
            sess_sync_wt.append(np.mean(allsync))
            
            sync_all_pyr_wt.extend(allsync_pyr)
            sess_sync_pyr_wt.append(np.mean(allsync_pyr))
            
            sync_all_fs_wt.extend(allsync_fs)
            sess_sync_fs_wt.append(np.mean(allsync_fs))
            
            
        else: 
            sync_all_ko.extend(allsync)
            sess_sync_ko.append(np.mean(allsync))
            
            sync_all_pyr_ko.extend(allsync_pyr)
            sess_sync_pyr_ko.append(np.mean(allsync_pyr))
            
            sync_all_fs_ko.extend(allsync_fs)
            sess_sync_fs_ko.append(np.mean(allsync_fs))
            

#%% Organize data to plot

wt = np.array(['WT' for x in range(len(sess_sync_wt))])
ko = np.array(['KO' for x in range(len(sess_sync_ko))])

wt2 = np.array(['WT' for x in range(len(sync_all_wt))])
ko2 = np.array(['KO' for x in range(len(sync_all_ko))])

genotype = np.hstack([wt, ko])
gt2 = np.hstack([wt2, ko2])

sess_syncs = []
sess_syncs.extend(sess_sync_wt)
sess_syncs.extend(sess_sync_ko)

allsyncs = []
allsyncs.extend(sync_all_wt)
allsyncs.extend(sync_all_ko)

s1 = pd.DataFrame(data = [sess_syncs, genotype], index = ['sync', 'genotype']).T
s2 = pd.DataFrame(data = [allsyncs, gt2], index = ['sync', 'genotype']).T

#%% Organize by celltype 

ex = np.array(['pyr_wt' for x in range(len(sess_sync_pyr_wt))])
ex2 = np.array(['pyr_ko' for x in range(len(sess_sync_pyr_ko))])

pv = np.array(['fs_wt' for x in range(len(sess_sync_fs_wt))])
pv2 = np.array(['fs_ko' for x in range(len(sess_sync_fs_ko))])

ex3 = np.array(['pyr_wt' for x in range(len(sync_all_pyr_wt))])
ex4 = np.array(['pyr_ko' for x in range(len(sync_all_pyr_ko))])

pv3 = np.array(['fs_wt' for x in range(len(sync_all_fs_wt))])
pv4 = np.array(['fs_ko' for x in range(len(sync_all_fs_ko))])

types = np.hstack([ex, ex2, pv, pv2])
types2 = np.hstack([ex3, ex4, pv3, pv4])

sess_syncs_celltype = []
sess_syncs_celltype.extend(sess_sync_pyr_wt)
sess_syncs_celltype.extend(sess_sync_pyr_ko)
sess_syncs_celltype.extend(sess_sync_fs_wt)
sess_syncs_celltype.extend(sess_sync_fs_ko)

allsyncs_celltype = []
allsyncs_celltype.extend(sync_all_pyr_wt)
allsyncs_celltype.extend(sync_all_pyr_ko)
allsyncs_celltype.extend(sync_all_fs_wt)
allsyncs_celltype.extend(sync_all_fs_ko)

s3 = pd.DataFrame(data = [sess_syncs_celltype, types], index = ['sync', 'type']).T
s4 = pd.DataFrame(data = [allsyncs_celltype, types2], index = ['sync', 'type']).T


#%% Plotting for population

# plt.figure()
# plt.title('Population synchrony by session')
# sns.set_style('white')
# palette = ['royalblue', 'indianred']
# ax = sns.violinplot( x = s1['genotype'], y=s1['sync'].astype(float) , data = s1, dodge=False,
#                     palette = palette,cut = 2,
#                     scale="width", inner=None)
# ax.tick_params(bottom=True, left=True)
# xlim = ax.get_xlim()
# ylim = ax.get_ylim()
# for violin in ax.collections:
#     x0, y0, width, height = violin.get_paths()[0].get_extents().bounds
#     violin.set_clip_path(plt.Rectangle((x0, y0), width / 2, height, transform=ax.transData))
# sns.boxplot(x = s1['genotype'], y=s1['sync'] , data = s1, saturation=1, showfliers=False,
#             width=0.3, boxprops={'zorder': 3, 'facecolor': 'none'}, ax=ax)
# old_len_collections = len(ax.collections)
# sns.swarmplot(x = s1['genotype'], y=s1['sync'], data = s1, color = 'k', dodge=False, ax=ax)
# # sns.stripplot(x = wakedf['type'], y = wakedf['rate'].astype(float), data = wakedf, color = 'k', dodge=False, ax=ax)
# for dots in ax.collections[old_len_collections:]:
#     dots.set_offsets(dots.get_offsets())
# ax.set_xlim(xlim)
# ax.set_ylim(ylim)
# plt.ylabel('Population Synchrony during ripple')
# ax.set_box_aspect(1)

# plt.figure()
# plt.title('Population synchrony by event')
# sns.set_style('white')
# palette = ['royalblue', 'indianred']
# ax = sns.violinplot( x = s2['genotype'], y=s2['sync'].astype(float) , data = s2, dodge=False,
#                     palette = palette,cut = 2,
#                     scale="width", inner=None)
# ax.tick_params(bottom=True, left=True)
# xlim = ax.get_xlim()
# ylim = ax.get_ylim()
# for violin in ax.collections:
#     x0, y0, width, height = violin.get_paths()[0].get_extents().bounds
#     violin.set_clip_path(plt.Rectangle((x0, y0), width / 2, height, transform=ax.transData))
# sns.boxplot(x = s2['genotype'], y=s2['sync'] , data = s2, saturation=1, showfliers=False,
#             width=0.3, boxprops={'zorder': 3, 'facecolor': 'none'}, ax=ax)

# plt.ylabel('Population Synchrony during ripple')
# ax.set_box_aspect(1)

#%% Stats 

### For population

# t, p = mannwhitneyu(s1[s1['genotype']=='WT']['sync'].values.astype(float), s1[s1['genotype']=='KO']['sync'].values.astype(float))
# t2, p2 = mannwhitneyu(s2[s2['genotype']=='WT']['sync'].values.astype(float), s2[s2['genotype']=='KO']['sync'].values.astype(float))


### For Celltype (Pyr)
t_pyr_sess, p_pyr_sess = mannwhitneyu(s3[s3['type']=='pyr_wt']['sync'].values.astype(float), s3[s3['type']=='pyr_ko']['sync'].values.astype(float))
t_fs_sess, p_fs_sess = mannwhitneyu(s3[s3['type']=='fs_wt']['sync'].values.astype(float), s3[s3['type']=='fs_ko']['sync'].values.astype(float))

### For Celltype (Fs)
t_pyr, p_pyr = mannwhitneyu(s4[s4['type']=='pyr_wt']['sync'].values.astype(float), s3[s3['type']=='pyr_ko']['sync'].values.astype(float))
t_fs, p_fs = mannwhitneyu(s4[s4['type']=='fs_wt']['sync'].values.astype(float), s3[s3['type']=='fs_ko']['sync'].values.astype(float))

#%% Plotting for celltype 

plt.figure()
plt.title('Population synchrony by session')
sns.set_style('white')
palette = ['royalblue', 'lightsteelblue', 'indianred', 'lightcoral']
ax = sns.violinplot( x = s3['type'], y=s3['sync'].astype(float) , data = s3, dodge=False,
                    palette = palette,cut = 2,
                    scale="width", inner=None)
ax.tick_params(bottom=True, left=True)
xlim = ax.get_xlim()
ylim = ax.get_ylim()
for violin in ax.collections:
    x0, y0, width, height = violin.get_paths()[0].get_extents().bounds
    violin.set_clip_path(plt.Rectangle((x0, y0), width / 2, height, transform=ax.transData))
sns.boxplot(x = s3['type'], y=s3['sync'] , data = s3, saturation=1, showfliers=False,
            width=0.3, boxprops={'zorder': 3, 'facecolor': 'none'}, ax=ax)
old_len_collections = len(ax.collections)
sns.swarmplot(x = s3['type'], y=s3['sync'], data = s1, color = 'k', dodge=False, ax=ax)
# sns.stripplot(x = wakedf['type'], y = wakedf['rate'].astype(float), data = wakedf, color = 'k', dodge=False, ax=ax)
for dots in ax.collections[old_len_collections:]:
    dots.set_offsets(dots.get_offsets())
ax.set_xlim(xlim)
ax.set_ylim(ylim)
plt.ylabel('Population Synchrony during ripple')
ax.set_box_aspect(1)

plt.figure()
plt.title('Population synchrony by event')
sns.set_style('white')
palette = ['royalblue', 'lightsteelblue', 'indianred', 'lightcoral']
ax = sns.violinplot( x = s4['type'], y=s4['sync'].astype(float) , data = s4, dodge=False,
                    palette = palette,cut = 2,
                    scale="width", inner=None)
ax.tick_params(bottom=True, left=True)
xlim = ax.get_xlim()
ylim = ax.get_ylim()
for violin in ax.collections:
    x0, y0, width, height = violin.get_paths()[0].get_extents().bounds
    violin.set_clip_path(plt.Rectangle((x0, y0), width / 2, height, transform=ax.transData))
sns.boxplot(x = s4['type'], y=s4['sync'] , data = s4, saturation=1, showfliers=False,
            width=0.3, boxprops={'zorder': 3, 'facecolor': 'none'}, ax=ax)

plt.ylabel('Population Synchrony during ripple')
ax.set_box_aspect(1)

