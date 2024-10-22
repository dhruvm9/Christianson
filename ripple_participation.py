#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 10:54:22 2024

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

sess_part_pyr_wt = []
sess_part_fs_wt = []

sess_part_pyr_ko = []
sess_part_fs_ko = []

part_all_pyr_wt = []
part_all_fs_wt = []

part_all_pyr_ko = []
part_all_fs_ko = []

for s in datasets:
    print(s)
    
            
    name = s.split('-')[0]
    path = os.path.join(data_directory, s)
    
    data = ntm.load_session(path, 'neurosuite')
    data.load_neurosuite_xml(path)
    epochs = data.epochs
    
    if name == 'B2613' or name == 'B2618' or name == 'B2627' or name == 'B2628':
        isWT = 0
    else: isWT = 1 
          
    file = os.path.join(path, s +'.sws.evt')
    sws_ep = data.read_neuroscope_intervals(name = 'SWS', path2file = file)
    
    file = os.path.join(path, s +'.rem.evt')
    rem_ep = data.read_neuroscope_intervals(name = 'REM', path2file = file)
    
    file = os.path.join(path, s +'.evt.py.rip')
    rip_ep = data.read_neuroscope_intervals(name = 'rip', path2file = file)
    
#%% Load classified spikes 

    # sp2 = np.load(os.path.join(path, 'spikedata.npz'), allow_pickle = True)
    sp2 = np.load(os.path.join(path, 'spikedata_0.55.npz'), allow_pickle = True)
    time_support = nap.IntervalSet(sp2['start'], sp2['end'])
    tsd = nap.Tsd(t=sp2['t'], d=sp2['index'], time_support = time_support)
    spikes = tsd.to_tsgroup()
    spikes.set_info(group = sp2['group'], location = sp2['location'], celltype = sp2['celltype'], tr2pk = sp2['tr2pk'])
    
#%% Ripple participation by cell type and genotype

    ripdur = rip_ep['end'] - rip_ep['start']        

    allpart_pyr = []
    allpart_fs = []
    
    spikes_by_celltype = spikes.getby_category('celltype')
    if 'pyr' in spikes._metadata['celltype'].values:
        pyr = spikes_by_celltype['pyr']
    else: pyr = []     
    
    keep = []
    
    for i in pyr.index:
        if pyr.restrict(nap.IntervalSet(epochs['wake'][0]))._metadata['rate'][i] > 0.5:
            keep.append(i)

    pyr2 = pyr[keep]
       
    if 'fs' in spikes._metadata['celltype'].values:
        fs = spikes_by_celltype['fs']
    else: fs = []        
        
    # print (fs)
           
    if len(pyr2) > 0: 
               
        for i in pyr2:
            count = 0
            
            for j in range(len(rip_ep)):
                r_pyr = pyr2[i].count(ripdur[j], nap.IntervalSet(start = rip_ep['start'][j], end = rip_ep['end'][j])) 
                
                if sum(r_pyr) > 0:
                    count += 1
                    
            allpart_pyr.append(count / len(rip_ep))
            
    if len(fs) > 0: 
            
        for i in fs:
            count = 0
            
            for j in range(len(rip_ep)):
                r_fs = fs[i].count(ripdur[j], nap.IntervalSet(start = rip_ep['start'][j], end = rip_ep['end'][j])) 
                
                if sum(r_fs) > 0:
                    count += 1
                    
            allpart_fs.append(count / len(rip_ep))
            


#%% Participation per session by genotype    

    if isWT == 1:
                  
        part_all_pyr_wt.extend(allpart_pyr)
        # sess_part_pyr_wt.append(np.mean(allpart_pyr))
        
        part_all_fs_wt.extend(allpart_fs)
        # sess_part_fs_wt.append(np.mean(allpart_fs))
        
        
    else: 
        
        part_all_pyr_ko.extend(allpart_pyr)
        # sess_part_pyr_ko.append(np.mean(allpart_pyr))
        
        part_all_fs_ko.extend(allpart_fs)
        # sess_part_fs_ko.append(np.mean(allpart_fs))
        
    del pyr, pyr2, fs

#%% Organize data by celltype 

# ex = np.array(['pyr_wt' for x in range(len(sess_part_pyr_wt))])
# ex2 = np.array(['pyr_ko' for x in range(len(sess_part_pyr_ko))])

# pv = np.array(['fs_wt' for x in range(len(sess_part_fs_wt))])
# pv2 = np.array(['fs_ko' for x in range(len(sess_part_fs_ko))])

ex3 = np.array(['pyr_wt' for x in range(len(part_all_pyr_wt))])
ex4 = np.array(['pyr_ko' for x in range(len(part_all_pyr_ko))])

pv3 = np.array(['fs_wt' for x in range(len(part_all_fs_wt))])
pv4 = np.array(['fs_ko' for x in range(len(part_all_fs_ko))])

# types = np.hstack([ex, ex2, pv, pv2])
types2 = np.hstack([ex3, ex4, pv3, pv4])

# sess_syncs_celltype = []
# sess_syncs_celltype.extend(sess_part_pyr_wt)
# sess_syncs_celltype.extend(sess_part_pyr_ko)
# sess_syncs_celltype.extend(sess_part_fs_wt)
# sess_syncs_celltype.extend(sess_part_fs_ko)

allsyncs_celltype = []
allsyncs_celltype.extend(part_all_pyr_wt)
allsyncs_celltype.extend(part_all_pyr_ko)
allsyncs_celltype.extend(part_all_fs_wt)
allsyncs_celltype.extend(part_all_fs_ko)

# s3 = pd.DataFrame(data = [sess_syncs_celltype, types], index = ['sync', 'type']).T
s4 = pd.DataFrame(data = [allsyncs_celltype, types2], index = ['sync', 'type']).T

#%% Stats 

### For Celltype (Pyr)
# t_pyr_sess, p_pyr_sess = mannwhitneyu(s3[s3['type']=='pyr_wt']['sync'].values.astype(float), s3[s3['type']=='pyr_ko']['sync'].values.astype(float))
# t_fs_sess, p_fs_sess = mannwhitneyu(s3[s3['type']=='fs_wt']['sync'].values.astype(float), s3[s3['type']=='fs_ko']['sync'].values.astype(float))

### For Celltype (Fs)
t_pyr, p_pyr = mannwhitneyu(s4[s4['type']=='pyr_wt']['sync'].values.astype(float), s4[s4['type']=='pyr_ko']['sync'].values.astype(float))
t_fs, p_fs = mannwhitneyu(s4[s4['type']=='fs_wt']['sync'].values.astype(float), s4[s4['type']=='fs_ko']['sync'].values.astype(float))

#%% Plotting for celltype 

# plt.figure()
# plt.title('Ripple Participation by session')
# sns.set_style('white')
# palette = ['royalblue', 'lightsteelblue', 'indianred', 'lightcoral']
# ax = sns.violinplot( x = s3['type'], y=s3['sync'].astype(float) , data = s3, dodge=False,
#                     palette = palette,cut = 2,
#                     scale="width", inner=None)
# ax.tick_params(bottom=True, left=True)
# xlim = ax.get_xlim()
# ylim = ax.get_ylim()
# for violin in ax.collections:
#     x0, y0, width, height = violin.get_paths()[0].get_extents().bounds
#     violin.set_clip_path(plt.Rectangle((x0, y0), width / 2, height, transform=ax.transData))
# sns.boxplot(x = s3['type'], y=s3['sync'] , data = s3, saturation=1, showfliers=False,
#             width=0.3, boxprops={'zorder': 3, 'facecolor': 'none'}, ax=ax)
# old_len_collections = len(ax.collections)
# sns.swarmplot(x = s3['type'], y=s3['sync'], data = s3, color = 'k', dodge=False, ax=ax)
# # sns.stripplot(x = wakedf['type'], y = wakedf['rate'].astype(float), data = wakedf, color = 'k', dodge=False, ax=ax)
# for dots in ax.collections[old_len_collections:]:
#     dots.set_offsets(dots.get_offsets())
# ax.set_xlim(xlim)
# ax.set_ylim(ylim)
# plt.ylabel('Proportion of ripples')
# ax.set_box_aspect(1)

plt.figure()
plt.title('Ripple Participation of individual cells')
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
sns.stripplot(x = s4['type'], y = s4['sync'].astype(float), data = s4, color = 'k', dodge=False, ax=ax, alpha = 0.2)
plt.ylabel('Proportion of ripples')
ax.set_box_aspect(1)

#%% 

# s3.to_csv(data_directory + '/Ripple_participation_session_avg.csv')
# s4.to_csv(data_directory + '/Ripple_participation_single_units.csv')
s4.to_csv(data_directory + '/Ripple_participation_single_units_0.55.csv')

#%% 
            
# s3 = pd.read_csv(data_directory + '/Ripple_participation_session_avg.csv', index_col = 0)
# s4 = pd.read_csv(data_directory + '/Ripple_participation_single_units.csv', index_col = 0)