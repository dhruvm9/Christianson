#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 16:32:34 2024

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
datasets = np.genfromtxt(os.path.join(data_directory,'dataset_DM_rippletravel.list'), delimiter = '\n', dtype = str, comments = '#')
ripplechannels = np.genfromtxt(os.path.join(data_directory,'ripplechannel.list'), delimiter = '\n', dtype = str, comments = '#')

Fs = 1250

sess_part_sup_wt = []
sess_part_deep_wt = []

sess_part_sup_ko = []
sess_part_deep_ko = []

part_all_sup_wt = []
part_all_deep_wt = []

part_all_sup_ko = []
part_all_deep_ko = []

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

    sp2 = np.load(os.path.join(path, 'pyr_layers.npz'), allow_pickle = True)
    time_support = nap.IntervalSet(sp2['start'], sp2['end'])
    tsd = nap.Tsd(t=sp2['t'], d=sp2['index'], time_support = time_support)
    spikes = tsd.to_tsgroup()
    spikes.set_info(group = sp2['group'], location = sp2['location'], celltype = sp2['celltype'], layer = sp2['layer'])
    
#%% Ripple participation by cell type and genotype

    ripdur = rip_ep['end'] - rip_ep['start']        

    allpart_sup = []
    allpart_deep = []
    
    spikes_by_layer = spikes.getby_category('layer')
    
    if 'sup' in spikes._metadata['layer'].values:
        sup = spikes_by_layer['sup']
    else: sup = []
        
    if 'deep' in spikes._metadata['layer'].values:
        deep = spikes_by_layer['deep']
    else: deep = []
        
    # print (fs)
           
    if len(sup) > 0: 
               
        for i in sup:
            count = 0
            
            for j in range(len(rip_ep)):
                r_sup = sup[i].count(ripdur[j], nap.IntervalSet(start = rip_ep['start'][j], end = rip_ep['end'][j])) 
                
                if sum(r_sup) > 0:
                    count += 1
                    
            allpart_sup.append(count / len(rip_ep))
            
    if len(deep) > 0: 
            
        for i in deep:
            count = 0
            
            for j in range(len(rip_ep)):
                r_deep = deep[i].count(ripdur[j], nap.IntervalSet(start = rip_ep['start'][j], end = rip_ep['end'][j])) 
                
                if sum(r_deep) > 0:
                    count += 1
                    
            allpart_deep.append(count / len(rip_ep))
            


#%% Participation per session by genotype    

    if isWT == 1:
                  
        part_all_sup_wt.extend(allpart_sup)
        # sess_part_pyr_wt.append(np.mean(allpart_pyr))
        
        part_all_deep_wt.extend(allpart_deep)
        # sess_part_fs_wt.append(np.mean(allpart_fs))
        
        
    else: 
        
        part_all_sup_ko.extend(allpart_sup)
        # sess_part_pyr_ko.append(np.mean(allpart_pyr))
        
        part_all_deep_ko.extend(allpart_deep)
        # sess_part_fs_ko.append(np.mean(allpart_fs))
        
    del sup, deep

#%% Organize data by celltype 

# ex = np.array(['pyr_wt' for x in range(len(sess_part_pyr_wt))])
# ex2 = np.array(['pyr_ko' for x in range(len(sess_part_pyr_ko))])

# pv = np.array(['fs_wt' for x in range(len(sess_part_fs_wt))])
# pv2 = np.array(['fs_ko' for x in range(len(sess_part_fs_ko))])

ex3 = np.array(['sup_wt' for x in range(len(part_all_sup_wt))])
ex4 = np.array(['sup_ko' for x in range(len(part_all_sup_ko))])

pv3 = np.array(['deep_wt' for x in range(len(part_all_deep_wt))])
pv4 = np.array(['deep_ko' for x in range(len(part_all_deep_ko))])

# types = np.hstack([ex, ex2, pv, pv2])
types2 = np.hstack([ex3, ex4, pv3, pv4])

# sess_syncs_celltype = []
# sess_syncs_celltype.extend(sess_part_pyr_wt)
# sess_syncs_celltype.extend(sess_part_pyr_ko)
# sess_syncs_celltype.extend(sess_part_fs_wt)
# sess_syncs_celltype.extend(sess_part_fs_ko)

allsyncs_celltype = []
allsyncs_celltype.extend(part_all_sup_wt)
allsyncs_celltype.extend(part_all_sup_ko)
allsyncs_celltype.extend(part_all_deep_wt)
allsyncs_celltype.extend(part_all_deep_ko)

# s3 = pd.DataFrame(data = [sess_syncs_celltype, types], index = ['sync', 'type']).T
s4 = pd.DataFrame(data = [allsyncs_celltype, types2], index = ['sync', 'type']).T

#%% Stats 

# t_pyr_sess, p_pyr_sess = mannwhitneyu(s3[s3['type']=='pyr_wt']['sync'].values.astype(float), s3[s3['type']=='pyr_ko']['sync'].values.astype(float))
# t_fs_sess, p_fs_sess = mannwhitneyu(s3[s3['type']=='fs_wt']['sync'].values.astype(float), s3[s3['type']=='fs_ko']['sync'].values.astype(float))

t_sup, p_sup = mannwhitneyu(s4[s4['type']=='sup_wt']['sync'].values.astype(float), s4[s4['type']=='sup_ko']['sync'].values.astype(float))
t_fs, p_deep = mannwhitneyu(s4[s4['type']=='deep_wt']['sync'].values.astype(float), s4[s4['type']=='deep_ko']['sync'].values.astype(float))

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
plt.title('Ripple Participation of PYR cells')
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

# s4.to_csv(data_directory + '/Ripple_participation_single_units_layers.csv')

#%% 
            
# s3 = pd.read_csv(data_directory + '/Ripple_participation_session_avg.csv', index_col = 0)
# s4 = pd.read_csv(data_directory + '/Ripple_participation_single_units.csv', index_col = 0)