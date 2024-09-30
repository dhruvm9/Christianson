#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 15:49:53 2023

@author: dhruv
"""

import numpy as np 
import pandas as pd 
import nwbmatic as ntm
import pynapple as nap
import os, sys
import matplotlib.pyplot as plt 
import seaborn as sns 
from scipy.stats import mannwhitneyu

#%% 

data_directory = '/media/dhruv/Expansion/Processed'
datasets = np.genfromtxt(os.path.join(data_directory,'dataset_DM.list'), delimiter = '\n', dtype = str, comments = '#')

celltype = []
genotype = []

pyr_wake_rate_wt = []
pyr_nrem_rate_wt = []
pyr_rem_rate_wt = []

fs_wake_rate_wt = []
fs_nrem_rate_wt = []
fs_rem_rate_wt = []

oth_wake_rate_wt = []
oth_nrem_rate_wt = []
oth_rem_rate_wt = []

pyr_wake_rate_ko = []
pyr_nrem_rate_ko = []
pyr_rem_rate_ko = []

fs_wake_rate_ko = []
fs_nrem_rate_ko = []
fs_rem_rate_ko = []

oth_wake_rate_ko = []
oth_nrem_rate_ko = []
oth_rem_rate_ko = []

pyr_rip_rate_wt = []
pyr_rip_rate_ko = []

fs_rip_rate_wt = []
fs_rip_rate_ko = []

oth_rip_rate_wt = []
oth_rip_rate_ko = []


for s in datasets[1:]:
    print(s)
    name = s.split('-')[0]
    path = os.path.join(data_directory, s)
    
    if name == 'B2613' or name == 'B2618'  or name == 'B2627' or name == 'B2628':
        isWT = 0
    else: isWT = 1 
    
    data = ntm.load_session(path, 'neurosuite')
    epochs = data.epochs
    
    file = os.path.join(path, s +'.sws.evt')
    sws_ep = nap.IntervalSet(data.read_neuroscope_intervals(name = 'SWS', path2file = file))
    
    file = os.path.join(path, s +'.rem.evt')
    rem_ep = nap.IntervalSet(data.read_neuroscope_intervals(name = 'REM', path2file = file))
    
    file = os.path.join(path, s +'.evt.py.rip')
    rip_ep = data.read_neuroscope_intervals(name = 'rip', path2file = file)
    
    
        
#%% Load classified spikes 

    sp2 = np.load(os.path.join(path, 'spikedata_0.55.npz'), allow_pickle = True)
    time_support = nap.IntervalSet(sp2['start'], sp2['end'])
    tsd = nap.Tsd(t=sp2['t'], d=sp2['index'], time_support = time_support)
    spikes = tsd.to_tsgroup()
    spikes.set_info(group = sp2['group'], location = sp2['location'], celltype = sp2['celltype'], tr2pk = sp2['tr2pk'])
     
    
#%% Sort by brain state
    
    wk = spikes.restrict(nap.IntervalSet(epochs['wake']))
    nr = spikes.restrict(sws_ep)
    rm = spikes.restrict(rem_ep)
    
    rp = spikes.restrict(nap.IntervalSet(rip_ep))
    
#%% Sort by genotype

    if isWT == 1:
        if 'pyr' in wk.getby_category('celltype').keys():
        
            pyr_wake_rate_wt.extend(wk.getby_category('celltype')['pyr']._metadata['rate'].values)
            pyr_nrem_rate_wt.extend(nr.getby_category('celltype')['pyr']._metadata['rate'].values)
            pyr_rem_rate_wt.extend(rm.getby_category('celltype')['pyr']._metadata['rate'].values)
            pyr_rip_rate_wt.extend(rp.getby_category('celltype')['pyr']._metadata['rate'].values)
        
               
        if 'fs' in wk.getby_category('celltype').keys():
        
            fs_wake_rate_wt.extend(wk.getby_category('celltype')['fs']._metadata['rate'].values)
            fs_nrem_rate_wt.extend(nr.getby_category('celltype')['fs']._metadata['rate'].values)
            fs_rem_rate_wt.extend(rm.getby_category('celltype')['fs']._metadata['rate'].values)
            fs_rip_rate_wt.extend(rp.getby_category('celltype')['fs']._metadata['rate'].values)
            
        
        if 'other' in wk.getby_category('celltype').keys():
        
            oth_wake_rate_wt.extend(wk.getby_category('celltype')['other']._metadata['rate'].values)
            oth_nrem_rate_wt.extend(nr.getby_category('celltype')['other']._metadata['rate'].values)
            oth_rem_rate_wt.extend(rm.getby_category('celltype')['other']._metadata['rate'].values)
            oth_rip_rate_wt.extend(rp.getby_category('celltype')['other']._metadata['rate'].values)
        
    else: 
        
        if 'pyr' in wk.getby_category('celltype').keys():
            
            pyr_wake_rate_ko.extend(wk.getby_category('celltype')['pyr']._metadata['rate'].values)
            pyr_nrem_rate_ko.extend(nr.getby_category('celltype')['pyr']._metadata['rate'].values)
            pyr_rem_rate_ko.extend(rm.getby_category('celltype')['pyr']._metadata['rate'].values)
            pyr_rip_rate_ko.extend(rp.getby_category('celltype')['pyr']._metadata['rate'].values)
        
        if 'fs' in wk.getby_category('celltype').keys():
        
            fs_wake_rate_ko.extend(wk.getby_category('celltype')['fs']._metadata['rate'].values)
            fs_nrem_rate_ko.extend(nr.getby_category('celltype')['fs']._metadata['rate'].values)
            fs_rem_rate_ko.extend(rm.getby_category('celltype')['fs']._metadata['rate'].values)
            fs_rip_rate_ko.extend(rp.getby_category('celltype')['fs']._metadata['rate'].values)
        
        if 'other' in wk.getby_category('celltype').keys():
        
            oth_wake_rate_ko.extend(wk.getby_category('celltype')['other']._metadata['rate'].values)
            oth_nrem_rate_ko.extend(nr.getby_category('celltype')['other']._metadata['rate'].values)
            oth_rem_rate_ko.extend(rm.getby_category('celltype')['other']._metadata['rate'].values)
            oth_rip_rate_ko.extend(rp.getby_category('celltype')['other']._metadata['rate'].values)
            
    
    del wk, nr, rm, rp
        
#%% Organize data to plot 

ex = np.array(['pyr_wt' for x in range(len(pyr_wake_rate_wt))])
ex2 = np.array(['pyr_ko' for x in range(len(pyr_wake_rate_ko))])

pv = np.array(['fs_wt' for x in range(len(fs_wake_rate_wt))])
pv2 = np.array(['fs_ko' for x in range(len(fs_wake_rate_ko))])

unc = np.array(['oth_wt' for x in range(len(oth_wake_rate_wt))])
unc2 = np.array(['oth_ko' for x in range(len(oth_wake_rate_ko))])

types = np.hstack([ex, ex2, pv, pv2, unc, unc2])

wakerates = []
wakerates.extend(pyr_wake_rate_wt)
wakerates.extend(pyr_wake_rate_ko)
wakerates.extend(fs_wake_rate_wt)
wakerates.extend(fs_wake_rate_ko)
wakerates.extend(oth_wake_rate_wt)
wakerates.extend(oth_wake_rate_ko)

wakedf = pd.DataFrame(data = [wakerates, types], index = ['rate', 'type']).T

##NREM 
ex = np.array(['pyr_wt' for x in range(len(pyr_nrem_rate_wt))])
ex2 = np.array(['pyr_ko' for x in range(len(pyr_nrem_rate_ko))])

pv = np.array(['fs_wt' for x in range(len(fs_nrem_rate_wt))])
pv2 = np.array(['fs_ko' for x in range(len(fs_nrem_rate_ko))])

unc = np.array(['oth_wt' for x in range(len(oth_nrem_rate_wt))])
unc2 = np.array(['oth_ko' for x in range(len(oth_nrem_rate_ko))])

types = np.hstack([ex, ex2, pv, pv2, unc, unc2])

nremrates = []
nremrates.extend(pyr_nrem_rate_wt)
nremrates.extend(pyr_nrem_rate_ko)
nremrates.extend(fs_nrem_rate_wt)
nremrates.extend(fs_nrem_rate_ko)
nremrates.extend(oth_nrem_rate_wt)
nremrates.extend(oth_nrem_rate_ko)

nremdf = pd.DataFrame(data = [nremrates, types], index = ['rate', 'type']).T

##REM 
ex = np.array(['pyr_wt' for x in range(len(pyr_rem_rate_wt))])
ex2 = np.array(['pyr_ko' for x in range(len(pyr_rem_rate_ko))])

pv = np.array(['fs_wt' for x in range(len(fs_rem_rate_wt))])
pv2 = np.array(['fs_ko' for x in range(len(fs_rem_rate_ko))])

unc = np.array(['oth_wt' for x in range(len(oth_rem_rate_wt))])
unc2 = np.array(['oth_ko' for x in range(len(oth_rem_rate_ko))])

types = np.hstack([ex, ex2, pv, pv2, unc, unc2])

remrates = []
remrates.extend(pyr_rem_rate_wt)
remrates.extend(pyr_rem_rate_ko)
remrates.extend(fs_rem_rate_wt)
remrates.extend(fs_rem_rate_ko)
remrates.extend(oth_rem_rate_wt)
remrates.extend(oth_rem_rate_ko)

remdf = pd.DataFrame(data = [remrates, types], index = ['rate', 'type']).T


#%% Plotting 

plt.figure()
plt.suptitle('Firing rate profile')
plt.subplot(131)
plt.title('Wake')
sns.set_style('white')
palette = ['royalblue', 'lightsteelblue', 'indianred', 'lightcoral', 'darkslategray','cadetblue']
ax = sns.violinplot( x = wakedf['type'], y=wakedf['rate'].astype(float) , data = wakedf, dodge=False,
                    palette = palette,cut = 2,
                    scale="width", inner=None)
ax.tick_params(bottom=True, left=True)
xlim = ax.get_xlim()
ylim = [-10, 80]
for violin in ax.collections:
    x0, y0, width, height = violin.get_paths()[0].get_extents().bounds
    violin.set_clip_path(plt.Rectangle((x0, y0), width / 2, height, transform=ax.transData))
sns.boxplot(x = wakedf['type'], y=wakedf['rate'].astype(float) , data = wakedf, saturation=1, showfliers=False,
            width=0.3, boxprops={'zorder': 3, 'facecolor': 'none'}, ax=ax)
old_len_collections = len(ax.collections)
sns.stripplot(x = wakedf['type'], y = wakedf['rate'].astype(float), data = wakedf, color = 'k', dodge=False, ax=ax, alpha = 0.2)
# sns.swarmplot(x = wakedf['type'], y = wakedf['rate'].astype(float), data = wakedf, color = 'k', dodge=False, ax=ax)
for dots in ax.collections[old_len_collections:]:
    dots.set_offsets(dots.get_offsets())
ax.set_xlim(xlim)
ax.set_ylim(ylim)
plt.ylabel('Firing rate (Hz)')
ax.set_box_aspect(1)

plt.subplot(132)
plt.title('NREM')
sns.set_style('white')
palette = ['royalblue', 'lightsteelblue', 'indianred', 'lightcoral', 'darkslategray','cadetblue']
ax = sns.violinplot( x = nremdf['type'], y=nremdf['rate'].astype(float) , data = nremdf, dodge=False,
                    palette = palette,cut = 2,
                    scale="width", inner=None)
ax.tick_params(bottom=True, left=True)
xlim = ax.get_xlim()
ylim = [-10, 80]
for violin in ax.collections:
    x0, y0, width, height = violin.get_paths()[0].get_extents().bounds
    violin.set_clip_path(plt.Rectangle((x0, y0), width / 2, height, transform=ax.transData))
sns.boxplot(x = nremdf['type'], y=nremdf['rate'].astype(float) , data = nremdf, saturation=1, showfliers=False,
            width=0.3, boxprops={'zorder': 3, 'facecolor': 'none'}, ax=ax)
old_len_collections = len(ax.collections)
sns.stripplot(x = nremdf['type'], y = nremdf['rate'].astype(float), data = nremdf, color = 'k', dodge=False, ax=ax, alpha = 0.2)
# sns.swarmplot(x = wakedf['type'], y = wakedf['rate'].astype(float), data = wakedf, color = 'k', dodge=False, ax=ax)
for dots in ax.collections[old_len_collections:]:
    dots.set_offsets(dots.get_offsets())
ax.set_xlim(xlim)
ax.set_ylim(ylim)
plt.ylabel('Firing rate (Hz)')
ax.set_box_aspect(1)

plt.subplot(133)
plt.title('REM')
sns.set_style('white')
palette = ['royalblue', 'lightsteelblue', 'indianred', 'lightcoral', 'darkslategray','cadetblue']
ax = sns.violinplot( x = remdf['type'], y=remdf['rate'].astype(float) , data = remdf, dodge=False,
                    palette = palette,cut = 2,
                    scale="width", inner=None)
ax.tick_params(bottom=True, left=True)
xlim = ax.get_xlim()
ylim = [-10, 80]
for violin in ax.collections:
    x0, y0, width, height = violin.get_paths()[0].get_extents().bounds
    violin.set_clip_path(plt.Rectangle((x0, y0), width / 2, height, transform=ax.transData))
sns.boxplot(x = remdf['type'], y=remdf['rate'].astype(float) , data = remdf, saturation=1, showfliers=False,
            width=0.3, boxprops={'zorder': 3, 'facecolor': 'none'}, ax=ax)
old_len_collections = len(ax.collections)
sns.stripplot(x = remdf['type'], y = remdf['rate'].astype(float), data = remdf, color = 'k', dodge=False, ax=ax, alpha = 0.2)
# sns.swarmplot(x = wakedf['type'], y = wakedf['rate'].astype(float), data = wakedf, color = 'k', dodge=False, ax=ax)
for dots in ax.collections[old_len_collections:]:
    dots.set_offsets(dots.get_offsets())
ax.set_xlim(xlim)
ax.set_ylim(ylim)
plt.ylabel('Firing rate (Hz)')
ax.set_box_aspect(1)

#%% Ratio of Ripple rate and NREM rate 

pyr_stab_wt = [i / j for i, j in zip(pyr_rip_rate_wt, pyr_nrem_rate_wt)]
fs_stab_wt = [i / j for i, j in zip(fs_rip_rate_wt, fs_nrem_rate_wt)]
oth_stab_wt = [i / j for i, j in zip(oth_rip_rate_wt, oth_nrem_rate_wt)]

pyr_stab_ko = [i / j for i, j in zip(pyr_rip_rate_ko, pyr_nrem_rate_ko)]
fs_stab_ko = [i / j for i, j in zip(fs_rip_rate_ko, fs_nrem_rate_ko)]
oth_stab_ko = [i / j for i, j in zip(oth_rip_rate_ko, oth_nrem_rate_ko)]

ex = np.array(['pyr_wt' for x in range(len(pyr_stab_wt))])
ex2 = np.array(['pyr_ko' for x in range(len(pyr_stab_ko))])

pv = np.array(['fs_wt' for x in range(len(fs_stab_wt))])
pv2 = np.array(['fs_ko' for x in range(len(fs_stab_ko))])

unc = np.array(['oth_wt' for x in range(len(oth_stab_wt))])
unc2 = np.array(['oth_ko' for x in range(len(oth_stab_ko))])

types = np.hstack([ex, ex2, pv, pv2, unc, unc2])
# types = np.hstack([pv, pv2, unc, unc2])

stabs = []
stabs.extend(pyr_stab_wt)
stabs.extend(pyr_stab_ko)
stabs.extend(fs_stab_wt)
stabs.extend(fs_stab_ko)
stabs.extend(oth_stab_wt)
stabs.extend(oth_stab_ko)

stabdf = pd.DataFrame(data = [stabs, types], index = ['rate', 'type']).T

plt.figure()
plt.title('Firing rate gain during ripple compared to NREM')
sns.set_style('white')
palette = ['royalblue', 'lightsteelblue', 'indianred', 'lightcoral', 'darkslategray','cadetblue']
# palette = ['indianred', 'lightcoral', 'darkslategray','cadetblue']
ax = sns.violinplot( x = stabdf['type'], y=stabdf['rate'].astype(float) , data = stabdf, dodge=False,
                    palette = palette,cut = 2,
                    scale="width", inner=None)
ax.tick_params(bottom=True, left=True)
xlim = ax.get_xlim()
ylim = ax.get_ylim()
for violin in ax.collections:
    x0, y0, width, height = violin.get_paths()[0].get_extents().bounds
    violin.set_clip_path(plt.Rectangle((x0, y0), width / 2, height, transform=ax.transData))
sns.boxplot(x = stabdf['type'], y=stabdf['rate'].astype(float) , data = stabdf, saturation=1, showfliers=False,
            width=0.3, boxprops={'zorder': 3, 'facecolor': 'none'}, ax=ax)
old_len_collections = len(ax.collections)
sns.stripplot(x = stabdf['type'], y = stabdf['rate'].astype(float), data = stabdf, color = 'k', dodge=False, ax=ax, alpha = 0.2)
# sns.swarmplot(x = wakedf['type'], y = wakedf['rate'].astype(float), data = wakedf, color = 'k', dodge=False, ax=ax)
for dots in ax.collections[old_len_collections:]:
    dots.set_offsets(dots.get_offsets())
ax.set_xlim(xlim)
ax.set_ylim(ylim)
plt.ylabel('Ripple FR normalized to NREM')
ax.set_box_aspect(1)

t_pyr_stab,p_pyr_stab = mannwhitneyu(pyr_stab_wt, pyr_stab_ko)
t_fs_stab,p_fs_stab = mannwhitneyu(fs_stab_wt, fs_stab_ko)
t_oth_stab,p_oth_stab = mannwhitneyu(oth_stab_wt, oth_stab_ko)

#%% Stats 

t_pyr_wake,p_pyr_wake = mannwhitneyu(pyr_wake_rate_wt, pyr_wake_rate_ko)
t_pyr_nrem,p_pyr_nrem = mannwhitneyu(pyr_nrem_rate_wt, pyr_nrem_rate_ko)
t_pyr_rem, p_pyr_rem = mannwhitneyu(pyr_rem_rate_wt, pyr_rem_rate_ko)
t_pyr_rip, p_pyr_rip = mannwhitneyu(pyr_rip_rate_wt, pyr_rip_rate_ko)

t_fs_wake,p_fs_wake = mannwhitneyu(fs_wake_rate_wt, fs_wake_rate_ko)
t_fs_nrem,p_fs_nrem = mannwhitneyu(fs_nrem_rate_wt, fs_nrem_rate_ko)
t_fs_rem, p_fs_rem = mannwhitneyu(fs_rem_rate_wt, fs_rem_rate_ko)
t_fs_rip, p_fs_rip = mannwhitneyu(fs_rip_rate_wt, fs_rip_rate_ko)

t_oth_wake,p_oth_wake = mannwhitneyu(oth_wake_rate_wt, oth_wake_rate_ko)
t_oth_nrem,p_oth_nrem = mannwhitneyu(oth_nrem_rate_wt, oth_nrem_rate_ko)
t_oth_rem, p_oth_rem = mannwhitneyu(oth_rem_rate_wt, oth_rem_rate_ko)
t_oth_rip, p_oth_rip = mannwhitneyu(oth_rip_rate_wt, oth_rip_rate_ko)

#%% Just ripple rates and plot

ex = np.array(['pyr_wt' for x in range(len(pyr_rip_rate_wt))])
ex2 = np.array(['pyr_ko' for x in range(len(pyr_rip_rate_ko))])

pv = np.array(['fs_wt' for x in range(len(fs_rip_rate_wt))])
pv2 = np.array(['fs_ko' for x in range(len(fs_rip_rate_ko))])

# unc = np.array(['oth_wt' for x in range(len(oth_rip_rate_wt))])
# unc2 = np.array(['oth_ko' for x in range(len(oth_rip_rate_ko))])

# types = np.hstack([ex, ex2, pv, pv2, unc, unc2])
types = np.hstack([ex, ex2, pv, pv2])

riprates = []
riprates.extend(pyr_rip_rate_wt)
riprates.extend(pyr_rip_rate_ko)
riprates.extend(fs_rip_rate_wt)
riprates.extend(fs_rip_rate_ko)
# riprates.extend(oth_rip_rate_wt)
# riprates.extend(oth_rip_rate_ko)

ripdf = pd.DataFrame(data = [riprates, types], index = ['rate', 'type']).T

plt.figure()
plt.title('Firing rate during ripples')
sns.set_style('white')
palette = ['royalblue', 'lightsteelblue', 'indianred', 'lightcoral', 'darkslategray','cadetblue']
ax = sns.violinplot( x = ripdf['type'], y=ripdf['rate'].astype(float) , data = ripdf, dodge=False,
                    palette = palette,cut = 2,
                    scale="width", inner=None)
ax.tick_params(bottom=True, left=True)
xlim = ax.get_xlim()
ylim = ax.get_ylim()
for violin in ax.collections:
    x0, y0, width, height = violin.get_paths()[0].get_extents().bounds
    violin.set_clip_path(plt.Rectangle((x0, y0), width / 2, height, transform=ax.transData))
sns.boxplot(x = ripdf['type'], y=ripdf['rate'].astype(float) , data = ripdf, saturation=1, showfliers=False,
            width=0.3, boxprops={'zorder': 3, 'facecolor': 'none'}, ax=ax)
old_len_collections = len(ax.collections)
sns.stripplot(x = ripdf['type'], y = ripdf['rate'].astype(float), data = ripdf, color = 'k', dodge=False, ax=ax, alpha = 0.2)
# sns.swarmplot(x = wakedf['type'], y = wakedf['rate'].astype(float), data = wakedf, color = 'k', dodge=False, ax=ax)
for dots in ax.collections[old_len_collections:]:
    dots.set_offsets(dots.get_offsets())
ax.set_xlim(xlim)
ax.set_ylim(ylim)
plt.ylabel('Firing rate (Hz)')
ax.set_box_aspect(1)
