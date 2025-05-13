#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 15:55:47 2023

@author: dhruv
"""

import numpy as np 
import pandas as pd
import nwbmatic as ntm
import pynapple as nap 
import os, sys
import seaborn as sns
import matplotlib.pyplot as plt 
import warnings
from scipy.stats import mannwhitneyu

#%% 

warnings.filterwarnings("ignore")

# data_directory = '/media/dhruv/Expansion/Processed'
# data_directory = '/media/dhruv/Expansion/Processed/CA3'
data_directory = '/media/dhruv/Expansion/Processed/LinearTrack'
datasets = np.genfromtxt(os.path.join(data_directory,'dataset_DM.list'), delimiter = '\n', dtype = str, comments = '#')

allripdurs_wt = []
allripdurs_ko = []

allriprates_wt = []
allriprates_ko = []

KOmice = ['B2613', 'B2618', 'B2627', 'B2628', 'B3805', 'B3813', 'B4701', 'B4704', 'B4709']

for s in datasets:
    print(s)
    name = s.split('-')[0]
    path = os.path.join(data_directory, s)
    
    if name in KOmice:
        isWT = 0
    else: isWT = 1 
    
    data = ntm.load_session(path, 'neurosuite')
    data.load_neurosuite_xml(path)
        
    file = os.path.join(path, s +'.sws.evt')
    sws_ep = data.read_neuroscope_intervals(name = 'SWS', path2file = file)
        
    file = os.path.join(path, s +'.evt.py.rip')
    rip_ep = data.read_neuroscope_intervals(name = 'rip', path2file = file)
    
    # file = os.path.join(path, s +'.evt.py3sd.rip')
    # rip_ep = data.read_neuroscope_intervals(path2file = file)
    
    # file = os.path.join(path, s +'.evt.py5sd.rip')
    # rip_ep = data.read_neuroscope_intervals(path2file = file)
    
    # rip_ep = data.load_nwb_intervals('sleep_ripples')
    # rip_tsd = data.load_nwb_timeseries('sleep_ripples')
    
#%% Sort by genotype

    ripdur = rip_ep['end'] - rip_ep['start']    
    riprate = len(rip_ep)/sws_ep.tot_length('s')   #len(rip_ep)/sws_ep.tot_length('s')
    
    # IRI = []
    # for i in range(len(rip_ep)-1):
    #     iei = rip_ep['start'][i+1] - rip_ep['end'][i]
    #     IRI.append(iei)

    if isWT == 1: 
       allripdurs_wt.append(np.mean(ripdur)*1e3)
       allriprates_wt.append(riprate)
       print('WT')
       # allriprates_wt.append(1 / np.mean(IRI))
    else: 
        allripdurs_ko.append(np.mean(ripdur)*1e3)
        allriprates_ko.append(riprate) 
        print('KO')
        # allriprates_ko.append(1 / np.mean(IRI))
         
    
#%% Organize data to plot 

wt = np.array(['WT' for x in range(len(allripdurs_wt))])
ko = np.array(['KO' for x in range(len(allripdurs_ko))])

genotype = np.hstack([wt, ko])

rippledurs = []
rippledurs.extend(allripdurs_wt)
rippledurs.extend(allripdurs_ko)

ripplerates = []
ripplerates.extend(allriprates_wt)
ripplerates.extend(allriprates_ko)

durdf = pd.DataFrame(data = [rippledurs, genotype], index = ['dur', 'genotype']).T
ratedf = pd.DataFrame(data = [ripplerates, genotype], index = ['rate', 'genotype']).T

#%% Plotting 

plt.figure()
plt.title('NREM Ripple Durations')
sns.set_style('white')
palette = ['royalblue', 'indianred']
ax = sns.violinplot( x = durdf['genotype'], y=durdf['dur'].astype(float) , data = durdf, dodge=False,
                    palette = palette,cut = 2,
                    scale="width", inner=None)
ax.tick_params(bottom=True, left=True)
xlim = ax.get_xlim()
ylim = ax.get_ylim()
for violin in ax.collections:
    x0, y0, width, height = violin.get_paths()[0].get_extents().bounds
    violin.set_clip_path(plt.Rectangle((x0, y0), width / 2, height, transform=ax.transData))
sns.boxplot(x = durdf['genotype'], y=durdf['dur'].astype(float) , data = durdf, saturation=1, showfliers=False,
            width=0.3, boxprops={'zorder': 3, 'facecolor': 'none'}, ax=ax)
old_len_collections = len(ax.collections)
sns.stripplot(x = durdf['genotype'], y = durdf['dur'].astype(float), data = durdf, color = 'k', dodge=False, ax=ax)
# sns.swarmplot(x = durdf['genotype'], y = durdf['dur'].astype(float), data = durdf, color = 'k', dodge=False, ax=ax)
for dots in ax.collections[old_len_collections:]:
    dots.set_offsets(dots.get_offsets())
ax.set_xlim(xlim)
ax.set_ylim(ylim)
plt.ylabel('Ripple duration (ms)')
ax.set_box_aspect(1)

plt.figure()
plt.title('NREM Ripple Rates')
sns.set_style('white')
palette = ['royalblue', 'indianred']
ax = sns.violinplot( x = ratedf['genotype'], y=ratedf['rate'].astype(float) , data = ratedf, dodge=False,
                    palette = palette,cut = 2,
                    scale="width", inner=None)
ax.tick_params(bottom=True, left=True)
xlim = ax.get_xlim()
ylim = ax.get_ylim()
for violin in ax.collections:
    x0, y0, width, height = violin.get_paths()[0].get_extents().bounds
    violin.set_clip_path(plt.Rectangle((x0, y0), width / 2, height, transform=ax.transData))
sns.boxplot(x = ratedf['genotype'], y=ratedf['rate'].astype(float) , data = ratedf, saturation=1, showfliers=False,
            width=0.3, boxprops={'zorder': 3, 'facecolor': 'none'}, ax=ax)
old_len_collections = len(ax.collections)
sns.stripplot(x = ratedf['genotype'], y = ratedf['rate'].astype(float), data = ratedf, color = 'k', dodge=False, ax=ax)
# sns.swarmplot(x = durdf['genotype'], y = durdf['dur'].astype(float), data = durdf, color = 'k', dodge=False, ax=ax)
for dots in ax.collections[old_len_collections:]:
    dots.set_offsets(dots.get_offsets())
ax.set_xlim(xlim)
ax.set_ylim(ylim)
plt.ylabel('Ripple rate (Hz)')
ax.set_box_aspect(1)


#%% Stats 

t_dur, p_dur = mannwhitneyu(allripdurs_wt, allripdurs_ko)
t_rate, p_rate = mannwhitneyu(allriprates_wt, allriprates_ko)