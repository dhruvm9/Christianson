#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 17:21:13 2024

@author: dhruv
"""

import numpy as np 
import pandas as pd
import nwbmatic as ntm
import pynapple as nap 
import os, sys
import seaborn as sns
import matplotlib.pyplot as plt 
from scipy.stats import mannwhitneyu

#%% 

data_directory = '/media/dhruv/Expansion/Processed'
datasets = np.genfromtxt(os.path.join(data_directory,'dataset_females.list'), delimiter = '\n', dtype = str, comments = '#')

allripdurs_wt = []
allripdurs_ko = []
allripdurs_f = []

allriprates_wt = []
allriprates_ko = []
allriprates_f = []

for s in datasets:
    print(s)
    name = s.split('-')[0]
    path = os.path.join(data_directory, s)
    
    if name == 'B2613' or name == 'B2618' or name == 'B2627' or name == 'B2628':
        isWT = 0
    elif name == 'B2632' or name == 'B2634': 
        isWT = 2
    else: isWT = 1
    
    file = os.path.join(path, s +'.evt.py.rip')
    if os.path.exists(file):
        tmp = np.genfromtxt(file)[:,0]
        tmp = tmp.reshape(len(tmp)//2,2)/1000
        rip_ep = nap.IntervalSet(start = tmp[:,0], end = tmp[:,1], time_units = 's')
        
    file = os.path.join(path, s +'.sws.evt')
    if os.path.exists(file):
        tmp = np.genfromtxt(file)[:,0]
        tmp = tmp.reshape(len(tmp)//2,2)/1000
        sws_ep = nap.IntervalSet(start = tmp[:,0], end = tmp[:,1], time_units = 's')
        
       
    
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
       # allriprates_wt.append(1 / np.mean(IRI))
    elif isWT == 0: 
        allripdurs_ko.append(np.mean(ripdur)*1e3)
        allriprates_ko.append(riprate) 
        # allriprates_ko.append(1 / np.mean(IRI))
    else: 
        allripdurs_f.append(np.mean(ripdur)*1e3)
        allriprates_f.append(riprate) 
        
         
    
#%% Organize data to plot 

wt = np.array(['WT_male' for x in range(len(allripdurs_wt))])
ko = np.array(['KO_male' for x in range(len(allripdurs_ko))])
fem = np.array(['KO_female' for x in range(len(allripdurs_ko))])

genotype = np.hstack([wt, ko, fem])

rippledurs = []
rippledurs.extend(allripdurs_wt)
rippledurs.extend(allripdurs_ko)
rippledurs.extend(allripdurs_f)

ripplerates = []
ripplerates.extend(allriprates_wt)
ripplerates.extend(allriprates_ko)
ripplerates.extend(allriprates_f)

durdf = pd.DataFrame(data = [rippledurs, genotype], index = ['dur', 'genotype']).T
ratedf = pd.DataFrame(data = [ripplerates, genotype], index = ['rate', 'genotype']).T

#%% Plotting 

plt.figure()
plt.title('NREM Ripple Durations')
sns.set_style('white')
palette = ['royalblue', 'indianred', 'darkslategray']
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
palette = ['royalblue', 'indianred', 'darkslategray']
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
plt.yticks([0,1,1.5])
plt.ylabel('Ripple rate (Hz)')
ax.set_box_aspect(1)


#%% Stats 

t_dur, p_dur = mannwhitneyu(allripdurs_wt, allripdurs_ko)
t_rate, p_rate = mannwhitneyu(allriprates_wt, allriprates_ko)