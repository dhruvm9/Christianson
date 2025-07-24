#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 10:38:59 2024

@author: dhruv
"""

import numpy as np 
import nwbmatic as ntm
import pynapple as nap 
import pynacollada as pyna
import os, sys
import matplotlib.pyplot as plt 
import pandas as pd
import pickle
import seaborn as sns 
from scipy.signal import filtfilt
from scipy.stats import mannwhitneyu

#%%

# data_directory = '/media/dhruv/Expansion/Processed'
data_directory = '/media/dhruv/Expansion/Processed/CA3'
datasets = np.genfromtxt(os.path.join(data_directory,'dataset_DM.list'), delimiter = '\n', dtype = str, comments = '#')
# datasets = np.genfromtxt(os.path.join(data_directory,'dataset_new_toadd.list'), delimiter = '\n', dtype = str, comments = '#')
ripplechannels = np.genfromtxt(os.path.join(data_directory,'ripplechannel.list'), delimiter = '\n', dtype = str, comments = '#')
# ripplechannels = np.genfromtxt(os.path.join(data_directory,'ripplechannel_new_toadd.list'), delimiter = '\n', dtype = str, comments = '#')

pbmag_wt = []
pbmag_ko = []

KOmice = ['B2613', 'B2618', 'B2627', 'B2628', 'B3805', 'B3813', 'B4701', 'B4704', 'B4709']

for r,s in enumerate(datasets):
    print(s)
    name = s.split('-')[0]
    path = os.path.join(data_directory, s)
    
    if name in KOmice:
        isWT = 0
    else: isWT = 1 
    
    data = ntm.load_session(path, 'neurosuite')
    data.load_neurosuite_xml(path)
        
    lfp = nap.load_eeg(path + '/' + s + '.eeg', channel = int(ripplechannels[r]), n_channels = 32, frequency = 1250)
        
    file = os.path.join(path, s +'.sws.evt')
    if os.path.exists(file):
        tmp = np.genfromtxt(file)[:,0]
        tmp = tmp.reshape(len(tmp)//2,2)/1000
        sws_ep = nap.IntervalSet(start = tmp[:,0], end = tmp[:,1], time_units = 's')

        
    spikes = data.spikes
    epochs = data.epochs
    
    #fishing out wake and sleep epochs
    sleep_ep = epochs['sleep']
    wake_ep = epochs['wake']
        
#%% 
         
    bin_size = 0.01 #s
    smoothing_window = 0.02

    rates = spikes.count(bin_size, sws_ep)
       
    total2 = rates.as_dataframe().rolling(window = 100 ,win_type='gaussian',
                                          center=True,min_periods=1, 
                                          axis = 0).mean(std= int(smoothing_window/bin_size))
    
    total2 = total2.sum(axis =1)
    total2 = nap.Tsd(total2)
    idx = total2.threshold(np.percentile(total2.values,90),'above')
            
    pb_ep = idx.time_support
    
    pb_ep = nap.IntervalSet(start = pb_ep['start'], end = pb_ep['end'])
    # pb_ep = pb_ep.drop_short_intervals(bin_size)
    pb_ep = pb_ep.merge_close_intervals(bin_size*2)
    pb_ep = pb_ep.drop_short_intervals(bin_size*3)
    pb_ep = pb_ep.drop_long_intervals(bin_size*10)
   
    # sys.exit() 
   
    pb_ep = sws_ep.intersect(pb_ep)
    
    pb_max = []
    pb_tsd = []
    for s, e in pb_ep.values:
        tmp = total2.as_series().loc[s:e]
        pb_tsd.append(tmp.idxmax())
        pb_max.append(tmp.max())
    
    pb_max = np.array(pb_max)
    pb_tsd = np.array(pb_tsd)
    
    if isWT == 1:
        pbmag_wt.append(np.mean(pb_max))
    else:
        pbmag_ko.append(np.mean(pb_max))
        print(np.mean(pb_max))
        
    
#%% Save for neuroscope
  

    # start = pb_ep.as_units('ms')['start'].values
    # ends = pb_ep.as_units('ms')['end'].values

    # datatowrite = np.vstack((start,ends)).T.flatten()

    # n = len(pb_ep)

    # texttowrite = np.vstack(((np.repeat(np.array(['PyRip start 1']), n)), 
    #                           (np.repeat(np.array(['PyRip stop 1']), n))
    #                           )).T.flatten()

    # evt_file = path + '/' + name + '.evt.py.rip'
    # f = open(evt_file, 'w')
    # for t, n in zip(datatowrite, texttowrite):
    #     f.writelines("{:1.6f}".format(t) + "\t" + n + "\n")
    # f.close()        
    
#%% Organize population burst amplitudes

wt = np.array(['WT' for x in range(len(pbmag_wt))])
ko = np.array(['KO' for x in range(len(pbmag_ko))])
genotype = np.hstack([wt, ko])

allpeaks = []
allpeaks.extend(pbmag_wt)
allpeaks.extend(pbmag_ko)

summ = pd.DataFrame(data = [allpeaks, genotype], index = ['freq', 'genotype']).T

plt.figure()
plt.title('Population Burst Amplitude')
sns.set_style('white')
palette = ['royalblue', 'indianred']
ax = sns.violinplot( x = summ['genotype'], y=summ['freq'].astype(float) , data = summ, dodge=False,
                    palette = palette,cut = 2,
                    scale="width", inner=None)
ax.tick_params(bottom=True, left=True)
xlim = ax.get_xlim()
ylim = ax.get_ylim()
for violin in ax.collections:
    x0, y0, width, height = violin.get_paths()[0].get_extents().bounds
    violin.set_clip_path(plt.Rectangle((x0, y0), width / 2, height, transform=ax.transData))
sns.boxplot(x = summ['genotype'], y=summ['freq'].astype(float) , data = summ, saturation=1, showfliers=False,
            width=0.3, boxprops={'zorder': 3, 'facecolor': 'none'}, ax=ax)
old_len_collections = len(ax.collections)
sns.stripplot(x = summ['genotype'], y = summ['freq'].astype(float), data = summ, color = 'k', dodge=False, ax=ax)
# sns.swarmplot(x = durdf['genotype'], y = durdf['dur'].astype(float), data = durdf, color = 'k', dodge=False, ax=ax)
for dots in ax.collections[old_len_collections:]:
    dots.set_offsets(dots.get_offsets())
ax.set_xlim(xlim)
ax.set_ylim(ylim)
plt.ylabel('Population rate')
ax.set_box_aspect(1)

t, p = mannwhitneyu(pbmag_wt, pbmag_ko)
