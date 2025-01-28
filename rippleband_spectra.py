#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 13:01:57 2023

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
# data_directory = '/media/dhruv/Expansion/Processed/CA3'
datasets = np.genfromtxt(os.path.join(data_directory,'dataset_DM.list'), delimiter = '\n', dtype = str, comments = '#')
ripplechannels = np.genfromtxt(os.path.join(data_directory,'ripplechannel.list'), delimiter = '\n', dtype = str, comments = '#')

fs = 1250

PSD_rip_wt = pd.DataFrame()
PSD_rip_ko = pd.DataFrame()

peakfreq_wt = []
peakfreq_ko = []

for r,s in enumerate(datasets):
    print(s)
    name = s.split('-')[0]
    path = os.path.join(data_directory, s)
    
    data = ntm.load_session(path, 'neurosuite')
    data.load_neurosuite_xml(path)
    epochs = data.epochs
    
    if name == 'B2613' or name == 'B2618' or name == 'B2627' or name == 'B2628' or name == 'B3805' or name == 'B3813':
        isWT = 0
    else: isWT = 1 
    
    lfp = nap.load_eeg(path + '/' + s + '.eeg', channel = int(ripplechannels[r]), n_channels = 32, frequency = fs)
    
    # file = os.path.join(path, s +'.sws.evt')
    # sws_ep = data.read_neuroscope_intervals(name = 'SWS', path2file = file)
    
    # file = os.path.join(path, s +'.rem.evt')
    # rem_ep = data.read_neuroscope_intervals(name = 'REM', path2file = file)
    
    file = os.path.join(path, s +'.evt.py.rip')
    rip_ep = data.read_neuroscope_intervals(name = 'rip', path2file = file)
    
    # with open(os.path.join(path, 'riptsd.pickle'), 'rb') as pickle_file:
    #     rip_tsd = pickle.load(pickle_file)
  
    
    # file = os.path.join(path, s +'.evt.py3sd.rip')
    # rip_ep = data.read_neuroscope_intervals(name = 'py3sd', path2file = file)
    
    # with open(os.path.join(path, 'riptsd_3sd.pickle'), 'rb') as pickle_file:
    #      rip_tsd = pickle.load(pickle_file)
    
             
    # file = os.path.join(path, s +'.evt.py5sd.rip')
    # rip_ep = data.read_neuroscope_intervals(name = 'py5sd', path2file = file)
    
    # with open(os.path.join(path, 'riptsd_5sd.pickle'), 'rb') as pickle_file:
    #     rip_tsd = pickle.load(pickle_file)
    
#%% Restrict LFP
    
    # twin = 0.2
    # ripple_events = nap.IntervalSet(start = rip_tsd.index.values - twin, end = rip_tsd.index.values + twin)
    
    # lfp_rip = lfp.restrict(nap.IntervalSet(ripple_events))
    lfp_rip = lfp.restrict(nap.IntervalSet(rip_ep))
    lfp_z = scipy.stats.zscore(lfp_rip)
    
#%% Power Spectral Density 
          
    freqs, P_xx = scipy.signal.welch(lfp_z, fs = fs)
    
    ix2 = np.where((freqs>=100) & (freqs <= 200))
    peakfreq = freqs[ix2][np.argmax(P_xx[ix2])]
           
    if isWT == 1:
        PSD_rip_wt = pd.concat([PSD_rip_wt, pd.Series(P_xx)], axis = 1)
        peakfreq_wt.append(peakfreq)
        
    else: 
        PSD_rip_ko = pd.concat([PSD_rip_ko, pd.Series(P_xx)], axis = 1)
        peakfreq_ko.append(peakfreq)
        
#%% Average spectrum 

ix = np.where((freqs >= 10) & (freqs <= 200))

##Wake 
plt.figure()
# plt.title('During SWRs')
plt.semilogx(freqs[ix], 10*np.log10(PSD_rip_wt.iloc[ix].mean(axis=1)), 'o-', label = 'WT', color = 'royalblue')
# err = 10*np.log10(PSD_rip_wt.iloc[ix].sem(axis=1))
# plt.fill_between(freqs[ix],
#                   10*np.log10(PSD_rip_wt.iloc[ix].mean(axis=1))-err, 
#                   10*np.log10(PSD_rip_wt.iloc[ix].mean(axis=1))+err, color = 'royalblue', alpha = 0.2)

plt.semilogx(freqs[ix], 10*np.log10(PSD_rip_ko.iloc[ix].mean(axis=1)), 'o-', label = 'KO', color = 'indianred')
# err = 10*np.log10(PSD_rip_ko.iloc[ix].sem(axis=1))
# plt.fill_between(freqs[ix],
#                   10*np.log10(PSD_rip_ko.iloc[ix].mean(axis=1))-err, 
#                   10*np.log10(PSD_rip_ko.iloc[ix].mean(axis=1))+err, color = 'indianred', alpha = 0.2)
plt.xlabel('Freq (Hz)')
plt.xticks([10, 100, 200])
plt.ylabel('Power (dB/Hz)')
plt.yticks([-20, -30])
plt.legend(loc = 'upper right')
plt.grid(True)
plt.gca().set_box_aspect(1)

#%% Organize peak frequency data and plot

wt = np.array(['WT' for x in range(len(peakfreq_wt))])
ko = np.array(['KO' for x in range(len(peakfreq_ko))])

genotype = np.hstack([wt, ko])

allpeaks = []
allpeaks.extend(peakfreq_wt)
allpeaks.extend(peakfreq_ko)

summ = pd.DataFrame(data = [allpeaks, genotype], index = ['freq', 'genotype']).T

plt.figure()
# plt.title('Peak Frequency in Ripple Band')
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
plt.ylabel('Frequency (Hz)')
plt.yticks([100, 160])
ax.set_box_aspect(1)

t, p = mannwhitneyu(peakfreq_wt, peakfreq_ko)