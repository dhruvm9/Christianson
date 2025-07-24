#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 29 15:03:07 2025

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

# data_directory = '/media/dhruv/Expansion/Processed'
# data_directory = '/media/dhruv/Expansion/Processed/CA3'
data_directory = '/media/dhruv/Expansion/Processed/LinearTrack'
# datasets = np.genfromtxt(os.path.join(data_directory,'dataset_DM.list'), delimiter = '\n', dtype = str, comments = '#')
datasets = np.genfromtxt(os.path.join(data_directory,'dataset_LTreplay.list'), delimiter = '\n', dtype = str, comments = '#')
# ripplechannels = np.genfromtxt(os.path.join(data_directory,'ripplechannel.list'), delimiter = '\n', dtype = str, comments = '#')
ripplechannels = np.genfromtxt(os.path.join(data_directory,'ripplechannel_LTreplay.list'), delimiter = '\n', dtype = str, comments = '#')

fs = 1250

PSD_rip_wt = pd.DataFrame()
PSD_rip_ko = pd.DataFrame()

peakfreq_wt_pre = []
peakfreq_wt_post = []

peakfreq_ko_pre = []
peakfreq_ko_post = []

KOmice = ['B2613', 'B2618', 'B2627', 'B2628', 'B3805', 'B3813', 'B4701', 'B4704', 'B4709']

for r,s in enumerate(datasets):
    print(s)
    name = s.split('-')[0]
    path = os.path.join(data_directory, s)
    
    data = ntm.load_session(path, 'neurosuite')
    data.load_neurosuite_xml(path)
    epochs = data.epochs
    
    if name in KOmice:
        isWT = 0
    else: isWT = 1 
    
    lfp = nap.load_eeg(path + '/' + s + '.eeg', channel = int(ripplechannels[r]), n_channels = 32, frequency = fs)
    
    # file = os.path.join(path, s +'.sws.evt')
    # sws_ep = data.read_neuroscope_intervals(name = 'SWS', path2file = file)
    
      
    file = os.path.join(path, s +'.evt.py.rip')
    rip_ep = data.read_neuroscope_intervals(name = 'rip', path2file = file)
    
    # file = os.path.join(path, s +'.evt.py.wpb')
    # if os.path.isfile(file):
    #         rip_ep = data.read_neuroscope_intervals(name = 'wpb', path2file = file)
    # else: 
    #     continue
    
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
    
    rip_ep_pre = rip_ep.intersect(epochs['sleep'][0])
    rip_ep_post = rip_ep.intersect(epochs['sleep'][1])
    
    lfp_pre = lfp.restrict(nap.IntervalSet(rip_ep_pre))
    lfp_post = lfp.restrict(nap.IntervalSet(rip_ep_post))
    
    lfp_z_pre = scipy.stats.zscore(lfp_pre)
    lfp_z_post = scipy.stats.zscore(lfp_post)
    
#%% Power Spectral Density 
          
    freqs_pre, P_xx_pre = scipy.signal.welch(lfp_z_pre, fs = fs)
    freqs_post, P_xx_post = scipy.signal.welch(lfp_z_post, fs = fs)
    
    ix2_pre = np.where((freqs_pre>=100) & (freqs_pre <= 200))
    ix2_post = np.where((freqs_pre>=100) & (freqs_pre <= 200))
       
    peakfreq_pre = freqs_pre[ix2_pre][np.argmax(P_xx_pre[ix2_pre])]
    peakfreq_post = freqs_post[ix2_post][np.argmax(P_xx_post[ix2_post])]
    
    ix_pre = np.where((freqs_pre >= 10) & (freqs_pre <= 200))
    ix_post = np.where((freqs_post >= 10) & (freqs_post <= 200))
    
    # plt.figure()
    # plt.title(s)
    # plt.semilogx(freqs_pre[ix_pre], 10*np.log10(P_xx_pre[ix_pre]), 'o-', color = 'royalblue', label = 'pre')
    # plt.semilogx(freqs_post[ix_post], 10*np.log10(P_xx_post[ix_post]), 'o-', color = 'indianred', label = 'post')
    # plt.legend(loc = 'upper right')
    # plt.xlabel('Freq (Hz)')
    # plt.xticks([10, 100, 200])
    # plt.ylabel('Power (dB/Hz)')
    # plt.yticks([-20, -30])
    # plt.grid(True)
    # plt.gca().set_box_aspect(1)
       
           
    if isWT == 1:
         peakfreq_wt_pre.append(peakfreq_pre)
         peakfreq_wt_post.append(peakfreq_post)
                
    else: 
        peakfreq_ko_pre.append(peakfreq_pre)
        peakfreq_ko_post.append(peakfreq_post)
        
#%% 

plt.figure()
plt.subplot(121)
plt.title('WT')
plt.scatter(peakfreq_wt_pre, peakfreq_wt_post)
plt.gca().axline((min(min(peakfreq_wt_pre), min(peakfreq_wt_post)), min(min(peakfreq_wt_pre), min(peakfreq_wt_post))), slope=1, color = 'silver', linestyle = '--')
plt.xlabel('Pre-sleep peak SWR frequency (Hz)')
plt.ylabel('Post-sleep peak SWR frequency (Hz)')
plt.axis('square')

plt.subplot(122)
plt.title('KO')
plt.scatter(peakfreq_ko_pre, peakfreq_ko_post)
plt.gca().axline((min(min(peakfreq_ko_pre), min(peakfreq_ko_post)), min(min(peakfreq_ko_pre), min(peakfreq_ko_post))), slope=1, color = 'silver', linestyle = '--')
plt.xlabel('Pre-sleep peak SWR frequency (Hz)')
plt.ylabel('Post-sleep peak SWR frequency (Hz)')
plt.axis('square')

        
#%% Average spectrum 

# ix = np.where((freqs >= 10) & (freqs <= 200))

# ##Wake 
# plt.figure()
# # plt.title('During SWRs')
# plt.semilogx(freqs[ix], 10*np.log10(PSD_rip_wt.iloc[ix].mean(axis=1)), 'o-', label = 'WT', color = 'royalblue')
# # plt.semilogx(freqs[ix], 10*np.log10(PSD_rip_wt.iloc[ix]), 'o-', label = 'WT', color = 'royalblue')
# # err = 10*np.log10(PSD_rip_wt.iloc[ix].sem(axis=1))
# # plt.fill_between(freqs[ix],
# #                   10*np.log10(PSD_rip_wt.iloc[ix].mean(axis=1))-err, 
# #                   10*np.log10(PSD_rip_wt.iloc[ix].mean(axis=1))+err, color = 'royalblue', alpha = 0.2)

# plt.semilogx(freqs[ix], 10*np.log10(PSD_rip_ko.iloc[ix].mean(axis=1)), 'o-', label = 'KO', color = 'indianred')
# # plt.semilogx(freqs[ix], 10*np.log10(PSD_rip_ko.iloc[ix]), 'o-', label = 'KO', color = 'indianred')
# # err = 10*np.log10(PSD_rip_ko.iloc[ix].sem(axis=1))
# # plt.fill_between(freqs[ix],
# #                   10*np.log10(PSD_rip_ko.iloc[ix].mean(axis=1))-err, 
# #                   10*np.log10(PSD_rip_ko.iloc[ix].mean(axis=1))+err, color = 'indianred', alpha = 0.2)
# plt.xlabel('Freq (Hz)')
# plt.xticks([10, 100, 200])
# plt.ylabel('Power (dB/Hz)')
# plt.yticks([-20, -30])
# # plt.legend(loc = 'upper right')
# plt.grid(True)
# plt.gca().set_box_aspect(1)

#%% Organize peak frequency data and plot

# wt = np.array(['WT' for x in range(len(peakfreq_wt))])
# ko = np.array(['KO' for x in range(len(peakfreq_ko))])

# genotype = np.hstack([wt, ko])

# allpeaks = []
# allpeaks.extend(peakfreq_wt)
# allpeaks.extend(peakfreq_ko)

# summ = pd.DataFrame(data = [allpeaks, genotype], index = ['freq', 'genotype']).T

# plt.figure()
# # plt.title('Peak Frequency in Ripple Band')
# sns.set_style('white')
# palette = ['royalblue', 'indianred']
# ax = sns.violinplot( x = summ['genotype'], y=summ['freq'].astype(float) , data = summ, dodge=False,
#                     palette = palette,cut = 2,
#                     scale="width", inner=None)
# ax.tick_params(bottom=True, left=True)
# xlim = ax.get_xlim()
# ylim = ax.get_ylim()
# for violin in ax.collections:
#     x0, y0, width, height = violin.get_paths()[0].get_extents().bounds
#     violin.set_clip_path(plt.Rectangle((x0, y0), width / 2, height, transform=ax.transData))
# sns.boxplot(x = summ['genotype'], y=summ['freq'].astype(float) , data = summ, saturation=1, showfliers=False,
#             width=0.3, boxprops={'zorder': 3, 'facecolor': 'none'}, ax=ax)
# old_len_collections = len(ax.collections)
# sns.stripplot(x = summ['genotype'], y = summ['freq'].astype(float), data = summ, color = 'k', dodge=False, ax=ax)
# # sns.swarmplot(x = durdf['genotype'], y = durdf['dur'].astype(float), data = durdf, color = 'k', dodge=False, ax=ax)
# for dots in ax.collections[old_len_collections:]:
#     dots.set_offsets(dots.get_offsets())
# ax.set_xlim(xlim)
# ax.set_ylim(ylim)
# plt.ylabel('Frequency (Hz)')
# plt.yticks([100, 160])
# ax.set_box_aspect(1)

# t, p = mannwhitneyu(peakfreq_wt, peakfreq_ko)