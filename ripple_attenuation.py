#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 12:07:13 2024

@author: dhruv
"""

import numpy as np 
import pandas as pd
import nwbmatic as ntm
import pynapple as nap 
import pynacollada as pyna
import scipy.io
import os, sys
import pickle
import seaborn as sns
import matplotlib.pyplot as plt 
from scipy.stats import mannwhitneyu
from scipy.signal import hilbert

#%% 

data_directory = '/media/dhruv/Expansion/Processed'
datasets = np.genfromtxt(os.path.join(data_directory,'dataset_DM.list'), delimiter = '\n', dtype = str, comments = '#')
ripplechannels = np.genfromtxt(os.path.join(data_directory,'ripplechannel.list'), delimiter = '\n', dtype = str, comments = '#')
refchannel = np.genfromtxt(os.path.join(data_directory,'rippleplus4.list'), delimiter = '\n', dtype = str, comments = '#')

fs = 1250

sess_att_wt = []
sess_att_ko = []

att_wt = []
att_ko = []

for r,s in enumerate(datasets):
    print(s)
    name = s.split('-')[0]
    path = os.path.join(data_directory, s)
    
    data = ntm.load_session(path, 'neurosuite')
    data.load_neurosuite_xml(path)
    epochs = data.epochs
    
    if name == 'B2613' or name == 'B2618' or name == 'B2627' or name == 'B2628':
        isWT = 0
    else: isWT = 1 
    
    lfp = nap.load_eeg(path + '/' + s + '.eeg', channel = int(ripplechannels[r]), n_channels = 32, frequency = fs)
    lfpref = nap.load_eeg(path + '/' + s + '.eeg', channel = int(refchannel[r]), n_channels = 32, frequency = fs)
    
    file = os.path.join(path, s +'.sws.evt')
    sws_ep = data.read_neuroscope_intervals(name = 'SWS', path2file = file)
    
    file = os.path.join(path, s +'.rem.evt')
    rem_ep = data.read_neuroscope_intervals(name = 'REM', path2file = file)
    
    file = os.path.join(path, s +'.evt.py.rip')
    rip_ep = data.read_neuroscope_intervals(name = 'rip', path2file = file)
       
            
#%% Filter LFP in ripple band, then restrict to ripple periods
    
    lfp_filt_rippleband = pyna.eeg_processing.bandpass_filter(lfp, 100, 200, fs)
    power_lfp = nap.Tsd(lfp_filt_rippleband.index.values, np.abs(hilbert(lfp_filt_rippleband.values)))
    # pls = power_lfp.as_series()
    # pls = pls.rolling(window = 8,win_type='gaussian',center=True,min_periods=1).mean(std=32)

    ref_filt_rippleband = pyna.eeg_processing.bandpass_filter(lfpref, 100, 200, fs)
    power_ref = nap.Tsd(ref_filt_rippleband.index.values, np.abs(hilbert(ref_filt_rippleband.values)))
    # prs = power_ref.as_series()
    # prs = pls.rolling(window = 8,win_type='gaussian',center=True,min_periods=1).mean(std=32)
    
    
    rip_lfp = power_lfp.restrict(nap.IntervalSet(rip_ep))
    ref_lfp = power_ref.restrict(nap.IntervalSet(rip_ep))

    # rip_lfp = nap.Tsd(pls).restrict(nap.IntervalSet(rip_ep))
    # ref_lfp = nap.Tsd(prs).restrict(nap.IntervalSet(rip_ep))
            

#%% Compute mean attenuation index for a session
    
    att_index = np.var(ref_lfp.values) / np.var(rip_lfp.values) #np.mean(ref_lfp.values/rip_lfp.values)
               
    evt_att = []
    
    for i in range(len(rip_ep)):
        # evt_att.append(np.mean(ref_lfp.restrict(rip_ep.loc[[i]]).values / rip_lfp.restrict(rip_ep.loc[[i]]).values))
        evt_att.append(np.var(ref_lfp.restrict(rip_ep.loc[[i]]).values) / np.var(rip_lfp.restrict(rip_ep.loc[[i]]).values))
     
             
    if isWT == 1: 
        sess_att_wt.append(att_index)
        att_wt.extend(evt_att)
    
    else: 
        sess_att_ko.append(att_index)
        att_ko.extend(evt_att)
    
#%% Sorting by genotype

wt = np.array(['WT' for x in range(len(sess_att_wt))])
ko = np.array(['KO' for x in range(len(sess_att_ko))])    

wt2 = np.array(['WT' for x in range(len(att_wt))])
ko2 = np.array(['KO' for x in range(len(att_ko))])    

genotype = np.hstack([wt, ko])
gt2 = np.hstack([wt2, ko2])

sess_att = []
sess_att.extend(sess_att_wt)
sess_att.extend(sess_att_ko)

evt_all = []
evt_all.extend(att_wt)
evt_all.extend(att_ko)

summ = pd.DataFrame(data = [sess_att, genotype], index = ['att', 'genotype']).T
s2 = pd.DataFrame(data = [evt_all, gt2], index = ['att', 'genotype']).T

#%% Plotting 

plt.figure()
plt.title('Ripple Power Across Channels')
sns.set_style('white')
palette = ['royalblue', 'indianred']
ax = sns.violinplot( x = summ['genotype'], y = summ['att'].astype(float) , data = summ, dodge = False,
                    palette = palette,cut = 2,
                    scale = "width", inner = None)
ax.tick_params(bottom=True, left=True)
xlim = ax.get_xlim()
ylim = ax.get_ylim()
for violin in ax.collections:
    x0, y0, width, height = violin.get_paths()[0].get_extents().bounds
    violin.set_clip_path(plt.Rectangle((x0, y0), width / 2, height, transform=ax.transData))
sns.boxplot(x = summ['genotype'], y = summ['att'] , data = summ, saturation = 1, showfliers = False,
            width = 0.3, boxprops = {'zorder': 3, 'facecolor': 'none'}, ax = ax)
old_len_collections = len(ax.collections)
sns.swarmplot(x = summ['genotype'], y=summ['att'], data = summ, color = 'k', dodge = False, ax = ax)
# sns.stripplot(x = wakedf['type'], y = wakedf['rate'].astype(float), data = wakedf, color = 'k', dodge=False, ax=ax)
for dots in ax.collections[old_len_collections:]:
    dots.set_offsets(dots.get_offsets())
ax.set_xlim(xlim)
ax.set_ylim(ylim)
plt.ylabel('Ripple Attenuation Index')
ax.set_box_aspect(1)

plt.figure()
plt.title('Ripple Power Across Channels')
sns.set_style('white')
palette = ['royalblue', 'indianred']
ax = sns.violinplot( x = s2['genotype'], y = s2['att'].astype(float) , data = s2, dodge = False,
                    palette = palette,cut = 2,
                    scale = "width", inner = None)
ax.tick_params(bottom=True, left=True)
xlim = ax.get_xlim()
ylim = ax.get_ylim()
for violin in ax.collections:
    x0, y0, width, height = violin.get_paths()[0].get_extents().bounds
    violin.set_clip_path(plt.Rectangle((x0, y0), width / 2, height, transform=ax.transData))
sns.boxplot(x = s2['genotype'], y = s2['att'] , data = s2, saturation = 1, showfliers = False,
            width = 0.3, boxprops = {'zorder': 3, 'facecolor': 'none'}, ax = ax)
old_len_collections = len(ax.collections)
sns.stripplot(x = s2['genotype'], y=s2['att'], data = s2, color = 'k', dodge = False, ax = ax, alpha = 0.2)
# sns.stripplot(x = wakedf['type'], y = wakedf['rate'].astype(float), data = wakedf, color = 'k', dodge=False, ax=ax)
for dots in ax.collections[old_len_collections:]:
    dots.set_offsets(dots.get_offsets())
ax.set_xlim(xlim)
ax.set_ylim(ylim)
plt.ylabel('Ripple Attenuation Index')
ax.set_box_aspect(1)



#%% Stats 

t, p = mannwhitneyu(sess_att_wt, sess_att_ko)
# t2, p2 = mannwhitneyu(att_wt, att_ko)
    
    