#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 15:11:57 2024

@author: dhruv
"""

import numpy as np 
import pandas as pd
import nwbmatic as ntm
import pynapple as nap 
import pynacollada as pyna
import pickle
import scipy.io
import os, sys
import seaborn as sns
import matplotlib.pyplot as plt 
from scipy.stats import mannwhitneyu
from scipy.signal import hilbert

#%% 

# data_directory = '/media/dhruv/Expansion/Processed'
data_directory = '/media/adrien/Expansion/Processed'
datasets = np.genfromtxt(os.path.join(data_directory,'dataset_DM.list'), delimiter = '\n', dtype = str, comments = '#')
ripplechannels = np.genfromtxt(os.path.join(data_directory,'ripplechannel.list'), delimiter = '\n', dtype = str, comments = '#')

fs = 1250

glt_w_wt = []
glt_n_wt = []
glt_r_wt = []

ght_w_wt = []
ght_n_wt = []
ght_r_wt = []

td_w_wt = []
td_n_wt = []
td_r_wt = []

glt_w_ko = []
glt_n_ko = []
glt_r_ko = []

ght_w_ko = []
ght_n_ko = []
ght_r_ko = []

td_w_ko = []
td_n_ko = []
td_r_ko = []


for r,s in enumerate(datasets):
    print(s)
    
    name = s.split('-')[0]
    path = os.path.join(data_directory, s)
    
    data = ntm.load_session(path, 'neurosuite')
    data.load_neurosuite_xml(path)
    epochs = data.epochs
    position = data.position
    
    if name == 'B2613' or name == 'B2618' or name == 'B2627' or name == 'B2628':
        isWT = 0
    else: isWT = 1 
    
    lfp = nap.load_eeg(path + '/' + s + '.eeg', channel = int(ripplechannels[r]), n_channels = 32, frequency = fs)
    
    file = os.path.join(path, s +'.sws.evt')
    sws_ep = data.read_neuroscope_intervals(name = 'SWS', path2file = file)
        
    file = os.path.join(path, s +'.rem.evt')
    rem_ep = data.read_neuroscope_intervals(name = 'REM', path2file = file)
    
#%% Moving epoch 

    speedbinsize = np.diff(position.index.values)[0]
    
    time_bins = np.arange(position.index[0], position.index[-1] + speedbinsize, speedbinsize)
    index = np.digitize(position.index.values, time_bins)
    tmp = position.as_dataframe().groupby(index).mean()
    tmp.index = time_bins[np.unique(index)-1]+(speedbinsize)/2
    distance = np.sqrt(np.power(np.diff(tmp['x']), 2) + np.power(np.diff(tmp['z']), 2)) * 100 #in cm
    speed = pd.Series(index = tmp.index.values[0:-1]+ speedbinsize/2, data = distance/speedbinsize) # in cm/s
    speed2 = speed.rolling(window = 25, win_type='gaussian', center=True, min_periods=1).mean(std=10) #Smooth over 200ms 
    speed2 = nap.Tsd(speed2)
    moving_ep = nap.IntervalSet(speed2.threshold(2).time_support) #Epochs in which speed is > 2 cm/s
    
 
#%% Filter LFP and do power ratios

    lfp_filt_delta = pyna.eeg_processing.bandpass_filter(lfp, 1, 4, 1250)
    lfp_filt_theta = pyna.eeg_processing.bandpass_filter(lfp, 6, 9, 1250)
    lfp_filt_gammalow = pyna.eeg_processing.bandpass_filter(lfp, 30, 50, 1250)
    lfp_filt_gammahigh = pyna.eeg_processing.bandpass_filter(lfp, 70, 90, 1250)

    delta_power = nap.Tsd(lfp_filt_delta.index.values, np.abs(hilbert(lfp_filt_delta.values)))
    theta_power = nap.Tsd(lfp_filt_theta.index.values, np.abs(hilbert(lfp_filt_theta.values)))
    gammalow_power = nap.Tsd(lfp_filt_gammalow.index.values, np.abs(hilbert(lfp_filt_gammalow.values)))
    gammahigh_power = nap.Tsd(lfp_filt_gammalow.index.values, np.abs(hilbert(lfp_filt_gammahigh.values))) 

    d_wake = delta_power.restrict(moving_ep)   
    d_nrem = delta_power.restrict(sws_ep)   
    d_rem = delta_power.restrict(rem_ep)   
    
    t_wake = theta_power.restrict(moving_ep)   
    t_nrem = theta_power.restrict(sws_ep)   
    t_rem = theta_power.restrict(rem_ep)   
    
    gl_wake = gammalow_power.restrict(moving_ep)   
    gl_nrem = gammalow_power.restrict(sws_ep)   
    gl_rem = gammalow_power.restrict(rem_ep)   
    
    gh_wake = gammahigh_power.restrict(moving_ep)   
    gh_nrem = gammahigh_power.restrict(sws_ep)   
    gh_rem = gammahigh_power.restrict(rem_ep)   
    
#%% Compute ratios and sort by genotype 
    
    glt_wake = np.mean(gl_wake) / np.mean(t_wake)
    glt_nrem = np.mean(gl_nrem) / np.mean(t_nrem)
    glt_rem = np.mean(gl_rem) / np.mean(t_rem)
    
    ght_wake = np.mean(gh_wake) / np.mean(t_wake)
    ght_nrem = np.mean(gh_nrem) / np.mean(t_nrem)
    ght_rem = np.mean(gh_rem) / np.mean(t_rem)
    
    td_wake = np.mean(t_wake) / np.mean(d_wake)
    td_nrem = np.mean(t_nrem) / np.mean(d_nrem)
    td_rem = np.mean(t_rem) / np.mean(d_rem)
    
    if isWT == 1:
        glt_w_wt.append(glt_wake)
        glt_n_wt.append(glt_nrem)
        glt_r_wt.append(glt_rem)
        
        ght_w_wt.append(ght_wake)
        ght_n_wt.append(ght_nrem)
        ght_r_wt.append(ght_rem)
        
        td_w_wt.append(td_wake)
        td_n_wt.append(td_nrem)
        td_r_wt.append(td_rem)
        
    else: 
        glt_w_ko.append(glt_wake)
        glt_n_ko.append(glt_nrem)
        glt_r_ko.append(glt_rem)
        
        ght_w_ko.append(ght_wake)
        ght_n_ko.append(ght_nrem)
        ght_r_ko.append(ght_rem)
        
        td_w_ko.append(td_wake)
        td_n_ko.append(td_nrem)
        td_r_ko.append(td_rem)
        
#%% Organize data to plot 

e = np.array(['LowGamma/Theta (WT)' for x in range(len(glt_w_wt))])
e2 = np.array(['LowGamma/Theta (KO)' for x in range(len(glt_w_ko))])

e3 = np.array(['HighGamma/Theta (WT)' for x in range(len(glt_w_wt))])
e4 = np.array(['HighGamma/Theta (KO)' for x in range(len(glt_w_ko))])

# e5 = np.array(['Theta/Delta (WT)' for x in range(len(glt_w_wt))])
# e6 = np.array(['Theta/Delta (KO)' for x in range(len(glt_w_ko))])

types = np.hstack([e, e2, e3, e4])

wakerates = []
wakerates.extend(glt_w_wt)
wakerates.extend(glt_w_ko)
wakerates.extend(ght_w_wt)
wakerates.extend(ght_w_ko)
# wakerates.extend(td_w_wt)
# wakerates.extend(td_w_ko)

wakedf = pd.DataFrame(data = [wakerates, types], index = ['rate', 'type']).T

##NREM 
e = np.array(['LowGamma/Theta (WT)' for x in range(len(glt_n_wt))])
e2 = np.array(['LowGamma/Theta (KO)' for x in range(len(glt_n_ko))])

e3 = np.array(['HighGamma/Theta (WT)' for x in range(len(glt_n_wt))])
e4 = np.array(['HighGamma/Theta (KO)' for x in range(len(glt_n_ko))])

# e5 = np.array(['Theta/Delta (WT)' for x in range(len(glt_n_wt))])
# e6 = np.array(['Theta/Delta (KO)' for x in range(len(glt_n_ko))])

types = np.hstack([e, e2, e3, e4])

nremrates = []
nremrates.extend(glt_n_wt)
nremrates.extend(glt_n_ko)
nremrates.extend(ght_n_wt)
nremrates.extend(ght_n_ko)
# nremrates.extend(td_n_wt)
# nremrates.extend(td_n_ko)

nremdf = pd.DataFrame(data = [nremrates, types], index = ['rate', 'type']).T

##REM 
e = np.array(['LowGamma/Theta (WT)' for x in range(len(glt_r_wt))])
e2 = np.array(['LowGamma/Theta (KO)' for x in range(len(glt_r_ko))])

e3 = np.array(['HighGamma/Theta (WT)' for x in range(len(glt_r_wt))])
e4 = np.array(['HighGamma/Theta (KO)' for x in range(len(glt_r_ko))])

e5 = np.array(['Theta/Delta (WT)' for x in range(len(glt_r_wt))])
e6 = np.array(['Theta/Delta (KO)' for x in range(len(glt_r_ko))])

types = np.hstack([e, e2, e3, e4])

remrates = []
remrates.extend(glt_r_wt)
remrates.extend(glt_r_ko)
remrates.extend(ght_r_wt)
remrates.extend(ght_r_ko)
# remrates.extend(td_r_wt)
# remrates.extend(td_r_ko)

remdf = pd.DataFrame(data = [remrates, types], index = ['rate', 'type']).T

#%% Plotting 

plt.figure()
plt.suptitle('LFP Power ratios')
plt.subplot(131)
plt.title('Wake')
sns.set_style('white')
palette = ['royalblue', 'lightsteelblue', 'indianred', 'lightcoral', 'darkslategray','cadetblue']
ax = sns.violinplot( x = wakedf['type'], y=wakedf['rate'].astype(float) , data = wakedf, dodge=False,
                    palette = palette,cut = 2,
                    scale="width", inner=None)
ax.tick_params(bottom=True, left=True)
xlim = ax.get_xlim()
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
plt.ylabel('Ratio')
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
plt.ylabel('Ratio')
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
plt.ylabel('Ratio')
ax.set_box_aspect(1)

#%% 

t,p = mannwhitneyu(glt_w_wt, glt_w_ko)
t,p = mannwhitneyu(glt_n_wt, glt_n_ko)
t,p = mannwhitneyu(glt_r_wt, glt_r_ko)

t,p = mannwhitneyu(ght_w_wt, ght_w_ko)
t,p = mannwhitneyu(ght_n_wt, ght_n_ko)
t,p = mannwhitneyu(ght_r_wt, ght_r_ko)

