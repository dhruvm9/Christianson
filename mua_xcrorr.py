#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 08:42:44 2025

@author: dhruv
"""

import numpy as np 
import pandas as pd 
import os, sys
import matplotlib.pyplot as plt 
import nwbmatic as ntm
import pynapple as nap
import pickle
from scipy.stats import mannwhitneyu, pearsonr, norm, zscore
import scipy
import warnings
import seaborn as sns 

#%% 

def fisher_z(r):
    return 0.5 * np.log((1 + r) / (1 - r))

def compare_correlations(r1, n1, r2, n2):
    z1 = fisher_z(r1)
    z2 = fisher_z(r2)
    
    # Standard error
    se = np.sqrt(1/(n1 - 3) + 1/(n2 - 3))
    
    # Z-score for the difference
    z = (z1 - z2) / se
    
    # Two-tailed p-value
    p_value = 2 * (1 - norm.cdf(abs(z)))
    
    return z, p_value

#%% 

warnings.filterwarnings("ignore")

data_directory = '/media/dhruv/Expansion/Processed'
# data_directory = '/media/dhruv/Expansion/Processed/LinearTrack'

datasets = np.genfromtxt(os.path.join(data_directory,'dataset_DM.list'), delimiter = '\n', dtype = str, comments = '#')
ripplechannels = np.genfromtxt(os.path.join(data_directory,'ripplechannel.list'), delimiter = '\n', dtype = str, comments = '#')

fsamp = 1250

sessions_WT = pd.DataFrame()
sessions_KO = pd.DataFrame()

minwt = []
maxwt = []
minko = []
maxko = []

min2wt = []
min2ko = []
max2wt = []
max2ko = []
durwt = []
durko = []

riprates_wt = []
riprates_ko = []

peakfreq_wt = []
peakfreq_ko = []

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
    
    lfp = nap.load_eeg(path + '/' + s + '.eeg', channel = int(ripplechannels[r]), n_channels = 32, frequency = fsamp)
        
    file = os.path.join(path, s +'.sws.evt')
    sws_ep = data.read_neuroscope_intervals(name = 'SWS', path2file = file)
    
        
    file = os.path.join(path, s +'.evt.py.rip')
    rip_ep = data.read_neuroscope_intervals(name = 'rip', path2file = file)
    
    with open(os.path.join(path, 'riptsd.pickle'), 'rb') as pickle_file:
        rip_tsd = pickle.load(pickle_file)

#%% Load spikes 
        
    # sp2 = np.load(os.path.join(path, 'spikedata_0.55.npz'), allow_pickle = True)
    # time_support = nap.IntervalSet(sp2['start'], sp2['end'])
    # tsd = nap.Tsd(t=sp2['t'], d=sp2['index'], time_support = time_support)
    # spikes = tsd.to_tsgroup()
    # spikes.set_info(group = sp2['group'], location = sp2['location'], celltype = sp2['celltype'], tr2pk = sp2['tr2pk'])
    
    spikes = nap.load_file(os.path.join(path, 'spikedata_0.55.npz'))
        
#%% Create MUA

    rip = nap.Ts(rip_tsd.index.values)
    
    mua = []
    
    if len(spikes) > 5:
    
        for n in spikes.keys():            
            mua.extend(spikes[n].index.values)
        
        mua = nap.TsGroup({0: nap.Ts(t = np.sort(mua))})
            
        
    #%% Cross correlogram
    
        xc_mua = nap.compute_eventcorrelogram(mua, rip, binsize = 0.005, windowsize = 0.2 , ep = nap.IntervalSet(sws_ep), norm = True)
        xc_mua = xc_mua.rolling(window=8, win_type='gaussian',center=True,min_periods=1).mean(std = 2)
        minpr = xc_mua[0:].min()
        maxpr = xc_mua.max()
                
        min2pr = xc_mua[0:].idxmin()
        max2pr = xc_mua[0.03:].idxmax()
        durpr = max2pr - min2pr
        
        if isWT == 1:
            sessions_WT = pd.concat([sessions_WT, xc_mua], axis = 1)
            
        else: sessions_KO = pd.concat([sessions_KO, xc_mua], axis = 1)
        
    #%% Peak ripple frequency 
    
        lfp_rip = lfp.restrict(nap.IntervalSet(rip_ep))
        lfp_z = zscore(lfp_rip)
              
        freqs, P_xx = scipy.signal.welch(lfp_z, fs = fsamp)
        
        ix2 = np.where((freqs>=100) & (freqs <= 200))
        peakfreq = freqs[ix2][np.argmax(P_xx[ix2])]
        
        riprate = len(rip_ep)/sws_ep.tot_length('s')
             
        if isWT == 1:
            peakfreq_wt.append(peakfreq) 
            minwt.append(minpr.values[0])
            maxwt.append(maxpr.values[0])
            min2wt.append(min2pr.values[0])
            max2wt.append(max2pr.values[0])
            durwt.append(durpr.values[0])
            riprates_wt.append(riprate)
                    
        else: 
            peakfreq_ko.append(peakfreq) 
            minko.append(minpr.values[0])
            maxko.append(maxpr.values[0])
            min2ko.append(min2pr.values[0])
            max2ko.append(max2pr.values[0])
            durko.append(durpr.values[0])
            riprates_ko.append(riprate)
    
#%% Plotting 

colnames_WT = np.arange(0, len(sessions_WT.columns))
colnames_KO = np.arange(0, len(sessions_KO.columns))

sessions_WT.columns = colnames_WT
sessions_KO.columns = colnames_KO

colors_wt = plt.cm.PuBu((np.array(peakfreq_wt) - np.min(peakfreq_wt)) / (np.max(peakfreq_wt) - np.min(peakfreq_wt)))
colors_ko = plt.cm.OrRd((np.array(peakfreq_ko) - np.min(peakfreq_ko)) / (np.max(peakfreq_ko) - np.min(peakfreq_ko)))

plt.figure()
plt.suptitle('Population activity around ripple')
plt.subplot(121)
plt.title('WT')
for i in sessions_WT.columns:
#     plt.plot(sessions_WT[i], color=colors_wt[sessions_WT.columns[i]])
    plt.plot(sessions_WT[i], color='silver')
plt.xlabel('Time from SWR (s)')
plt.ylabel('Norm. rate')
plt.gca().set_box_aspect(1)
plt.subplot(122)
plt.title('KO')
for i in sessions_KO.columns:
    # plt.plot(sessions_KO[i], color=colors_ko[sessions_KO.columns[i]])
    plt.plot(sessions_KO[i], color='silver')
plt.xlabel('Time from SWR (s)')
plt.ylabel('Norm. rate')
plt.gca().set_box_aspect(1)


plt.figure()
plt.tight_layout()
plt.title('Ripple onset Cross-correlogram')       
plt.xlabel('Time from SWR (s)')
plt.ylabel('norm. rate')
plt.plot(sessions_WT.mean(axis=1), color = 'royalblue', label = 'WT')
err = sessions_WT.sem(axis=1)
plt.fill_between(sessions_WT.index.values, sessions_WT.mean(axis=1) - err, sessions_WT.mean(axis=1) + err, alpha = 0.2, color = 'lightsteelblue') 
plt.plot(sessions_KO.mean(axis=1), color = 'indianred', label = 'KO')
err = sessions_KO.sem(axis=1)
plt.fill_between(sessions_KO.index.values, sessions_KO.mean(axis=1) - err, sessions_KO.mean(axis=1) + err, alpha = 0.2, color = 'lightcoral') 
plt.legend(loc = 'upper right')
plt.gca().set_box_aspect(1)

    
#%% Organize min, max, dur 

###Min

wt1 = np.array(['WT' for x in range(len(min2wt))])
ko1 = np.array(['KO' for x in range(len(min2ko))])

genotype = np.hstack([wt1, ko1])

sinfos = []
sinfos.extend(min2wt)
sinfos.extend(min2ko)

allinfos = pd.DataFrame(data = [sinfos, genotype], index = ['corr', 'type']).T

###Max

wt1 = np.array(['WT' for x in range(len(max2wt))])
ko1 = np.array(['KO' for x in range(len(max2ko))])

genotype = np.hstack([wt1, ko1])

sinfos = []
sinfos.extend(max2wt)
sinfos.extend(max2ko)

allinfos2 = pd.DataFrame(data = [sinfos, genotype], index = ['corr', 'type']).T

# ###Dur

wt1 = np.array(['WT' for x in range(len(durwt))])
ko1 = np.array(['KO' for x in range(len(durko))])

genotype = np.hstack([wt1, ko1])

sinfos = []
sinfos.extend(durwt)
sinfos.extend(durko)

allinfos3 = pd.DataFrame(data = [sinfos, genotype], index = ['corr', 'type']).T

###Rate at min

wt1 = np.array(['WT' for x in range(len(minwt))])
ko1 = np.array(['KO' for x in range(len(minko))])

genotype = np.hstack([wt1, ko1])

sinfos = []
sinfos.extend(minwt)
sinfos.extend(minko)

allinfos4 = pd.DataFrame(data = [sinfos, genotype], index = ['corr', 'type']).T

###Max norm rate

wt1 = np.array(['WT' for x in range(len(maxwt))])
ko1 = np.array(['KO' for x in range(len(maxko))])

genotype = np.hstack([wt1, ko1])

sinfos = []
sinfos.extend(maxwt)
sinfos.extend(maxko)

allinfos5 = pd.DataFrame(data = [sinfos, genotype], index = ['corr', 'type']).T

#%% 

tmin2, pmin2 = mannwhitneyu(min2wt, min2ko)
tmax2, pmax2 = mannwhitneyu(max2wt, max2ko)
tdur, pdur = mannwhitneyu(durwt, durko)

tminrate, pminrate = mannwhitneyu(minwt, minko)
tmaxrate, pmaxrate = mannwhitneyu(maxwt, maxko)


#%% 

# plt.figure()
# plt.title('Time of min rate')
# sns.set_style('white')
# palette = ['royalblue','indianred']
# ax = sns.violinplot( x = allinfos['type'], y=allinfos['corr'].astype(float) , data = allinfos, dodge=False,
#                     palette = palette,cut = 2,
#                     scale="width", inner=None)
# ax.tick_params(bottom=True, left=True)
# xlim = ax.get_xlim()
# ylim = ax.get_ylim()
# for violin in ax.collections:
#     x0, y0, width, height = violin.get_paths()[0].get_extents().bounds
#     violin.set_clip_path(plt.Rectangle((x0, y0), width / 2, height, transform=ax.transData))
# sns.boxplot(x = allinfos['type'], y=allinfos['corr'] , data = allinfos, saturation=1, showfliers=False,
#             width=0.3, boxprops={'zorder': 3, 'facecolor': 'none'}, ax=ax)
# sns.stripplot(x = allinfos['type'], y = allinfos['corr'].astype(float), data = allinfos, color = 'k', dodge=False, ax=ax, alpha = 0.2)
# plt.ylabel('Time (s)')
# ax.set_box_aspect(1)

# plt.figure()
# plt.title('Time of onset')
# sns.set_style('white')
# palette = ['royalblue', 'indianred']
# ax = sns.violinplot( x = allinfos2['type'], y=allinfos2['corr'].astype(float) , data = allinfos2, dodge=False,
#                     palette = palette,cut = 2,
#                     scale="width", inner=None)
# ax.tick_params(bottom=True, left=True)
# xlim = ax.get_xlim()
# ylim = ax.get_ylim()
# for violin in ax.collections:
#     x0, y0, width, height = violin.get_paths()[0].get_extents().bounds
#     violin.set_clip_path(plt.Rectangle((x0, y0), width / 2, height, transform=ax.transData))
# sns.boxplot(x = allinfos2['type'], y=allinfos2['corr'] , data = allinfos2, saturation=1, showfliers=False,
#             width=0.3, boxprops={'zorder': 3, 'facecolor': 'none'}, ax=ax)
# sns.stripplot(x = allinfos2['type'], y = allinfos2['corr'].astype(float), data = allinfos2, color = 'k', dodge=False, ax=ax, alpha = 0.2)
# plt.ylabel('Time (s)')
# ax.set_box_aspect(1)

# plt.figure()
# plt.title('Duration of recovery')
# sns.set_style('white')
# palette = ['royalblue', 'indianred']
# ax = sns.violinplot( x = allinfos3['type'], y=allinfos3['corr'].astype(float) , data = allinfos3, dodge=False,
#                     palette = palette,cut = 2,
#                     scale="width", inner=None)
# ax.tick_params(bottom=True, left=True)
# xlim = ax.get_xlim()
# ylim = ax.get_ylim()
# for violin in ax.collections:
#     x0, y0, width, height = violin.get_paths()[0].get_extents().bounds
#     violin.set_clip_path(plt.Rectangle((x0, y0), width / 2, height, transform=ax.transData))
# sns.boxplot(x = allinfos3['type'], y=allinfos3['corr'] , data = allinfos3, saturation=1, showfliers=False,
#             width=0.3, boxprops={'zorder': 3, 'facecolor': 'none'}, ax=ax)
# sns.stripplot(x = allinfos3['type'], y = allinfos3['corr'].astype(float), data = allinfos3, color = 'k', dodge=False, ax=ax, alpha = 0.2)
# plt.ylabel('Time (s)')
# ax.set_box_aspect(1)

plt.figure()
plt.title('Rate at minima')
sns.set_style('white')
palette = ['royalblue', 'indianred']
ax = sns.violinplot( x = allinfos4['type'], y=allinfos4['corr'].astype(float) , data = allinfos3, dodge=False,
                    palette = palette,cut = 2,
                    scale="width", inner=None)
ax.tick_params(bottom=True, left=True)
xlim = ax.get_xlim()
ylim = ax.get_ylim()
for violin in ax.collections:
    x0, y0, width, height = violin.get_paths()[0].get_extents().bounds
    violin.set_clip_path(plt.Rectangle((x0, y0), width / 2, height, transform=ax.transData))
sns.boxplot(x = allinfos4['type'], y=allinfos4['corr'] , data = allinfos4, saturation=1, showfliers=False,
            width=0.3, boxprops={'zorder': 3, 'facecolor': 'none'}, ax=ax)
sns.stripplot(x = allinfos4['type'], y = allinfos4['corr'].astype(float), data = allinfos4, color = 'k', dodge=False, ax=ax, alpha = 0.2)
plt.ylabel('Norm rate')
ax.set_box_aspect(1)

plt.figure()
plt.title('Max rate')
sns.set_style('white')
palette = ['royalblue', 'indianred']
ax = sns.violinplot( x = allinfos5['type'], y=allinfos5['corr'].astype(float) , data = allinfos5, dodge=False,
                    palette = palette,cut = 2,
                    scale="width", inner=None)
ax.tick_params(bottom=True, left=True)
xlim = ax.get_xlim()
ylim = ax.get_ylim()
for violin in ax.collections:
    x0, y0, width, height = violin.get_paths()[0].get_extents().bounds
    violin.set_clip_path(plt.Rectangle((x0, y0), width / 2, height, transform=ax.transData))
sns.boxplot(x = allinfos5['type'], y=allinfos5['corr'] , data = allinfos5, saturation=1, showfliers=False,
            width=0.3, boxprops={'zorder': 3, 'facecolor': 'none'}, ax=ax)
sns.stripplot(x = allinfos5['type'], y = allinfos5['corr'].astype(float), data = allinfos5, color = 'k', dodge=False, ax=ax, alpha = 0.2)
plt.ylabel('Norm rate')
ax.set_box_aspect(1)

#%% 

r_wt, p_wt = pearsonr(peakfreq_wt, minwt)
r_ko, p_ko = pearsonr(peakfreq_ko, minko)

plt.figure()
plt.scatter(peakfreq_wt, minwt, color = 'royalblue', label = 'WT')
plt.scatter(peakfreq_ko, minko, color = 'indianred', label = 'KO')
plt.xlabel('SWR peak freq (Hz)')
plt.ylabel('Rate at minima of ripple AHP')
plt.legend(loc = 'upper right')
plt.gca().set_box_aspect(1)

# plt.figure()
# plt.scatter(riprates_wt, minwt, color = 'royalblue', label = 'WT')
# plt.scatter(riprates_ko, minko, color = 'indianred', label = 'KO')
# plt.xlabel('SWR occurrence rate (Hz)')
# plt.ylabel('Rate at minima of ripple AHP')
# plt.legend(loc = 'upper right')
# plt.gca().set_box_aspect(1)

a = []
a.extend(peakfreq_wt)
a.extend(peakfreq_ko)

b = []
b.extend(minwt)
b.extend(minko)

r_pooled, p_pooled = pearsonr(a,b)

z_12, p_12 = compare_correlations(r_wt, len(minwt), r_ko, len(minko))
z_13, p_13 = compare_correlations(r_wt, len(minwt), r_pooled, len(a))
z_23, p_23 = compare_correlations(r_ko, len(minko), r_pooled, len(a))

plt.figure()
plt.scatter(peakfreq_wt, maxwt, color = 'royalblue', label = 'WT')
plt.scatter(peakfreq_ko, maxko, color = 'indianred', label = 'KO')
plt.xlabel('SWR peak freq (Hz)')
plt.ylabel('Rate at maxima of ripple AHP')
plt.legend(loc = 'upper left')
plt.gca().set_box_aspect(1)