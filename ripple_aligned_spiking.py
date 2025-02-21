#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 15:53:26 2025

@author: dhruv
"""

import numpy as np 
import pandas as pd 
import os, sys
import matplotlib.pyplot as plt 
import nwbmatic as ntm
import pynapple as nap
import pickle
import seaborn as sns 
from scipy.stats import mannwhitneyu

#%% 

data_directory = '/media/dhruv/Expansion/Processed'

datasets = np.genfromtxt(os.path.join(data_directory,'dataset_DM.list'), delimiter = '\n', dtype = str, comments = '#')
ripplechannels = np.genfromtxt(os.path.join(data_directory,'ripplechannel.list'), delimiter = '\n', dtype = str, comments = '#')

fsample = 1250

ras_pyr_wt = pd.DataFrame()
ras_fs_wt = pd.DataFrame()

ras_pyr_ko = pd.DataFrame()
ras_fs_ko = pd.DataFrame()

min_pyr_wt = []
max_pyr_wt = []
dur_pyr_wt = [] 
min_pyr_ko = []
max_pyr_ko = []
dur_pyr_ko = [] 

min_fs_wt = []
max_fs_wt = []
dur_fs_wt = [] 
min_fs_ko = []
max_fs_ko = []
dur_fs_ko = [] 


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
    
    lfp = nap.load_eeg(path + '/' + s + '.eeg', channel = int(ripplechannels[r]), n_channels = 32, frequency = fsample)
    
    file = os.path.join(path, s +'.sws.evt')
    sws_ep = data.read_neuroscope_intervals(name = 'SWS', path2file = file)
    
    file = os.path.join(path, s +'.rem.evt')
    rem_ep = data.read_neuroscope_intervals(name = 'REM', path2file = file)
    
    file = os.path.join(path, s +'.evt.py.rip')
    rip_ep = data.read_neuroscope_intervals(name = 'rip', path2file = file)
    
    with open(os.path.join(path, 'riptsd.pickle'), 'rb') as pickle_file:
        rip_tsd = pickle.load(pickle_file)
    
#%% 

    sp2 = np.load(os.path.join(path, 'spikedata_0.55.npz'), allow_pickle = True)
    time_support = nap.IntervalSet(sp2['start'], sp2['end'])
    tsd = nap.Tsd(t=sp2['t'], d=sp2['index'], time_support = time_support)
    spikes = tsd.to_tsgroup()
    spikes.set_info(group = sp2['group'], location = sp2['location'], celltype = sp2['celltype'], tr2pk = sp2['tr2pk'])
    
    spikes_by_celltype = spikes.getby_category('celltype')
    
    if 'pyr' in spikes._metadata['celltype'].values:
        pyr = spikes_by_celltype['pyr']
    else: fs = [] 
        
    if 'fs' in spikes._metadata['celltype'].values:
        fs = spikes_by_celltype['fs']
    else: fs = [] 
            
#%% 

## ((counts) / (binsize*number of events) / firing rate) 

    binsize = 0.005
    extent = (-0.25, 0.25)

    if len(pyr) > 5 and len(fs) > 0:    

        rip = nap.Ts(rip_tsd.index.values)
               
        pyrcounts = {}
        pyr_ras = pd.DataFrame()
        for i in pyr:
            pyr_peth = nap.compute_perievent(pyr[i], rip, minmax = extent)
            pyrcounts[i] = (np.sum(pyr_peth.count(binsize), 1) * binsize) / (pyr.restrict(nap.IntervalSet(sws_ep))._metadata['rate'][i] * len(rip_tsd))
            pyr_ras = pd.concat([pyr_ras, pyrcounts[i].as_series()], axis=1)
        pyr_ras.columns = pyrcounts.keys()
            
        pix_pyr = np.where(pyr_ras.T.mean().index.values > 0)[0][0]
        minp_pyr = pyr_ras.T.mean()[pix_pyr:].idxmin()
        minixp_pyr = np.where(pyr_ras.T.mean().index.values == minp_pyr)[0][0]
        
        # wp = np.diff(pyr_ras.T.mean()[minixp_pyr:])
        # w1p = pd.Series(data = wp, index = pyr_ras.T.mean()[minixp_pyr:].index.values[0:-1])
        # w2p = w1p.rolling(window=8,win_type='gaussian',center=True,min_periods=1).mean(std=2)
        # maxp_pyr = w2p.idxmax()
        
        maxp_pyr = pyr_ras.T.mean()[minixp_pyr:].idxmax()
        durp_pyr = maxp_pyr - minp_pyr
            
### FS 
        
        fscounts = {}
        fs_ras = pd.DataFrame()
        for i in fs:
            fs_peth = nap.compute_perievent(fs[i], rip, minmax = extent)
            fscounts[i] = (np.sum(fs_peth.count(binsize), 1) * binsize)  / (fs.restrict(nap.IntervalSet(sws_ep))._metadata['rate'][i] * len(rip_tsd))
            fs_ras = pd.concat([fs_ras, fscounts[i].as_series()], axis=1)
        fs_ras.columns = fscounts.keys()
        
        pix_fs = np.where(fs_ras.T.mean().index.values > 0)[0][0]
        minp_fs = fs_ras.T.mean()[pix_fs:].idxmin()
        minixp_fs = np.where(fs_ras.T.mean().index.values == minp_fs)[0][0]
        
        # wf = np.diff(fs_ras.T.mean()[minixp_fs:])
        # w1f = pd.Series(data = wf, index = fs_ras.T.mean()[minixp_fs:].index.values[0:-1])
        # w2f = w1f.rolling(window=8,win_type='gaussian',center=True,min_periods=1).mean(std=2)
        # maxp_fs = w2f.idxmax()
        
        maxp_fs = fs_ras.T.mean()[minixp_fs:].idxmax()
        durp_fs = maxp_fs - minp_fs

### Genotype

        if isWT == 1: 
            ras_pyr_wt = pd.concat([ras_pyr_wt, pyr_ras.T.mean()], axis=1)
            ras_fs_wt = pd.concat([ras_fs_wt, fs_ras.T.mean()], axis=1)
            min_pyr_wt.append(minp_pyr)
            max_pyr_wt.append(maxp_pyr)
            dur_pyr_wt.append(durp_pyr)
            min_fs_wt.append(minp_fs)
            max_fs_wt.append(maxp_fs)
            dur_fs_wt.append(durp_fs)
        else:
            ras_pyr_ko = pd.concat([ras_pyr_ko, pyr_ras.T.mean()], axis=1)
            ras_fs_ko = pd.concat([ras_fs_ko, fs_ras.T.mean()], axis=1)
            min_pyr_ko.append(minp_pyr)
            max_pyr_ko.append(maxp_pyr)
            dur_pyr_ko.append(durp_pyr)
            min_fs_ko.append(minp_fs)
            max_fs_ko.append(maxp_fs)
            dur_fs_ko.append(durp_fs)
                      
              
        

#%% 
            
        # plt.figure()
        # plt.suptitle(s)
        # plt.subplot(221)
        # plt.title('PYR')
        # plt.imshow(pyr_ras.T,extent = [extent[0], extent[1], len(pyr)+1 , 1], origin = 'lower', aspect = 'auto', cmap = 'inferno')
        # plt.colorbar()
        # plt.xlabel('Time from SWR (s)')
        # plt.ylabel('Neuron')
        # plt.gca().set_box_aspect(1)
        # plt.subplot(223)
        # plt.plot(pyr_ras.T.mean(), color = 'royalblue')
        # plt.axvline(0, color = 'silver', linestyle = '--')
        # plt.gca().set_box_aspect(1)
        # plt.xlabel('Time from SWR (s)')
        # plt.ylabel('Norm rate')
        # plt.gca().set_box_aspect(1)
        # plt.subplot(222)
        # plt.title('FS')
        # plt.imshow(fs_ras.T,extent = [-0.25 , 0.25, len(fs)+1 , 1], origin = 'lower', aspect = 'auto', cmap = 'inferno')
        # plt.colorbar()
        # plt.xlabel('Time from SWR (s)')
        # plt.ylabel('Neuron')
        # plt.gca().set_box_aspect(1)
        # plt.subplot(224)
        # plt.plot(fs_ras.T.mean(), color = 'indianred')
        # plt.axvline(0, color = 'silver', linestyle = '--')
        # plt.gca().set_box_aspect(1)
        # plt.xlabel('Time from SWR (s)')
        # plt.ylabel('Norm rate')
        # plt.gca().set_box_aspect(1)
                   
#%% 

# plt.figure()
# plt.suptitle('PYR')
# plt.subplot(121)
# plt.title('WT')
# plt.plot(ras_pyr_wt, color = 'silver')
# plt.plot(ras_pyr_wt.T.mean(), color = 'royalblue')
# plt.xlabel('Time from SWR (s)')
# plt.ylabel('Norm rate')
# plt.gca().set_box_aspect(1)
# plt.subplot(122)
# plt.title('KO')
# plt.plot(ras_pyr_ko, color = 'silver')
# plt.plot(ras_pyr_ko.T.mean(), color = 'indianred')
# plt.xlabel('Time from SWR (s)')
# plt.ylabel('Norm rate')
# plt.gca().set_box_aspect(1)

# plt.figure()
# plt.suptitle('FS')
# plt.subplot(121)
# plt.title('WT')
# plt.plot(ras_fs_wt, color = 'silver')
# plt.plot(ras_fs_wt.T.mean(), color = 'royalblue')
# plt.xlabel('Time from SWR (s)')
# plt.ylabel('Norm rate')
# plt.gca().set_box_aspect(1)
# plt.subplot(122)
# plt.title('KO')
# plt.plot(ras_fs_ko, color = 'silver')
# plt.plot(ras_fs_ko.T.mean(), color = 'indianred')
# plt.xlabel('Time from SWR (s)')
# plt.ylabel('Norm rate')
# plt.gca().set_box_aspect(1)


### Average Plot 
plt.figure()
plt.subplot(121)
plt.title('PYR')
plt.plot(ras_pyr_wt.T.mean(), color = 'royalblue', label = 'WT')
err = ras_pyr_wt.T.sem()
plt.fill_between(np.array(ras_pyr_wt.index.values).astype(np.float64), (ras_pyr_wt.T.mean()-err).values, (ras_pyr_wt.T.mean()+err).values, color = 'lightsteelblue', alpha = 0.2)
plt.plot(ras_pyr_ko.T.mean(), color = 'indianred', label = 'KO')
err = ras_pyr_ko.T.sem()
plt.fill_between(np.array(ras_pyr_ko.index.values).astype(np.float64), (ras_pyr_ko.T.mean()-err).values, (ras_pyr_ko.T.mean()+err).values, color = 'lightcoral', alpha = 0.2)
plt.axvline(0, color = 'silver', linestyle = '--')
plt.xlabel('Time from SWR (s)')
plt.ylabel('Norm rate')
plt.legend(loc = 'upper right')
plt.gca().set_box_aspect(1)
plt.subplot(122)
plt.title('FS')
plt.plot(ras_fs_wt.T.mean(), color = 'royalblue', label = 'WT')
err = ras_fs_wt.T.sem()
plt.fill_between(np.array(ras_fs_wt.index.values).astype(np.float64), (ras_fs_wt.T.mean()-err).values, (ras_fs_wt.T.mean()+err).values, color = 'lightsteelblue', alpha = 0.2)
plt.plot(ras_fs_ko.T.mean(), color = 'indianred', label = 'KO')
err = ras_fs_ko.T.sem()
plt.fill_between(np.array(ras_fs_ko.index.values).astype(np.float64), (ras_fs_ko.T.mean()-err).values, (ras_fs_ko.T.mean()+err).values, color = 'lightcoral', alpha = 0.2)
plt.axvline(0, color = 'silver', linestyle = '--')
plt.xlabel('Time from SWR (s)')
plt.ylabel('Norm rate')
plt.legend(loc = 'upper right')
plt.gca().set_box_aspect(1)

#%% Organize min, max, dur 

###Min

wt1 = np.array(['pyr_WT' for x in range(len(min_pyr_wt))])
ko1 = np.array(['pyr_KO' for x in range(len(min_pyr_ko))])
wt2 = np.array(['fs_WT' for x in range(len(min_fs_wt))])
ko2 = np.array(['fs_KO' for x in range(len(min_fs_ko))])

genotype = np.hstack([wt1, ko1, wt2, ko2])

sinfos = []
sinfos.extend(min_pyr_wt)
sinfos.extend(min_pyr_ko)
sinfos.extend(min_fs_wt)
sinfos.extend(min_fs_ko)

allinfos = pd.DataFrame(data = [sinfos, genotype], index = ['corr', 'type']).T

###Max

wt1 = np.array(['pyr_WT' for x in range(len(max_pyr_wt))])
ko1 = np.array(['pyr_KO' for x in range(len(max_pyr_ko))])
wt2 = np.array(['fs_WT' for x in range(len(max_fs_wt))])
ko2 = np.array(['fs_KO' for x in range(len(max_fs_ko))])

genotype = np.hstack([wt1, ko1, wt2, ko2])

sinfos = []
sinfos.extend(max_pyr_wt)
sinfos.extend(max_pyr_ko)
sinfos.extend(max_fs_wt)
sinfos.extend(max_fs_ko)

allinfos2 = pd.DataFrame(data = [sinfos, genotype], index = ['corr', 'type']).T

# ###Dur

wt1 = np.array(['pyr_WT' for x in range(len(dur_pyr_wt))])
ko1 = np.array(['pyr_KO' for x in range(len(dur_pyr_ko))])
wt2 = np.array(['fs_WT' for x in range(len(dur_fs_wt))])
ko2 = np.array(['fs_KO' for x in range(len(dur_fs_ko))])

genotype = np.hstack([wt1, ko1, wt2, ko2])

sinfos = []
sinfos.extend(dur_pyr_wt)
sinfos.extend(dur_pyr_ko)
sinfos.extend(dur_fs_wt)
sinfos.extend(dur_fs_ko)

allinfos3 = pd.DataFrame(data = [sinfos, genotype], index = ['corr', 'type']).T

#%%

tmin_pyr, pmin_pyr = mannwhitneyu(min_pyr_wt, min_pyr_ko)
tmin_fs, pmin_fs = mannwhitneyu(min_fs_wt, min_fs_ko)

tmax_pyr, pmax_pyr = mannwhitneyu(max_pyr_wt, max_pyr_ko)
tmax_fs, pmax_fs = mannwhitneyu(max_fs_wt, max_fs_ko)

td_pyr, pd_pyr = mannwhitneyu(dur_pyr_wt, dur_pyr_ko)
td_fs, pd_fs = mannwhitneyu(dur_fs_wt, dur_fs_ko)


#%% 

plt.figure()
plt.title('Time of min rate')
sns.set_style('white')
palette = ['royalblue', 'lightsteelblue', 'indianred', 'lightcoral']
ax = sns.violinplot( x = allinfos['type'], y=allinfos['corr'].astype(float) , data = allinfos, dodge=False,
                    palette = palette,cut = 2,
                    scale="width", inner=None)
ax.tick_params(bottom=True, left=True)
xlim = ax.get_xlim()
ylim = ax.get_ylim()
for violin in ax.collections:
    x0, y0, width, height = violin.get_paths()[0].get_extents().bounds
    violin.set_clip_path(plt.Rectangle((x0, y0), width / 2, height, transform=ax.transData))
sns.boxplot(x = allinfos['type'], y=allinfos['corr'] , data = allinfos, saturation=1, showfliers=False,
            width=0.3, boxprops={'zorder': 3, 'facecolor': 'none'}, ax=ax)
sns.stripplot(x = allinfos['type'], y = allinfos['corr'].astype(float), data = allinfos, color = 'k', dodge=False, ax=ax, alpha = 0.2)
plt.ylabel('Time (s)')
ax.set_box_aspect(1)

plt.figure()
plt.title('Time of onset')
sns.set_style('white')
palette = ['royalblue', 'lightsteelblue', 'indianred', 'lightcoral']
ax = sns.violinplot( x = allinfos2['type'], y=allinfos2['corr'].astype(float) , data = allinfos2, dodge=False,
                    palette = palette,cut = 2,
                    scale="width", inner=None)
ax.tick_params(bottom=True, left=True)
xlim = ax.get_xlim()
ylim = ax.get_ylim()
for violin in ax.collections:
    x0, y0, width, height = violin.get_paths()[0].get_extents().bounds
    violin.set_clip_path(plt.Rectangle((x0, y0), width / 2, height, transform=ax.transData))
sns.boxplot(x = allinfos2['type'], y=allinfos2['corr'] , data = allinfos2, saturation=1, showfliers=False,
            width=0.3, boxprops={'zorder': 3, 'facecolor': 'none'}, ax=ax)
sns.stripplot(x = allinfos2['type'], y = allinfos2['corr'].astype(float), data = allinfos2, color = 'k', dodge=False, ax=ax, alpha = 0.2)
plt.ylabel('Time (s)')
ax.set_box_aspect(1)

plt.figure()
plt.title('Duration of recovery')
sns.set_style('white')
palette = ['royalblue', 'lightsteelblue', 'indianred', 'lightcoral']
ax = sns.violinplot( x = allinfos3['type'], y=allinfos3['corr'].astype(float) , data = allinfos3, dodge=False,
                    palette = palette,cut = 2,
                    scale="width", inner=None)
ax.tick_params(bottom=True, left=True)
xlim = ax.get_xlim()
ylim = ax.get_ylim()
for violin in ax.collections:
    x0, y0, width, height = violin.get_paths()[0].get_extents().bounds
    violin.set_clip_path(plt.Rectangle((x0, y0), width / 2, height, transform=ax.transData))
sns.boxplot(x = allinfos3['type'], y=allinfos3['corr'] , data = allinfos3, saturation=1, showfliers=False,
            width=0.3, boxprops={'zorder': 3, 'facecolor': 'none'}, ax=ax)
sns.stripplot(x = allinfos3['type'], y = allinfos3['corr'].astype(float), data = allinfos3, color = 'k', dodge=False, ax=ax, alpha = 0.2)
plt.ylabel('Time (s)')
ax.set_box_aspect(1)

