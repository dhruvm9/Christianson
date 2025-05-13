#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 15:58:56 2025

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

#%% 

data_directory = '/media/dhruv/Expansion/Processed'
# data_directory = '/media/dhruv/Expansion/Processed/CA3'
datasets = np.genfromtxt(os.path.join(data_directory,'dataset_DM.list'), delimiter = '\n', dtype = str, comments = '#')
ripplechannels = np.genfromtxt(os.path.join(data_directory,'ripplechannel.list'), delimiter = '\n', dtype = str, comments = '#')

fsamp = 1250

sessions_WT = pd.DataFrame()
sessions_KO = pd.DataFrame()

minwt = []
maxwt = []
minko = []
maxko = []

riprates_wt = []
riprates_ko = []

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
    
    lfp = nap.load_eeg(path + '/' + s + '.eeg', channel = int(ripplechannels[r]), n_channels = 32, frequency = fsamp)
    
    file = os.path.join(path, s +'.sws.evt')
    sws_ep = data.read_neuroscope_intervals(name = 'SWS', path2file = file)
    
    file = os.path.join(path, s +'.rem.evt')
    rem_ep = data.read_neuroscope_intervals(name = 'REM', path2file = file)
    
    file = os.path.join(path, s +'.evt.py.rip')
    rip_ep = data.read_neuroscope_intervals(name = 'rip', path2file = file)
    
    with open(os.path.join(path, 'riptsd.pickle'), 'rb') as pickle_file:
        rip_tsd = pickle.load(pickle_file)
        
    # file = os.path.join(path, s +'.evt.py3sd.rip')
    # rip_ep = data.read_neuroscope_intervals(name = 'py3sd', path2file = file)
    
    # with open(os.path.join(path, 'riptsd_3sd.pickle'), 'rb') as pickle_file:
    #     rip_tsd = pickle.load(pickle_file)
    
    # file = os.path.join(path, s +'.evt.py5sd.rip')
    # rip_ep = data.read_neuroscope_intervals(name = 'py5sd', path2file = file)
    
    # with open(os.path.join(path, 'riptsd_5sd.pickle'), 'rb') as pickle_file:
    #     rip_tsd = pickle.load(pickle_file)
    
        
#%% 

    rip = nap.Ts(rip_tsd.index.values)
   
    realigned = lfp.as_series().index[lfp.as_series().index.get_indexer(rip.index.values, method='nearest')]
          
    peth = nap.compute_perievent(lfp, nap.Ts(realigned.values), minmax = (-0.25, 0.25), time_unit = 's')
    
    peth_all = []
    for j in range(len(peth)):
        peth_all.append(peth[j].as_series())
    
    avgripple = pd.concat(peth_all, axis = 1, join = 'outer').dropna(how = 'all')
    
    if isWT == 1:
        sessions_WT = pd.concat([sessions_WT, avgripple.mean(axis=1)], axis = 1)
        
    else: sessions_KO = pd.concat([sessions_KO, avgripple.mean(axis=1)], axis = 1)
    
#%% Max and min post-ripple 

    minpr = avgripple.mean(axis=1)[0:].min()
    maxpr = avgripple.mean(axis=1)[0:].max()
    
    lfp_rip = lfp.restrict(nap.IntervalSet(rip_ep))
    lfp_z = zscore(lfp_rip)
          
    freqs, P_xx = scipy.signal.welch(lfp_z, fs = fsamp)
    
    ix2 = np.where((freqs>=100) & (freqs <= 200))
    peakfreq = freqs[ix2][np.argmax(P_xx[ix2])]
    
       
    riprate = len(rip_ep)/sws_ep.tot_length('s')
    
    if isWT == 1:
        minwt.append(minpr)
        maxwt.append(maxpr)
        peakfreq_wt.append(peakfreq) 
        riprates_wt.append(riprate)
        
    else: 
        minko.append(minpr)
        maxko.append(maxpr)
        peakfreq_ko.append(peakfreq) 
        riprates_ko.append(riprate)
        
    
#%% Plotting 

# plt.figure()
# # plt.suptitle('Average ripple across sessions')
# plt.subplot(121)
# plt.title('WT')
# plt.plot(sessions_WT, color = 'silver')
# plt.plot(sessions_WT.mean(axis = 1), color = 'royalblue', linewidth = 2)
# plt.xlabel('Time from SWR (s)')
# plt.ylim([-1700, 3500])
# plt.gca().set_box_aspect(1)
# # plt.yticks([])
# plt.subplot(122)
# plt.title('KO')
# plt.plot(sessions_KO, color = 'silver')
# plt.plot(sessions_KO.mean(axis = 1), color = 'indianred', linewidth = 2)
# plt.xlabel('Time from SWR (s)')
# plt.ylim([-1700, 3500])
# # plt.yticks([])
# plt.gca().set_box_aspect(1)

#%% 

colnames_WT = np.arange(0, len(sessions_WT.columns))
colnames_KO = np.arange(0, len(sessions_KO.columns))

sessions_WT.columns = colnames_WT
sessions_KO.columns = colnames_KO

colors_wt = plt.cm.winter((np.array(peakfreq_wt) - np.min(peakfreq_wt)) / (np.max(peakfreq_wt) - np.min(peakfreq_wt)))
colors_ko = plt.cm.autumn((np.array(peakfreq_ko) - np.min(peakfreq_ko)) / (np.max(peakfreq_ko) - np.min(peakfreq_ko)))

plt.figure()
plt.suptitle('Avg LFP around ripple')
plt.subplot(121)
plt.title('WT')
for i in sessions_WT.columns:
    plt.plot(sessions_WT[i], color=colors_wt[sessions_WT.columns[i]])
    plt.xlabel('Time from SWR (s)')
    plt.ylim([-1700, 3500])
    plt.gca().set_box_aspect(1)
    # plt.legend(loc = 'upper right')
plt.subplot(122)
plt.title('KO')
for i in sessions_KO.columns:
    plt.plot(sessions_KO[i], color=colors_ko[sessions_KO.columns[i]])
    plt.xlabel('Time from SWR (s)')
    plt.ylim([-1700, 3500])
    plt.gca().set_box_aspect(1)
    # plt.legend(loc = 'upper right')
    
    
# a = np.array([[0,1]])
# plt.figure(figsize=(9, 1.5))
# img = plt.imshow(a, cmap='autumn')
# plt.gca().set_visible(False)
# cax = plt.axes([0.1, 0.2, 0.8, 0.6])
# plt.colorbar(orientation="horizontal", cax=cax)
