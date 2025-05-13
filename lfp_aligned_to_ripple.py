#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 11:32:27 2023

@author: dhruv
"""

import numpy as np 
import pandas as pd 
import os, sys
import matplotlib.pyplot as plt 
import nwbmatic as ntm
import pynapple as nap
import pickle

#%% 

# data_directory = '/media/dhruv/Expansion/Processed'
# data_directory = '/media/dhruv/Expansion/Processed/CA3'
data_directory = '/media/dhruv/Expansion/Processed/LinearTrack'
datasets = np.genfromtxt(os.path.join(data_directory,'dataset_DM.list'), delimiter = '\n', dtype = str, comments = '#')
ripplechannels = np.genfromtxt(os.path.join(data_directory,'ripplechannel.list'), delimiter = '\n', dtype = str, comments = '#')

fs = 1250

sessions_WT = pd.DataFrame()
sessions_KO = pd.DataFrame()

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
    
    file = os.path.join(path, s +'.sws.evt')
    sws_ep = data.read_neuroscope_intervals(name = 'SWS', path2file = file)
    
    # file = os.path.join(path, s +'.rem.evt')
    # rem_ep = data.read_neuroscope_intervals(name = 'REM', path2file = file)
    
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
    
#%% Plotting 

plt.figure()
# plt.suptitle('Average ripple across sessions')
plt.subplot(121)
plt.title('WT')
plt.plot(sessions_WT, color = 'silver')
plt.plot(sessions_WT.mean(axis = 1), color = 'royalblue', linewidth = 2)
plt.xlabel('Time from SWR (s)')
plt.ylim([-1700, 3500])
plt.gca().set_box_aspect(1)
# plt.yticks([])
plt.subplot(122)
plt.title('KO')
plt.plot(sessions_KO, color = 'silver')
plt.plot(sessions_KO.mean(axis = 1), color = 'indianred', linewidth = 2)
plt.xlabel('Time from SWR (s)')
plt.ylim([-1700, 3500])
# plt.yticks([])
plt.gca().set_box_aspect(1)