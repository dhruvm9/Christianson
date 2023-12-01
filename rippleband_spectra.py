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
datasets = np.genfromtxt(os.path.join(data_directory,'dataset_DM.list'), delimiter = '\n', dtype = str, comments = '#')
ripplechannels = np.genfromtxt(os.path.join(data_directory,'ripplechannel.list'), delimiter = '\n', dtype = str, comments = '#')

fs = 1250

PSD_rip_wt = pd.DataFrame()
PSD_rip_ko = pd.DataFrame()

for r,s in enumerate(datasets):
    print(s)
    name = s.split('-')[0]
    path = os.path.join(data_directory, s)
    
    data = ntm.load_session(path, 'neurosuite')
    data.load_neurosuite_xml(path)
    epochs = data.epochs
    
    if name == 'B2613' or name == 'B2618':
        isWT = 0
    else: isWT = 1 
    
    lfp = nap.load_eeg(path + '/' + s + '.eeg', channel = int(ripplechannels[r]), n_channels = 32, frequency = fs)
    
    file = os.path.join(path, s +'.sws.evt')
    sws_ep = data.read_neuroscope_intervals(name = 'SWS', path2file = file)
    
    file = os.path.join(path, s +'.rem.evt')
    rem_ep = data.read_neuroscope_intervals(name = 'REM', path2file = file)
    
    file = os.path.join(path, s +'.evt.py.rip')
    rip_ep = data.read_neuroscope_intervals(name = 'rip', path2file = file)
    
    with open(os.path.join(path, 'riptsd.pickle'), 'rb') as pickle_file:
        rip_tsd = pickle.load(pickle_file)
    
#%% Restrict LFP
    
    twin = 0.2
    ripple_events = nap.IntervalSet(start = rip_tsd.index.values - twin, end = rip_tsd.index.values + twin)
    
    # lfp_rip = lfp.restrict(nap.IntervalSet(ripple_events))
    lfp_rip = lfp.restrict(nap.IntervalSet(rip_ep))
    lfp_z = scipy.stats.zscore(lfp_rip)
    
#%% Power Spectral Density 
          
    freqs, P_xx = scipy.signal.welch(lfp_z, fs = fs)
           
    if isWT == 1:
        PSD_rip_wt = pd.concat([PSD_rip_wt, pd.Series(P_xx)], axis = 1)
        
    else: 
        PSD_rip_ko = pd.concat([PSD_rip_ko, pd.Series(P_xx)], axis = 1)
        
#%% Average spectrum 

ix = np.where((freqs>=100) & (freqs <= 300))

##Wake 
plt.figure()
plt.title('During SWRs')
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
plt.ylabel('Power spectral density (dB/Hz)')
plt.legend(loc = 'upper right')
plt.grid(True)
