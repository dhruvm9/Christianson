#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 12:04:33 2024

@author: dhruv
"""

import numpy as np 
import nwbmatic as ntm
import pynapple as nap 
import pynacollada as pyna
import os, sys
import matplotlib.pyplot as plt 
import pickle
from scipy.signal import filtfilt, hilbert

#%% 

data_directory = '/media/dhruv/Expansion/Processed/CA3'
datasets = np.genfromtxt(os.path.join(data_directory,'dataset_DM.list'), delimiter = '\n', dtype = str, comments = '#')

ripplechannels = np.genfromtxt(os.path.join(data_directory,'ripplechannel.list'), delimiter = '\n', dtype = str, comments = '#')

for r,s in enumerate(datasets):
    print(s)
    name = s.split('/')[-1]
    path = os.path.join(data_directory, s)
    
    # data = ntm.load_session(path, 'neurosuite')
    # data.load_neurosuite_xml(path)
       
    # rip_ep = data.load_nwb_intervals('sleep_ripples')
    # rip_tsd = data.load_nwb_timeseries('sleep_ripples')
    
    lfp = nap.load_eeg(path + '/' + s + '.eeg', channel = int(ripplechannels[r]), n_channels = 32, frequency = 1250)
        
    file = os.path.join(path, name +'.sws.evt')
    if os.path.exists(file):
        tmp = np.genfromtxt(file)[:,0]
        tmp = tmp.reshape(len(tmp)//2,2)/1000
        sws_ep = nap.IntervalSet(start = tmp[:,0], end = tmp[:,1], time_units = 's')
   
        
    file = os.path.join(path, name +'.rem.evt')
    if os.path.exists(file):
        tmp = np.genfromtxt(file)[:,0]
        tmp = tmp.reshape(len(tmp)//2,2)/1000
        rem_ep = nap.IntervalSet(start = tmp[:,0], end = tmp[:,1], time_units = 's')
    
#%% 

    ex_ep = nap.IntervalSet(start = 5738, end = 5741)
       
    lfpnrem = lfp.restrict(sws_ep)
            
    signal = pyna.eeg_processing.bandpass_filter_zerophase(lfpnrem, 0.5, 50, 1250)
    # signal = pyna.eeg_processing.bandpass_filter_zerophase(lfpnrem, 0.2, 5, 1250, 2)
    
    
    plt.figure(figsize=(15,5))
    plt.subplot(211)
    plt.plot(lfpnrem.restrict(ex_ep))
    plt.subplot(212)
    plt.plot(signal.restrict(ex_ep))
    plt.xlabel("Time (s)")
    
#%% 
    
    power = nap.Tsd(signal.index.values, np.abs(hilbert(signal.values)))
    
    plt.figure(figsize=(15,5))
    plt.subplot(311)
    plt.plot(lfpnrem.restrict(ex_ep))
    plt.subplot(312)
    plt.plot(signal.restrict(ex_ep))
    plt.subplot(313)
    plt.plot(power.restrict(ex_ep))
    plt.xlabel("Time (s)")
    plt.tight_layout()
    
#%% 

    thres_rms = 2.5*np.sqrt(np.mean(power.values**2))
    
    nSS2 = power.threshold(thres_rms, method = 'above')
    
    plt.figure(figsize=(15,5))
    plt.subplot(311)
    plt.plot(lfpnrem.restrict(ex_ep).as_units('s'))
    plt.subplot(312)
    plt.plot(signal.restrict(ex_ep).as_units('s'))
    plt.subplot(313)
    plt.plot(power.restrict(ex_ep).as_units('s'))
    plt.plot(nSS2.restrict(ex_ep).as_units('s'))
            
    plt.axhline(thres_rms, color = 'k')
    plt.xlabel("Time (s)")
    plt.tight_layout()
    
#%% 

    minLen = 40 # ms
    maxLen = 100 # ms
    
    sw_ep = nSS2.time_support
    sw_ep = sw_ep.drop_short_intervals(minLen, time_units = 'ms')
    sw_ep = sw_ep.drop_long_intervals(maxLen, time_units = 'ms')
    
    minInterRippleInterval = 20 # ms
    sw_ep = sw_ep.merge_close_intervals(minInterRippleInterval, time_units = 'ms')
    
#%% 

    plt.figure(figsize=(15,5))
    plt.subplot(311)
    plt.plot(lfpnrem.restrict(ex_ep).as_units('s'))
    plt.subplot(312)
    plt.plot(signal.restrict(ex_ep).as_units('s'))
    plt.subplot(313)
    plt.plot(nSS2.restrict(ex_ep).as_units('s'))
    plt.plot(nSS2.restrict(sw_ep.intersect(ex_ep)).as_units('s'), '.')
    plt.axhline(thres_rms)
    plt.xlabel("Time (s)")
    plt.tight_layout()