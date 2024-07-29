#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 11:33:34 2024

@author: dhruv
"""

import numpy as np 
import nwbmatic as ntm
import pandas as pd
import pynapple as nap 
import pynacollada as pyna
import os, sys
import matplotlib.pyplot as plt 
import pickle
from scipy.signal import filtfilt

#%% 

data_directory = '/media/dhruv/Expansion/Processed'
datasets = np.genfromtxt(os.path.join(data_directory,'dataset_DM_rippletravel.list'), delimiter = '\n', dtype = str, comments = '#')
ripplechannels = pd.read_csv(os.path.join(data_directory,'ripplechannel_byshank.list'), dtype = int, header=None)

for r,s in enumerate(datasets):
    print(s)
    name = s.split('/')[-1]
    path = os.path.join(data_directory, s)
    
    data = ntm.load_session(path, 'neurosuite')
    data.load_neurosuite_xml(path)
    
        
    lfp = nap.load_eeg(path + '/' + s + '.eeg', channel = list(ripplechannels.loc[r].values), n_channels = 32, frequency = 1250)
    
    file = os.path.join(path, name +'.sws.evt')
    sws_ep = data.read_neuroscope_intervals(name = 'SWS', path2file = file)
    
    lfpnrem = lfp.restrict(sws_ep)
    
    rip_intervals = {}
    rip_maxes = {}
    
    for n, k in enumerate(lfpnrem.columns):
         
        signal = pyna.eeg_processing.bandpass_filter(lfpnrem[:,n], 100, 200, 1250)
        windowLength = 51
        squared_signal = np.square(signal.values)
        window = np.ones(windowLength)/windowLength
        nSS = filtfilt(window, 1, squared_signal, axis=0)
        nSS = (nSS - np.mean(nSS))/np.std(nSS)
        
        nSS = nap.Tsd(t = signal.index.values, d = nSS, time_support = signal.time_support)
        
        low_thres = 1
        high_thres = 10
        
        nSS2 = nSS.threshold(low_thres, method = 'above')
        nSS3 = nSS2.threshold(high_thres, method = 'below')
        
        minRipLen = 20 # ms
        maxRipLen = 200 # ms

        rip_ep = nSS3.time_support
        rip_ep = rip_ep.drop_short_intervals(minRipLen, time_units = 'ms')
        rip_ep = rip_ep.drop_long_intervals(maxRipLen, time_units = 'ms')
        
        minInterRippleInterval = 20 # ms
        rip_ep = rip_ep.merge_close_intervals(minInterRippleInterval, time_units = 'ms')
        
        rip_max = []
        rip_tsd = []
        for s, e in rip_ep.values:
            tmp = nSS.as_series().loc[s:e]     #Change if nSS is TsdFrame
            rip_tsd.append(tmp.idxmax())
            rip_max.append(tmp.max())
        
        rip_max = np.array(rip_max)
        rip_tsd = np.array(rip_tsd)
        
        rip_tsd = nap.Tsd(t=rip_tsd, d=rip_max, time_support=lfpnrem.time_support)
        
        rip_intervals[n] = nap.IntervalSet(rip_ep)
        rip_maxes[n] = nap.Ts(rip_tsd.index)
        
#%% 

    cc = nap.compute_eventcorrelogram(nap.TsGroup(rip_maxes), rip_maxes[0], binsize = 0.005, windowsize = 0.05, norm = True)
    
#%%   
    
    plt.figure()
    plt.tight_layout()
    plt.suptitle(datasets[r])
    plt.subplot(121)
    plt.plot(cc[cc.columns[1:]], 'o-', label = cc.columns[1:])
    plt.legend(loc = 'upper right')
    plt.xlabel('time from SWR')
    plt.ylabel('norm rate')
    plt.gca().set_box_aspect(1)
    
    plt.subplot(122)
    plt.imshow(cc[cc.columns[1:]].T, aspect = 'auto', cmap = 'magma', interpolation='bilinear',
               extent = [cc[cc.columns[1:]].index.values[0], 
                         cc[cc.columns[1:]].index.values[-1], 3, 1])
    plt.xlabel('time from SWR')
    plt.gca().set_box_aspect(1)
    
#%% 

    with open(os.path.join(path, 'rip_intervals.pickle'), 'wb') as pickle_file:
        pickle.dump(rip_intervals, pickle_file)

    with open(os.path.join(path, 'rip_maxes.pickle'), 'wb') as pickle_file:
        pickle.dump(rip_maxes, pickle_file)
