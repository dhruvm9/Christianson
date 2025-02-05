#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  2 14:43:39 2023

@author: dhruv
"""
import numpy as np 
import nwbmatic as ntm
import pynapple as nap 
import pynacollada as pyna
import os, sys
import matplotlib.pyplot as plt 
import pickle
from scipy.signal import filtfilt

#%%

data_directory = '/media/dhruv/Expansion/Processed'
# data_directory = '/media/dhruv/Expansion/Processed/CA3'
# datasets = np.genfromtxt(os.path.join(data_directory,'dataset_DM.list'), delimiter = '\n', dtype = str, comments = '#')
datasets = np.genfromtxt(os.path.join(data_directory,'dataset_new_toadd.list'), delimiter = '\n', dtype = str, comments = '#')
# datasets = np.genfromtxt(os.path.join(data_directory,'dataset_test.list'), delimiter = '\n', dtype = str, comments = '#')
# ripplechannels = np.genfromtxt(os.path.join(data_directory,'ripplechannel.list'), delimiter = '\n', dtype = str, comments = '#')
ripplechannels = np.genfromtxt(os.path.join(data_directory,'ripplechannel_new_toadd.list'), delimiter = '\n', dtype = str, comments = '#')
# ripplechannels = np.genfromtxt(os.path.join(data_directory,'ripplechannel_test.list'), delimiter = '\n', dtype = str, comments = '#')

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
         
    ex_ep = nap.IntervalSet(start = 334, end = 338)
       
    lfpnrem = lfp.restrict(sws_ep)
            
    signal = pyna.eeg_processing.bandpass_filter(lfpnrem, 100, 200, 1250)
    # signal = pyna.eeg_processing.bandpass_filter(lfpnrem, 90, 120, 1250)
    
    plt.figure(figsize=(15,5))
    plt.subplot(211)
    plt.plot(lfpnrem.restrict(ex_ep))
    plt.subplot(212)
    plt.plot(signal.restrict(ex_ep))
    plt.xlabel("Time (s)")
    
#%% 

    windowLength = 51
    squared_signal = np.square(signal.values)
    window = np.ones(windowLength)/windowLength
    nSS = filtfilt(window, 1, squared_signal, axis=0)
    nSS = (nSS - np.mean(nSS))/np.std(nSS)
    
    nSS = nap.Tsd(t = signal.index.values, d = nSS, time_support = signal.time_support)
    # nSS = nap.TsdFrame(t = signal.index.values, d = nSS, time_support = signal.time_support)
    
    plt.figure(figsize=(15,5))
    plt.subplot(311)
    plt.plot(lfpnrem.restrict(ex_ep))
    plt.subplot(312)
    plt.plot(signal.restrict(ex_ep))
    plt.subplot(313)
    plt.plot(nSS.restrict(ex_ep))
    plt.xlabel("Time (s)")
    plt.tight_layout()
    
#%% 

    low_thres = 2
    high_thres = 10

    # nSS2 = {}
        
    # for i in nSS.columns: 
    #     nSS2[i] = nSS[i].threshold(low_thres, method = 'above')
 
    # nSS3 = {}
    # for i in nSS2.keys(): 
    #     nSS3[i] = nSS2[i].threshold(high_thres, method = 'below')
        
    # nSS3 = nap.TsGroup(nSS3)
    
    nSS2 = nSS.threshold(low_thres, method = 'above')
    nSS3 = nSS2.threshold(high_thres, method = 'below')
    
    # plt.figure(figsize=(15,5))
    # plt.subplot(311)
    # plt.plot(lfpnrem.restrict(ex_ep).as_units('s'))
    # plt.subplot(312)
    # plt.plot(signal.restrict(ex_ep).as_units('s'))
    # plt.subplot(313)
    # plt.plot(nSS.restrict(ex_ep).as_units('s'))
    # plt.plot(nSS3.restrict(ex_ep).as_units('s'))
    
    # for i in nSS3.keys():
    #     plt.plot(nSS3[i].restrict(sws_ep).as_units('s'), '.')
        
        
    plt.axhline(low_thres, color = 'k')
    plt.xlabel("Time (s)")
    plt.tight_layout()
    
#%% 

    minRipLen = 20 # ms
    maxRipLen = 200 # ms

    rip_ep = nSS3.time_support
    rip_ep = rip_ep.drop_short_intervals(minRipLen, time_units = 'ms')
    rip_ep = rip_ep.drop_long_intervals(maxRipLen, time_units = 'ms')

    # print(rip_ep)

#%% 

    minInterRippleInterval = 20 # ms
    rip_ep = rip_ep.merge_close_intervals(minInterRippleInterval, time_units = 'ms')
    # rip_ep = rip_ep.reset_index(drop=True)

#%% 

    rip_max = []
    rip_tsd = []
    for s, e in rip_ep.values:
        tmp = nSS.as_series().loc[s:e]     #Change if nSS is TsdFrame
        rip_tsd.append(tmp.idxmax())
        rip_max.append(tmp.max())
    
    rip_max = np.array(rip_max)
    rip_tsd = np.array(rip_tsd)
    
    rip_tsd = nap.Tsd(t=rip_tsd, d=rip_max, time_support=lfpnrem.time_support)
    
    plt.figure(figsize=(15,5))
    plt.subplot(311)
    plt.plot(lfpnrem.restrict(ex_ep).as_units('s'))
    plt.subplot(312)
    plt.plot(signal.restrict(ex_ep).as_units('s'))
    plt.subplot(313)
    plt.plot(nSS.restrict(ex_ep).as_units('s'))
    plt.plot(nSS3.restrict(rip_ep.intersect(ex_ep)).as_units('s'), '.')
    [plt.axvline(t, color = 'green') for t in rip_tsd.restrict(ex_ep).as_units('s').index.values]
    plt.axhline(low_thres)
    plt.xlabel("Time (s)")
    plt.tight_layout()

#%% 

    # data.write_neuroscope_intervals(extension = '.evt.py.rip', isets = rip_ep, name = 'rip') 
    
    with open(os.path.join(path, 'riptsd.pickle'), 'wb') as pickle_file:
        pickle.dump(rip_tsd, pickle_file)
    
    # data.save_nwb_intervals(rip_ep, 'sleep_ripples')
    # data.save_nwb_timeseries(rip_tsd, 'sleep_ripples')

#%% 

    # rip_ep, rip_tsd = pyna.eeg_processing.detect_oscillatory_events(
    #                                         lfp = lfp,
    #                                         epoch = sws_ep,
    #                                         freq_band = (100,300),
    #                                         thres_band = (1, 10),
    #                                         duration_band = (0.02,0.2),
    #                                         min_inter_duration = 0.02
    #                                         )
    
#%% Write to neuroscope

    start = rip_ep.as_units('ms')['start'].values
    ends = rip_ep.as_units('ms')['end'].values

    datatowrite = np.vstack((start,ends)).T.flatten()

    n = len(rip_ep)

    texttowrite = np.vstack(((np.repeat(np.array(['PyRip start 1']), n)), 
                              (np.repeat(np.array(['PyRip stop 1']), n))
                              )).T.flatten()

    evt_file = path + '/' + name + '.evt.py.rip'
    f = open(evt_file, 'w')
    for t, n in zip(datatowrite, texttowrite):
        f.writelines("{:1.6f}".format(t) + "\t" + n + "\n")
    f.close()        

    # sys.exit()