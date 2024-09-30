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
    
    # lfpnrem = lfp.restrict(sws_ep)
    
    # rip_intervals = {}
    # rip_maxes = {}
    
    # for n, k in enumerate(lfpnrem.columns):
         
    #     signal = pyna.eeg_processing.bandpass_filter(lfpnrem[:,n], 100, 200, 1250)
    #     windowLength = 51
    #     squared_signal = np.square(signal.values)
    #     window = np.ones(windowLength)/windowLength
    #     nSS = filtfilt(window, 1, squared_signal, axis=0)
    #     nSS = (nSS - np.mean(nSS))/np.std(nSS)
        
    #     nSS = nap.Tsd(t = signal.index.values, d = nSS, time_support = signal.time_support)
        
    #     low_thres = 1
    #     high_thres = 10
        
    #     nSS2 = nSS.threshold(low_thres, method = 'above')
    #     nSS3 = nSS2.threshold(high_thres, method = 'below')
        
    #     minRipLen = 20 # ms
    #     maxRipLen = 200 # ms

    #     rip_ep = nSS3.time_support
    #     rip_ep = rip_ep.drop_short_intervals(minRipLen, time_units = 'ms')
    #     rip_ep = rip_ep.drop_long_intervals(maxRipLen, time_units = 'ms')
        
    #     minInterRippleInterval = 20 # ms
    #     rip_ep = rip_ep.merge_close_intervals(minInterRippleInterval, time_units = 'ms')
        
    #     rip_max = []
    #     rip_tsd = []
    #     for s, e in rip_ep.values:
    #         tmp = nSS.as_series().loc[s:e]     #Change if nSS is TsdFrame
    #         rip_tsd.append(tmp.idxmax())
    #         rip_max.append(tmp.max())
        
    #     rip_max = np.array(rip_max)
    #     rip_tsd = np.array(rip_tsd)
        
    #     rip_tsd = nap.Tsd(t=rip_tsd, d=rip_max, time_support=lfpnrem.time_support)
        
    #     rip_intervals[n] = nap.IntervalSet(rip_ep)
    #     rip_maxes[n] = nap.Ts(rip_tsd.index)
    
#%% 

    with open(os.path.join(path, 'rip_intervals.pickle'), 'rb') as pickle_file:
        rip_intervals = pickle.load(pickle_file)
    
    with open(os.path.join(path, 'rip_maxes.pickle'), 'rb') as pickle_file:
        rip_maxes = pickle.load(pickle_file)
    
        
#%% 

    #cc = nap.compute_eventcorrelogram(nap.TsGroup(rip_maxes), rip_maxes[0], binsize = 0.0008, windowsize = 0.01, norm = True)
    
    # plt.figure()
    # plt.tight_layout()
    # plt.suptitle(datasets[r])
    # plt.subplot(121)
    # plt.plot(cc[cc.columns[1:]], 'o-', label = cc.columns[1:])
    # plt.legend(loc = 'upper right')
    # plt.xlabel('time from SWR')
    # plt.ylabel('norm rate')
    # plt.gca().set_box_aspect(1)
    
    # plt.subplot(122)
    # plt.imshow(cc[cc.columns[1:]].T, aspect = 'auto', cmap = 'magma', interpolation='bilinear',
    #            extent = [cc[cc.columns[1:]].index.values[0], 
    #                      cc[cc.columns[1:]].index.values[-1], 3, 1])
    # plt.xlabel('time from SWR')
    # plt.gca().set_box_aspect(1)
    
#%% 
    
    # rip_ref = rip_maxes[0]
    # realigned = lfp.as_dataframe().index[lfp.as_dataframe().index.get_indexer(rip_ref.index.values, method='nearest')]
    
    
    # # cc = nap.compute_event_trigger_average(nap.TsGroup(lfp), feature, binsize)
          
    # peth = {}
    # for i in range(len(lfp.columns)):
    #     peth[i] = nap.compute_perievent(lfp[:,i], rip_ref, minmax = (-0.05, 0.05), time_unit = 's')
    
    # peth_all_0 = []
    # for j in range(len(peth[0])):
    #     peth_all_0.append(peth[0][j].as_series())
    
    # avg_0 = pd.concat(peth_all_0, axis = 1, join = 'outer').dropna(how = 'all')
    # avg0 = avg_0.mean(axis=1)    
    
    
    # peth_all_1 = []
    # for j in range(len(peth[1])):
    #     peth_all_1.append(peth[1][j].as_series())
    
    # avg_1 = pd.concat(peth_all_1, axis = 1, join = 'outer').dropna(how = 'all')
    # avg1 = avg_1.mean(axis=1)    
    
    
    # peth_all_2 = []
    # for j in range(len(peth[2])):
    #     peth_all_2.append(peth[2][j].as_series())
    
    # avg_2 = pd.concat(peth_all_2, axis = 1, join = 'outer').dropna(how = 'all')
    # avg2 = avg_2.mean(axis=1)    
   
    
    # peth_all_3 = []
    # for j in range(len(peth[3])):
    #     peth_all_3.append(peth[3][j].as_series())
    
    # avg_3 = pd.concat(peth_all_3, axis = 1, join = 'outer').dropna(how = 'all')
    # avg3 = avg_3.mean(axis=1)    
    
    # plt.figure()
    # plt.tight_layout()
    # plt.suptitle(datasets[r])
    # plt.plot(avg0, label = '0')
    # plt.plot(avg1, label = '1')
    # plt.plot(avg2, label = '2')
    # plt.plot(avg3, label = '3')
    # plt.legend(loc = 'upper right')
    # plt.xlabel('time from SWR')
    # plt.gca().set_box_aspect(1)
    
#%% 
    
    # lfpts = {}
    
    # for i in range(len(lfp.columns)):
    #     tmp = nap.Tsd(lfp[:,i].as_series())
    #     lfpts[lfp.columns[i]] = tmp
    
    # lfpgroup = nap.TsGroup(lfpts)
    
    reftimes = {0: rip_maxes[0]}   
    refgroup = nap.TsGroup(reftimes)
    
    cc = nap.compute_event_trigger_average(refgroup, lfp, binsize = 0.0008, windowsize=(-0.25, 0.25), ep = sws_ep)
    
    plt.figure()
    plt.tight_layout()
    plt.suptitle(datasets[r])
    plt.plot(cc[:,0], label = cc[:,0].columns)
    plt.axvline(0, color = 'k', linestyle = '--')
    plt.legend(loc = 'upper right')
    plt.xlabel('time from SWR')
    plt.gca().set_box_aspect(1)
    
    
#%% 
    
    
    
    # plt.subplot(122)
    # plt.imshow(cc[cc.columns[1:]].T, aspect = 'auto', cmap = 'magma', interpolation='bilinear',
    #             extent = [cc[cc.columns[1:]].index.values[0], 
    #                       cc[cc.columns[1:]].index.values[-1], 3, 1])
    # plt.xlabel('time from SWR')
    # plt.gca().set_box_aspect(1)
    
  
    
    
#%% 

    # with open(os.path.join(path, 'rip_intervals.pickle'), 'wb') as pickle_file:
    #     pickle.dump(rip_intervals, pickle_file)

    # with open(os.path.join(path, 'rip_maxes.pickle'), 'wb') as pickle_file:
    #     pickle.dump(rip_maxes, pickle_file)
