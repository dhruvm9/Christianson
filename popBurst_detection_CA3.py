#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 10:38:59 2024

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

# data_directory = '/media/dhruv/Expansion/Processed'
data_directory = '/media/dhruv/Expansion/Processed/CA3'
datasets = np.genfromtxt(os.path.join(data_directory,'dataset_DM.list'), delimiter = '\n', dtype = str, comments = '#')
# datasets = np.genfromtxt(os.path.join(data_directory,'dataset_new_toadd.list'), delimiter = '\n', dtype = str, comments = '#')
ripplechannels = np.genfromtxt(os.path.join(data_directory,'ripplechannel.list'), delimiter = '\n', dtype = str, comments = '#')
# ripplechannels = np.genfromtxt(os.path.join(data_directory,'ripplechannel_new_toadd.list'), delimiter = '\n', dtype = str, comments = '#')


for r,s in enumerate(datasets):
    print(s)
    name = s.split('/')[-1]
    path = os.path.join(data_directory, s)
    
    data = ntm.load_session(path, 'neurosuite')
    data.load_neurosuite_xml(path)
        
    lfp = nap.load_eeg(path + '/' + s + '.eeg', channel = int(ripplechannels[r]), n_channels = 32, frequency = 1250)
        
    file = os.path.join(path, name +'.sws.evt')
    if os.path.exists(file):
        tmp = np.genfromtxt(file)[:,0]
        tmp = tmp.reshape(len(tmp)//2,2)/1000
        sws_ep = nap.IntervalSet(start = tmp[:,0], end = tmp[:,1], time_units = 's')

        
    spikes = data.spikes
    epochs = data.epochs
    
    #fishing out wake and sleep epochs
    sleep_ep = epochs['sleep']
    wake_ep = epochs['wake']
        
#%% 
         
    bin_size = 0.01 #s
    smoothing_window = 0.02

    rates = spikes.count(bin_size, sws_ep)
       
    total2 = rates.as_dataframe().rolling(window = 100 ,win_type='gaussian',
                                          center=True,min_periods=1, 
                                          axis = 0).mean(std= int(smoothing_window/bin_size))
    
    total2 = total2.sum(axis =1)
    total2 = nap.Tsd(total2)
    idx = total2.threshold(np.percentile(total2.values,90),'above')
            
    pb_ep = idx.time_support
    
    pb_ep = nap.IntervalSet(start = pb_ep['start'], end = pb_ep['end'])
    pb_ep = pb_ep.drop_short_intervals(bin_size)
    pb_ep = pb_ep.merge_close_intervals(bin_size*2)
    pb_ep = pb_ep.drop_short_intervals(bin_size*3)
    pb_ep = pb_ep.drop_long_intervals(bin_size*10)
   
    # sys.exit() 
   
    pb_ep = sws_ep.intersect(pb_ep)
    
#%% 
  

    start = pb_ep.as_units('ms')['start'].values
    ends = pb_ep.as_units('ms')['end'].values

    datatowrite = np.vstack((start,ends)).T.flatten()

    n = len(pb_ep)

    texttowrite = np.vstack(((np.repeat(np.array(['PyRip start 1']), n)), 
                              (np.repeat(np.array(['PyRip stop 1']), n))
                              )).T.flatten()

    evt_file = path + '/' + name + '.evt.py.rip'
    f = open(evt_file, 'w')
    for t, n in zip(datatowrite, texttowrite):
        f.writelines("{:1.6f}".format(t) + "\t" + n + "\n")
    f.close()        
    
    
