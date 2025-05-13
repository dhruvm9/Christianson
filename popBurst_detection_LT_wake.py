#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 11:40:11 2025

@author: dhruv
"""

import numpy as np 
import nwbmatic as ntm
import pynapple as nap 
import pynacollada as pyna
import os, sys
import matplotlib.pyplot as plt 
import pickle
import warnings
from scipy.signal import filtfilt
from functions_DM import *

#%% 

warnings.filterwarnings("ignore")

# data_directory = '/media/dhruv/Expansion/Processed'
data_directory = '/media/dhruv/Expansion/Processed/LinearTrack'
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
        
    spikes = nap.load_file(os.path.join(path, 'spikedata_0.55.npz'))
    
    epochs = data.epochs
    position = data.position
    
    #fishing out wake and sleep epochs
    sleep_ep = epochs['sleep']
    wake_ep = epochs['wake']
    
#%% Rotate position 

    rot_pos = []
        
    xypos = np.array(position[['x', 'z']])
    
    if name in ['B4701', 'B4702', 'B4704', 'B4705', 'B4707']:
        rad = 0.2
    else: rad = 0    
            
    for i in range(len(xypos)):
        newx, newy = rotate_via_numpy(xypos[i], rad)
        rot_pos.append((newx, newy))
        
    rot_pos = nap.TsdFrame(t = position.index.values, d = rot_pos, columns = ['x', 'z'])
    
#%% Get only high-firing PYR cells
    
    spikes_by_celltype = spikes.getby_category('celltype')
    
    if 'pyr' in spikes._metadata['celltype'].values:
        pyr = spikes_by_celltype['pyr']
    
    keep = []
    
    for i in pyr.index:
        if pyr.restrict(nap.IntervalSet(epochs['wake']))._metadata['rate'][i] > 0.5:
            keep.append(i)
              
            
    pyr2 = pyr[keep]
    
#%% Compute speed during wake 

    if len(pyr2) >= 10:
        print('Yes!')
                
        speedbinsize = np.diff(rot_pos.index.values)[0]
        
        time_bins = np.arange(rot_pos.index[0], rot_pos.index[-1] + speedbinsize, speedbinsize)
        index = np.digitize(rot_pos.index.values, time_bins)
        tmp = rot_pos.as_dataframe().groupby(index).mean()
        tmp.index = time_bins[np.unique(index)-1]+(speedbinsize)/2
        distance = np.sqrt(np.power(np.diff(tmp['x']), 2) + np.power(np.diff(tmp['z']), 2)) * 100 #in cm
        speed = pd.Series(index = tmp.index.values[0:-1]+ speedbinsize/2, data = distance/speedbinsize) # in cm/s
        speed2 = speed.rolling(window = 25, win_type='gaussian', center=True, min_periods=1).mean(std=10) #Smooth over 200ms 
        speed2 = nap.Tsd(speed2)
             
        moving_ep = nap.IntervalSet(speed2.threshold(2).time_support) #Epochs in which speed is > 2 cm/s
        
        still_ep = wake_ep.set_diff(moving_ep)
    
#%% Detect population bursts during periods of stillness
      
     
        bin_size = 0.01 #s
        smoothing_window = 0.02
    
        rates = spikes.count(bin_size, still_ep)
           
        total2 = rates.as_dataframe().rolling(window = 100 ,win_type='gaussian',
                                              center=True,min_periods=1, 
                                              axis = 0).mean(std= int(smoothing_window/bin_size))
        
        total2 = total2.sum(axis =1)
        total2 = nap.Tsd(total2)
        idx = total2.threshold(np.percentile(total2.values,90),'above')
                
        pb_ep = idx.time_support
        
        pb_ep = nap.IntervalSet(start = pb_ep['start'], end = pb_ep['end'])
        # pb_ep = pb_ep.drop_short_intervals(bin_size)
        pb_ep = pb_ep.merge_close_intervals(bin_size*2)
        pb_ep = pb_ep.drop_short_intervals(bin_size*3)
        pb_ep = pb_ep.drop_long_intervals(bin_size*50)
       
        # sys.exit() 
       
        pb_ep = wake_ep.intersect(pb_ep)    
    
#%% 
  

        start = pb_ep.as_units('ms')['start'].values
        ends = pb_ep.as_units('ms')['end'].values
    
        datatowrite = np.vstack((start,ends)).T.flatten()
    
        n = len(pb_ep)
    
        texttowrite = np.vstack(((np.repeat(np.array(['PyW_PB start 1']), n)), 
                                  (np.repeat(np.array(['PyW_PB stop 1']), n))
                                  )).T.flatten()
    
        evt_file = path + '/' + name + '.evt.py.wpb'
        f = open(evt_file, 'w')
        for t, n in zip(datatowrite, texttowrite):
            f.writelines("{:1.6f}".format(t) + "\t" + n + "\n")
        f.close()        