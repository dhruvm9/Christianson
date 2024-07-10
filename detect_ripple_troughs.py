#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 17:00:28 2024

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
datasets = np.genfromtxt(os.path.join(data_directory,'dataset_DM.list'), delimiter = '\n', dtype = str, comments = '#')
ripplechannels = np.genfromtxt(os.path.join(data_directory,'ripplechannel.list'), delimiter = '\n', dtype = str, comments = '#')

for r,s in enumerate(datasets):
    print(s)
    name = s.split('/')[-1]
    path = os.path.join(data_directory, s)
    
    data = ntm.load_session(path, 'neurosuite')
    data.load_neurosuite_xml(path)
       
    lfp = nap.load_eeg(path + '/' + s + '.eeg', channel = int(ripplechannels[r]), n_channels = 32, frequency = 1250)
    
    file = os.path.join(path, name +'.sws.evt')
    sws_ep = nap.IntervalSet(data.read_neuroscope_intervals(name = 'SWS', path2file = file))
        
    file = os.path.join(path, s +'.evt.py.rip')
    rip_ep = data.read_neuroscope_intervals(name = 'rip', path2file = file)
    
#%%  Detect ripple trough

    rip_trough = []

    for i in range(len(rip_ep)):
        tmp = lfp.restrict(nap.IntervalSet(rip_ep.loc[[i]]))
        rip_trough.append(tmp.as_series().idxmin())

    with open(os.path.join(path, 'riptrough.pickle'), 'wb') as pickle_file:
        pickle.dump(rip_trough, pickle_file)
        
        
        
        