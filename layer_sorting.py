#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 11:28:12 2024

@author: dhruv
"""

import numpy as np 
import pandas as pd 
import nwbmatic as ntm
import os, sys
import pynapple as nap 
import matplotlib.pyplot as plt 

#%% 

data_directory = '/media/dhruv/Expansion/Processed'
datasets = np.genfromtxt(os.path.join(data_directory,'dataset_DM_rippletravel.list'), delimiter = '\n', dtype = str, comments = '#')
ripplechannels = pd.read_csv(os.path.join(data_directory,'ripplechannel_byshank.list'), dtype = int, header=None)

for r,s in enumerate(datasets):
    print(s)
    name = s.split('-')[0]
    path = os.path.join(data_directory, s)
    
    if name == 'B2613' or name == 'B2618' or name == 'B2627' or name == 'B2628':
        isWT = 0
    else: isWT = 1 
    
    data = ntm.load_session(path, 'neurosuite')
    data.load_neurosuite_xml(path)
    
    ripchannel = list(ripplechannels.loc[r].values)
    
    channelorder = data.group_to_channel
    
    
                    # {0: [16, 30, 17, 31, 18, 20, 22, 21],
                    # 1: [29, 19, 27, 23, 26, 28, 25, 24],
                    # 2: [12, 2, 8, 4, 3, 5, 7, 6],
                    # 3: [1, 15, 0, 14, 9, 13, 10, 11]}
    
#%% Find position of ripple channel and sort channels into superficial and deep

    ripshank = [0,1,2,3]
    rip_index = []


    for i in ripshank:
        tmp = np.where(channelorder[i] == [ripchannel[i]])[0][0]
        rip_index.append(tmp)
    
    # sup_channels = {}
    # deep_channels = {}
    
    # for i in channelorder.keys():
    #     sup_channels[i] = channelorder[i][0:rip_index] 
    #     deep_channels[i] = channelorder[i][rip_index+1:] 
    
    
#%% Load classified spikes 

    # t = time.time()
    sp2 = np.load(os.path.join(path, 'spikedata_0.55.npz'), allow_pickle = True)
    time_support = nap.IntervalSet(sp2['start'], sp2['end'])
    tsd = nap.Tsd(t=sp2['t'], d=sp2['index'], time_support = time_support)
    spikes = tsd.to_tsgroup()
    spikes.set_info(group = sp2['group'], location = sp2['location'], celltype = sp2['celltype'], maxch = sp2['maxch'])
    
#%% Load pyr cells only 

    spikes_by_celltype = spikes.getby_category('celltype')
    
    if 'pyr' in spikes._metadata['celltype'].values:
        pyr = spikes_by_celltype['pyr']
    
    layer = pd.Series(index = pyr.keys(), dtype = 'string')
    
    for i in pyr:
        group = pyr['group'][i]
        maxch = pyr['maxch'][i]
        
        if maxch < rip_index[group]:
            layer[i] = 'deep'
        elif maxch > rip_index[group]:
            layer[i] = 'sup'
        else: layer[i] = 'mid'
        
#%% 
        
    pyr.set_info(layer = layer)    
    pyr.save(os.path.join(path, 'pyr_layers.npz'))
        
    
    
        
    
    
    