#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 10:25:36 2023

@author: dhruv
"""

import numpy as np 
import pandas as pd 
import nwbmatic as ntm
import pynapple as nap
import os, sys
import matplotlib.pyplot as plt 
import pickle 

#%% 

# data_directory = '/media/dhruv/Expansion/Processed'
data_directory = '/media/dhruv/Expansion/Processed/CA3'
datasets = np.genfromtxt(os.path.join(data_directory,'dataset_DM.list'), delimiter = '\n', dtype = str, comments = '#')
# datasets = np.genfromtxt(os.path.join(data_directory,'dataset_test.list'), delimiter = '\n', dtype = str, comments = '#')
# datasets = np.genfromtxt(os.path.join(data_directory,'dataset_new_toadd.list'), delimiter = '\n', dtype = str, comments = '#')

isWT = []

for s in datasets:
    print(s)
    name = s.split('/')[-1]
        
    if name == 'B2613' or name == 'B2618' or name == 'B2627' or name == 'B2628':
        isWT = 0
    else: isWT = 1
    
    path = os.path.join(data_directory, s)
    
    data = ntm.load_session(path, 'neurosuite')
    spikes = data.spikes
    epochs = data.epochs
    
    # meanwf, maxch = data.load_mean_waveforms()
    
    # with open(os.path.join(path, 'meanwf.pickle'), 'wb') as pickle_file:
    #     pickle.dump(meanwf, pickle_file)
        
    # with open(os.path.join(path, 'maxch.pickle'), 'wb') as pickle_file:
    #     pickle.dump(maxch, pickle_file)
        
    with open(os.path.join(path, 'meanwf.pickle'), 'rb') as pickle_file:
        meanwf = pickle.load(pickle_file)
    
    with open(os.path.join(path, 'maxch.pickle'), 'rb') as pickle_file:
        maxch = pickle.load(pickle_file)
        
    spikes.set_info(maxch = maxch)
    
        
#%% 

    maxwf = {neuron: meanwf[neuron][maxch[neuron]] for neuron in meanwf}
    
#Let's visualize the mean waveforms of the neurons
    # plt.figure(figsize = (15,9))
    # plt.suptitle(s)
    # for i,neuron in enumerate(maxwf):
    #     plt.subplot(5,3,i+1)
    #     # plt.plot(meanwf[neuron])
    #     plt.plot(maxwf[neuron])
    # plt.tight_layout()

#%% Trough-to-peak

    tr2pk = pd.Series(index = maxwf.keys(), dtype = 'float64')
    alltroughs = {}
    allpeaks = {} 
    
    # plt.figure(figsize = (15,9))
    for i,neuron in enumerate(maxwf):
        trough = maxwf[neuron].idxmin() #find the time of waveform trough
        peak = maxwf[neuron][trough:].idxmax() #find the time of the peak occurring after the trough
        tr2pk[neuron] = peak - trough
    
        #Save the peak and the trough of the neuron outside the loop - we'll be using them later
        alltroughs[neuron] = trough
        allpeaks[neuron] = peak
        
        #Plot the waveforms to check we found the right points
    #     plt.subplot(5,3,i+1)
    #     plt.plot(maxwf[neuron])
    #     plt.axvline(trough, linestyle = 'dashed', c = 'r', label = 'trough')
    #     plt.axvline(peak, linestyle = 'dotted', c = 'r', label = 'peak')
    #     plt.legend()
    # plt.tight_layout()
        
    spikes.set_info(tr2pk = tr2pk*1e3)    
        
#%% Half peak width

    # half_peak_width = pd.Series(index = maxwf.keys(), dtype = 'float64')
    
    # plt.figure(figsize = (15,9))
    # for i,neuron in enumerate(maxwf):
    
    #     peak_amp = maxwf[neuron][alltroughs[neuron]:].max()/2 #half of max amplitude of the peak after the trough
        
    #     #find the closest point to the peak amplitude before and after the peak
    #     width1 = np.abs(maxwf[neuron][alltroughs[neuron]:allpeaks[neuron]] - peak_amp).idxmin()
    #     width2 = np.abs(maxwf[neuron][allpeaks[neuron]:] - peak_amp).idxmin()
        
    #     #append half peak width
    #     half_peak_width[neuron] = width2-width1
        
    #     #plot to make sure we are finding the right points
    #     plt.subplot(5,3,i+1)
    #     plt.plot(maxwf[neuron])
    #     plt.axhline(peak_amp, linestyle = 'dotted', c = 'k', label = 'Half max amplitude')
    #     plt.axvline(width1, linestyle = 'dashed', c = 'r', label = 'half peak width')
    #     plt.axvline(width2, linestyle = 'dashed', c = 'r')
    #     plt.legend()
    # plt.tight_layout()

#%% Append data to spikes 
    
    celltype = pd.Series(index = spikes.keys(), dtype = 'string')
    for i in spikes:
        # if tr2pk[i]*1e3 > 0.38 and spikes[i].rate < 10:
        #     celltype[i] = 'pyr'
        # elif tr2pk[i]*1e3 < 0.38 and spikes[i].rate > 10:
        #     celltype[i] = 'fs'
        # else: celltype[i] = 'other'
    
        if tr2pk[i]*1e3 > 0.55 and spikes.restrict(nap.IntervalSet(epochs['wake'][0]))._metadata['rate'][i] < 10:
            celltype[i] = 'pyr'
        elif tr2pk[i]*1e3 < 0.55 and spikes.restrict(nap.IntervalSet(epochs['wake'][0]))._metadata['rate'][i] > 10:
            celltype[i] = 'fs'
        else: celltype[i] = 'other'
    
        
    # with open(os.path.join(path, 'celltype.pickle'), 'wb') as pickle_file:
    #     pickle.dump(celltype, pickle_file)
    
    # with open(os.path.join(path, 'celltype.pickle'), 'rb') as pickle_file:
    #     celltype = pickle.load(pickle_file)
        
    spikes.set_info(celltype = celltype)    
    
    spikes_by_celltype = spikes.getby_category(celltype)
    if 'pyr' in celltype.values == True:
        pyr = spikes_by_celltype['pyr']
    
    if 'fs' in celltype.values == True:
        fs = spikes_by_celltype['fs']
    
    if 'other' in celltype.values == True:
        oth = spikes_by_celltype['other']
   

#%% Plotting firing rate v/s tr2pk
    
    plt.figure()
    plt.title(s)
    if 'pyr' in celltype.values == True:
        plt.scatter(pyr._metadata['tr2pk'], pyr.restrict(nap.IntervalSet(epochs['wake'][0]))._metadata['rate'], color = 'b', label = 'EX')
    
    if 'fs' in celltype.values == True:
        plt.scatter(fs._metadata['tr2pk'], fs.restrict(nap.IntervalSet(epochs['wake'][0]))._metadata['rate'], color = 'r', label = 'FS')
    
    if 'other' in celltype.values == True:
        plt.scatter(oth._metadata['tr2pk'], oth.restrict(nap.IntervalSet(epochs['wake'][0]))._metadata['rate'], color = 'silver')
   
    plt.axhline(10, linestyle = '--', color = 'k')
    # plt.axvline(0.38, linestyle = '--', color = 'k')
    plt.axvline(0.55, linestyle = '--', color = 'k')
    plt.xlabel('Trough to peak (ms)')
    plt.ylabel('Firing rate (Hz)')   
    plt.legend(loc = 'upper right')
    
    # spikes.save(os.path.join(path, 'spikedata.npz'))
    spikes.save(os.path.join(path, 'spikedata_0.55.npz'))

#%% 


#Load the npz file 
    # sp2 = np.load(os.path.join(path, 'spikedata.npz'), allow_pickle = True)
    # time_support = nap.IntervalSet(sp2['start'], sp2['end'])
    # tsd = nap.Tsd(t=sp2['t'], d=sp2['index'], time_support = time_support)
    # tsgroup = tsd.to_tsgroup()
    # tsgroup.set_info(group = sp2['group'], location = sp2['location'], celltype = sp2['celltype'])
        
    # sys.exit()
    
    
    
    