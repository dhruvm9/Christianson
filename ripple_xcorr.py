#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 15:01:09 2024

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

all_xc_pyr_wt = pd.DataFrame()
all_xc_fs_wt = pd.DataFrame()

all_xc_pyr_ko = pd.DataFrame()
all_xc_fs_ko = pd.DataFrame()

for s in datasets:
    print(s)
    name = s.split('-')[0]
       
    path = os.path.join(data_directory, s)
    
    if name == 'B2613' or name == 'B2618':
        isWT = 0
    else: isWT = 1 

    # sp2 = np.load(os.path.join(path, 'spikedata.npz'), allow_pickle = True)
    sp2 = np.load(os.path.join(path, 'spikedata_0.55.npz'), allow_pickle = True)
    time_support = nap.IntervalSet(sp2['start'], sp2['end'])
    tsd = nap.Tsd(t=sp2['t'], d=sp2['index'], time_support = time_support)
    spikes = tsd.to_tsgroup()
    spikes.set_info(group = sp2['group'], location = sp2['location'], celltype = sp2['celltype'], tr2pk = sp2['tr2pk'])
    
    data = ntm.load_session(path, 'neurosuite')
    epochs = data.epochs
    position = data.position
    
    file = os.path.join(path, s +'.sws.evt')
    sws_ep = nap.IntervalSet(data.read_neuroscope_intervals(name = 'SWS', path2file = file))
    
    file = os.path.join(path, s +'.evt.py.rip')
    rip_ep = data.read_neuroscope_intervals(name = 'rip', path2file = file)
    
    # with open(os.path.join(path, 'riptsd.pickle'), 'rb') as pickle_file:
        # rip_tsd = pickle.load(pickle_file)
        
    with open(os.path.join(path, 'riptrough.pickle'), 'rb') as pickle_file:
        rip_trough = pickle.load(pickle_file)
          
#%% Ripple cross corrs

    spikes_by_celltype = spikes.getby_category('celltype')
    
    if 'pyr' in spikes._metadata['celltype'].values:
        pyr = spikes_by_celltype['pyr']
        
        xc_pyr = nap.compute_eventcorrelogram(pyr, nap.Ts(rip_ep['start']), binsize = 0.005, windowsize = 0.1 , ep = nap.IntervalSet(sws_ep), norm = True)
        # xc_pyr = nap.compute_eventcorrelogram(pyr, nap.Ts(np.array(rip_trough)), binsize = 0.0005, windowsize = 0.1 , ep = nap.IntervalSet(sws_ep), norm = True)
    
        if isWT == 1:
            all_xc_pyr_wt = pd.concat([all_xc_pyr_wt, xc_pyr], axis = 1)
            
        else: all_xc_pyr_ko = pd.concat([all_xc_pyr_ko, xc_pyr], axis = 1)
                
    else: pyr = []
    
    if 'fs' in spikes._metadata['celltype'].values:
        fs = spikes_by_celltype['fs']
        
        xc_fs = nap.compute_eventcorrelogram(fs, nap.Ts(rip_ep['start']), binsize = 0.005, windowsize = 0.1 , ep = nap.IntervalSet(sws_ep), norm = True)
        # xc_fs = nap.compute_eventcorrelogram(fs, nap.Ts(np.array(rip_trough)), binsize = 0.0005, windowsize = 0.1 , ep = nap.IntervalSet(sws_ep), norm = True)
        
        if isWT == 1:
            all_xc_fs_wt = pd.concat([all_xc_fs_wt, xc_fs], axis = 1)
        else: all_xc_fs_ko = pd.concat([all_xc_fs_ko, xc_fs], axis = 1)
           
    else: fs = []
    
    del pyr, fs

       
               
        
#%% Plotting 


    # plt.figure()
    # plt.suptitle(s)
    # for n in range(len(spikes)):
    #     plt.subplot(9,8,n+1)
    #     plt.title(spikes._metadata['celltype'][n])
    #     plt.plot(xc[n])

plt.figure()
plt.tight_layout()
# plt.suptitle('Ripple onset Cross-correlogram')       

plt.subplot(121)
plt.title('PYR')
plt.xlabel('Time from SWR (s)')
plt.ylabel('norm. rate')
plt.plot(all_xc_pyr_wt.mean(axis=1), color = 'lightsteelblue', label = 'WT')
err = all_xc_pyr_wt.sem(axis=1)
plt.fill_between(all_xc_pyr_wt.index.values, all_xc_pyr_wt.mean(axis=1) - err, all_xc_pyr_wt.mean(axis=1) + err, alpha = 0.2, color = 'lightsteelblue') 

plt.plot(all_xc_pyr_ko.mean(axis=1), color = 'lightcoral', label = 'KO')
err = all_xc_pyr_ko.sem(axis=1)
plt.fill_between(all_xc_pyr_ko.index.values, all_xc_pyr_ko.mean(axis=1) - err, all_xc_pyr_ko.mean(axis=1) + err, alpha = 0.2, color = 'lightcoral') 
plt.legend(loc = 'upper right')
plt.xticks([-0.1, 0, 0.1])
plt.yticks([1,5])
plt.gca().set_box_aspect(1)

        
plt.subplot(122)
plt.title('FS')
plt.xlabel('Time from SWR (s)')
# plt.ylabel('norm. rate')
plt.plot(all_xc_fs_wt.mean(axis=1), color = 'royalblue', label = 'WT')
err = all_xc_fs_wt.sem(axis=1)
plt.fill_between(all_xc_fs_wt.index.values, all_xc_fs_wt.mean(axis=1) - err, all_xc_fs_wt.mean(axis=1) + err, alpha = 0.2, color = 'royalblue') 

plt.plot(all_xc_fs_ko.mean(axis=1), color = 'indianred', label = 'KO')
err = all_xc_fs_ko.sem(axis=1)
plt.fill_between(all_xc_fs_ko.index.values, all_xc_fs_ko.mean(axis=1) - err, all_xc_fs_ko.mean(axis=1) + err, alpha = 0.2, color = 'indianred') 
plt.legend(loc = 'upper right')
plt.xticks([-0.1, 0, 0.1])
plt.yticks([1,4.5])
plt.gca().set_box_aspect(1)