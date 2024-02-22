#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 16:33:16 2024

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
ripplechannels = np.genfromtxt(os.path.join(data_directory,'ripplechannel.list'), delimiter = '\n', dtype = str, comments = '#')

fs = 1250

s = datasets[5] #KO 
# s = datasets[13] #WT 

print(s)
name = s.split('-')[0]
path = os.path.join(data_directory, s)

data = ntm.load_session(path, 'neurosuite')
data.load_neurosuite_xml(path)
epochs = data.epochs


lfp = nap.load_eeg(path + '/' + s + '.eeg', channel = [1,15,0,14, 9, 13, 10, 11], n_channels = 32, frequency = fs) ### KO
# lfp = nap.load_eeg(path + '/' + s + '.eeg', channel = [29,19,27,23, 26, 28, 25, 24], n_channels = 32, frequency = fs) ### WT
    
file = os.path.join(path, s +'.evt.py.rip')
rip_ep = data.read_neuroscope_intervals(name = 'rip', path2file = file)


sp2 = np.load(os.path.join(path, 'spikedata.npz'), allow_pickle = True)
time_support = nap.IntervalSet(sp2['start'], sp2['end'])
tsd = nap.Tsd(t=sp2['t'], d=sp2['index'], time_support = time_support)
spikes = tsd.to_tsgroup()
spikes.set_info(group = sp2['group'], location = sp2['location'], celltype = sp2['celltype'], tr2pk = sp2['tr2pk'])

#%% 

# ex_ep = nap.IntervalSet(start = 4253.9, end = 4254.9) ###KO

ex_ep = nap.IntervalSet(start = 6999, end = 7001) ###wake KO

# ex_ep = nap.IntervalSet(start = 11844, end = 11846) ###WT



#%% 

###KO 

plt.figure()
k = 10
for i, n in enumerate(lfp.columns):
    plt.subplot(211)
    plt.title('KO')
    plt.plot(lfp.restrict(ex_ep).loc[[n]] + k, color = 'indianred')
    plt.xlim([ex_ep['start'].values, ex_ep['end'].values])
    # plt.plot(lfp.restrict(ex_ep.intersect(rip_ep).loc[[1]]).loc[[n]] + k, color = 'indianred')
    # plt.xlim([ex_ep.intersect(rip_ep).loc[[1]]['start'].values, ex_ep.intersect(rip_ep).loc[[1]]['end'].values])
    k +=4000
    plt.gca().set_box_aspect(1)
    
for i in range(len(spikes)):
    plt.subplot(212)
    plt.plot(spikes[i].restrict(ex_ep).fillna(i), '|', color = 'indianred')
    plt.xlim([ex_ep['start'].values, ex_ep['end'].values])
    # plt.plot(spikes[i].restrict(ex_ep.intersect(rip_ep).loc[[1]]).fillna(i), '|', color = 'indianred')
    # plt.xlim([ex_ep.intersect(rip_ep).loc[[1]]['start'].values, ex_ep.intersect(rip_ep).loc[[1]]['end'].values])
    plt.gca().set_box_aspect(1)
###WT 

# plt.figure()
# k = 10
# for i, n in enumerate(lfp.columns):
#     plt.subplot(211)
#     plt.title('WT')
#     # plt.plot(lfp.restrict(ex_ep).loc[[n]] + k, color = 'royalblue')
#     # plt.xlim([ex_ep['start'].values, ex_ep['end'].values])
#     plt.plot(lfp.restrict(ex_ep.intersect(rip_ep).loc[[0]]).loc[[n]] + k, color = 'royalblue')
#     plt.xlim([ex_ep.intersect(rip_ep).loc[[0]]['start'].values, ex_ep.intersect(rip_ep).loc[[0]]['end'].values])
#     k +=4000
#     plt.gca().set_box_aspect(1)

# for i in range(len(spikes)):
#     plt.subplot(212)
#     # plt.plot(spikes[i].restrict(ex_ep).fillna(i), '|', color = 'royalblue')
#     # plt.xlim([ex_ep['start'].values, ex_ep['end'].values])
#     plt.plot(spikes[i].restrict(ex_ep.intersect(rip_ep).loc[[0]]).fillna(i), '|', color = 'royalblue')
#     plt.xlim([ex_ep.intersect(rip_ep).loc[[0]]['start'].values, ex_ep.intersect(rip_ep).loc[[0]]['end'].values])
#     plt.gca().set_box_aspect(1)
