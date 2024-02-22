#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 11:54:13 2024

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
from random import sample

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


lfp = nap.load_eeg(path + '/' + s + '.eeg', channel = [1,15,0,14, 9, 13, 10, 11], n_channels = 32, frequency = fs) 
    
file = os.path.join(path, s +'.evt.py.rip')
rip_ep = data.read_neuroscope_intervals(name = 'rip', path2file = file)

with open(os.path.join(path, 'riptsd.pickle'), 'rb') as pickle_file:
    rip_tsd = pickle.load(pickle_file)


sp2 = np.load(os.path.join(path, 'spikedata.npz'), allow_pickle = True)
time_support = nap.IntervalSet(sp2['start'], sp2['end'])
tsd = nap.Tsd(t=sp2['t'], d=sp2['index'], time_support = time_support)
spikes = tsd.to_tsgroup()
spikes.set_info(group = sp2['group'], location = sp2['location'], celltype = sp2['celltype'], tr2pk = sp2['tr2pk'])

#%% Plot participation of cell over a few select ripples 

spikes_by_celltype = spikes.getby_category('celltype')
if 'pyr' in spikes._metadata['celltype'].values:
    pyr = spikes_by_celltype['pyr']
   
if 'fs' in spikes._metadata['celltype'].values:
    fs = spikes_by_celltype['fs']

# peth = nap.compute_perievent(fs, nap.Ts(rip_tsd.index.values), minmax = (-0.07, 0.07)) ###WT
# samples = sample(list(peth[10].index), 75)

peth = nap.compute_perievent(fs[42], nap.Ts(rip_tsd.index.values), minmax = (-0.07, 0.07)) ###KO
samples = sample(list(peth.index), 75)


#%% 

###WT
# ex_ep = nap.IntervalSet(start = 11844, end = 11846)
# ex_ep = ex_ep.intersect(rip_ep).loc[[0]]

# ripcenter = rip_tsd.restrict(ex_ep).index.values

# twin = 0.07
# ripple_events = nap.IntervalSet(start = ripcenter - twin, end = ripcenter + twin)

# plt.figure()
# plt.subplot(211)
# plt.plot(lfp.restrict(ripple_events).loc[[15]], color = 'royalblue') ### Channel 15 is best for this example
# plt.gca().set_box_aspect(1)

# plt.subplot(212)
# for i in range(len(samples)):
#     plt.plot(peth[10][samples[i]].fillna(i), '|', color = 'royalblue')
# plt.gca().set_box_aspect(1)


###KO
ex_ep = nap.IntervalSet(start = 4253.9, end = 4254.9)
ex_ep = ex_ep.intersect(rip_ep).loc[[0]]

ripcenter = rip_tsd.restrict(ex_ep).index.values

twin = 0.07
ripple_events = nap.IntervalSet(start = ripcenter - twin, end = ripcenter + twin)

plt.figure()
plt.subplot(211)
plt.plot(lfp.restrict(ripple_events).loc[[0]], color = 'indianred') ### Channel 0 is best for this example
plt.gca().set_box_aspect(1)

plt.subplot(212)
for i in range(len(samples)):
    plt.plot(peth[samples[i]].fillna(i), '|', color = 'indianred')
plt.gca().set_box_aspect(1)