#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 16:00:04 2023

@author: dhruv
"""

import numpy as np 
import pynapple as nap
import os, sys
import matplotlib.pyplot as plt 

#%% 

data_directory = '/media/dhruv/Expansion/Processed'
datasets = np.genfromtxt(os.path.join(data_directory,'dataset_DM.list'), delimiter = '\n', dtype = str, comments = '#')

isWT = []

t2p = []
rates = []
celltype = []
genotype = []

for s in datasets:
    print(s)
    name = s.split('-')[0]
        
    if name == 'B2613' or name == 'B2618':
        isWT.append(0)
    else: isWT.append(1)
    
    path = os.path.join(data_directory, s)

#%% Load classified spikes 

    sp2 = np.load(os.path.join(path, 'spikedata.npz'), allow_pickle = True)
    time_support = nap.IntervalSet(sp2['start'], sp2['end'])
    tsd = nap.Tsd(t=sp2['t'], d=sp2['index'], time_support = time_support)
    spikes = tsd.to_tsgroup()
    spikes.set_info(group = sp2['group'], location = sp2['location'], celltype = sp2['celltype'], tr2pk = sp2['tr2pk'])
    
    rates.extend(spikes._metadata['rate'].values)
    t2p.extend(spikes._metadata['tr2pk'].values)
    celltype.extend(spikes._metadata['celltype'].values)
    
    if name == 'B2613' or name == 'B2618':
        genotype.extend(np.zeros_like(spikes._metadata['rate'].values, dtype = 'bool'))
    else: genotype.extend(np.ones_like(spikes._metadata['rate'].values, dtype ='bool'))
    
   
#%% Sorting by genotype and plotting

t2p_wt = np.array(t2p)[genotype]
rates_wt = np.array(rates)[genotype]

pyr = [i for i, x in enumerate(np.array(celltype)[genotype]) if x == 'pyr']
fs = [i for i, x in enumerate(np.array(celltype)[genotype]) if x == 'fs']
oth = [i for i, x in enumerate(np.array(celltype)[genotype]) if x == 'other']

plt.figure()
plt.suptitle('Cell Type Classification')

plt.subplot(121)
plt.title('WT')
plt.scatter([t2p_wt[i] for i in pyr], [rates_wt[i] for i in pyr] , color = 'b', label = 'EX')        
plt.scatter([t2p_wt[i] for i in fs], [rates_wt[i] for i in fs] , color = 'r', label = 'FS')        
plt.scatter([t2p_wt[i] for i in oth], [rates_wt[i] for i in oth] , color = 'silver', label = 'Unclassified')       

plt.xlim([0, 1.5])
plt.ylim([0, 50])
plt.axhline(10, linestyle = '--', color = 'k')
plt.axvline(0.38, linestyle = '--', color = 'k')
plt.xlabel('Trough to peak (ms)')
plt.ylabel('Firing rate (Hz)')   
plt.legend(loc = 'upper right') 

t2p_ko = np.array(t2p)[np.invert(genotype)]
rates_ko = np.array(rates)[np.invert(genotype)]

pyr = [i for i, x in enumerate(np.array(celltype)[np.invert(genotype)]) if x == 'pyr']
fs = [i for i, x in enumerate(np.array(celltype)[np.invert(genotype)]) if x == 'fs']
oth = [i for i, x in enumerate(np.array(celltype)[np.invert(genotype)]) if x == 'other']

plt.subplot(122)
plt.title('KO')
plt.scatter([t2p_ko[i] for i in pyr], [rates_ko[i] for i in pyr] , color = 'b', label = 'EX')        
plt.scatter([t2p_ko[i] for i in fs], [rates_ko[i] for i in fs] , color = 'r', label = 'FS')        
plt.scatter([t2p_ko[i] for i in oth], [rates_ko[i] for i in oth] , color = 'silver', label = 'Unclassified')        

plt.xlim([0, 1.5])
plt.ylim([0, 50])
plt.axhline(10, linestyle = '--', color = 'k')
plt.axvline(0.38, linestyle = '--', color = 'k')
plt.xlabel('Trough to peak (ms)')
plt.ylabel('Firing rate (Hz)')   
plt.legend(loc = 'upper right')

