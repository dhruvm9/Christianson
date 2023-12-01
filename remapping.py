#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 15:17:52 2023

@author: dhruv
"""

import numpy as np 
import pandas as pd 
import nwbmatic as ntm
import scipy.io
import pynapple as nap 
import os, sys
import matplotlib.pyplot as plt 
import seaborn as sns
from scipy.stats import mannwhitneyu
from matplotlib.backends.backend_pdf import PdfPages    

#%% #%% 

def multipage(filename, figs=None, dpi=200):
    pp = PdfPages(filename)
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()
    
    
#%% 

data_directory = '/media/dhruv/Expansion/Processed'
datasets = np.genfromtxt(os.path.join(data_directory,'remapping_DM.list'), delimiter = '\n', dtype = str, comments = '#')


for s in datasets:
    print(s)
    name = s.split('-')[0]
       
    path = os.path.join(data_directory, s)
    
    if name == 'B2618':
        isWT = 0
    else: isWT = 1 

    sp2 = np.load(os.path.join(path, 'spikedata.npz'), allow_pickle = True)
    time_support = nap.IntervalSet(sp2['start'], sp2['end'])
    tsd = nap.Tsd(t=sp2['t'], d=sp2['index'], time_support = time_support)
    spikes = tsd.to_tsgroup()
    spikes.set_info(group = sp2['group'], location = sp2['location'], celltype = sp2['celltype'], tr2pk = sp2['tr2pk'])
    
    data = ntm.load_session(path, 'neurosuite')
    epochs = data.epochs
    position = data.position
    
    w1 = nap.IntervalSet(start = epochs['wake'].loc[0]['start'], end = epochs['wake'].loc[0]['end'])
    w2 = nap.IntervalSet(start = epochs['wake'].loc[1]['start'], end = epochs['wake'].loc[1]['end'])
                                     
    placefields1, binsxy1 = nap.compute_2d_tuning_curves(group = spikes, 
                                                       feature = position[['x', 'z']], 
                                                       ep = w1, 
                                                       nb_bins=20)      
    
    placefields2, binsxy2 = nap.compute_2d_tuning_curves(group = spikes, 
                                                       feature = position[['x', 'z']], 
                                                       ep = w2, 
                                                       nb_bins=20)
    
    for i in spikes.keys(): 
        placefields1[i][np.isnan(placefields1[i])] = 0
        placefields1[i] = scipy.ndimage.gaussian_filter(placefields1[i], 1)
        
        placefields2[i][np.isnan(placefields2[i])] = 0
        placefields2[i] = scipy.ndimage.gaussian_filter(placefields2[i], 1)
    
    
#%% Plot tracking 

    plt.figure()
    plt.suptitle(s)
    plt.subplot(121)
    plt.plot(position['x'].restrict(w1), position['z'].restrict(w1))
    plt.subplot(122)
    plt.plot(position['x'].restrict(w2), position['z'].restrict(w2))
    
#%% Plot remapping 

    plt.figure()
    plt.suptitle(s + ' Wake1')
    for n in range(len(spikes)):
        plt.subplot(9,8,n+1)
        plt.title(spikes._metadata['celltype'][n])
        plt.imshow(placefields1[n], extent=(binsxy1[1][0],binsxy1[1][-1],binsxy1[0][0],binsxy1[0][-1]), cmap = 'jet')        
        plt.colorbar()

    plt.figure()
    plt.suptitle(s + ' Wake2')
    for n in range(len(spikes)):
        plt.subplot(9,8,n+1)
        plt.title(spikes._metadata['celltype'][n])
        plt.imshow(placefields2[n], extent=(binsxy2[1][0],binsxy2[1][-1],binsxy2[0][0],binsxy2[0][-1]), cmap = 'jet')        
        plt.colorbar()

    
    