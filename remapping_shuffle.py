#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 09:27:12 2024

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
import warnings
from scipy.stats import mannwhitneyu, wilcoxon
from functions_DM import *

#%% 

warnings.filterwarnings("ignore")

data_directory = '/media/dhruv/Expansion/Processed'
# data_directory = '/media/dhruv/Expansion/Processed/CA3'
# data_directory = '/media/adrien/Expansion/Processed'
datasets = np.genfromtxt(os.path.join(data_directory,'remapping_DM.list'), delimiter = '\n', dtype = str, comments = '#')
# datasets = np.genfromtxt(os.path.join(data_directory,'remapping_2sq.list'), delimiter = '\n', dtype = str, comments = '#')
# datasets = np.genfromtxt(os.path.join(data_directory,'remapping_CA3.list'), delimiter = '\n', dtype = str, comments = '#')

shuffs = []

for s in datasets:
    print(s)
    name = s.split('-')[0]
       
    path = os.path.join(data_directory, s)
    
    if name == 'B2618' or name == 'B2627' or name == 'B2628' or name == 'B3805' or name == 'B3813':
        isWT = 0
    else: isWT = 1 

    sp2 = np.load(os.path.join(path, 'spikedata_0.55.npz'), allow_pickle = True)
    time_support = nap.IntervalSet(sp2['start'], sp2['end'])
    tsd = nap.Tsd(t=sp2['t'], d=sp2['index'], time_support = time_support)
    spikes = tsd.to_tsgroup()
    spikes.set_info(group = sp2['group'], location = sp2['location'], celltype = sp2['celltype'], tr2pk = sp2['tr2pk'])
    
    data = ntm.load_session(path, 'neurosuite')
    epochs = data.epochs
    position = data.position
    
#%% Rotate position 

    rot_pos = []
        
    xypos = np.array(position[['x', 'z']])
    
    if name == 'B2625' or name == 'B3800':
        rad = 0.6
    elif name == 'B2618' :
        rad = 0.95
    elif s == 'B2627-240528' or s == 'B2627-240530' or name == 'B3804' or name == 'B3807' or name == 'B3813':
        rad = 0.05
    elif name == 'B3811':
        rad = 0.1
    else: rad = 0    
       
    for i in range(len(xypos)):
        newx, newy = rotate_via_numpy(xypos[i], rad)
        rot_pos.append((newx, newy))
        
    rot_pos = nap.TsdFrame(t = position.index.values, d = rot_pos, columns = ['x', 'z'])
    
    w1 = nap.IntervalSet(start = epochs['wake'][0]['start'], end = epochs['wake'][0]['end'])
    w2 = nap.IntervalSet(start = epochs['wake'][1]['start'], end = epochs['wake'][1]['end'])
    
#%% Get cells with wake rate more than 0.5Hz
        
    spikes_by_celltype = spikes.getby_category('celltype')
    
    if 'pyr' in spikes._metadata['celltype'].values:
        pyr = spikes_by_celltype['pyr']
    
    keep = []
    
    for i in pyr.index:
        if pyr.restrict(nap.IntervalSet(epochs['wake'][0]))._metadata['rate'][i] > 0.5:
            keep.append(i)

    pyr2 = pyr[keep]
    
#%% Compute speed during wake 

    if len(pyr2) > 2:
        
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
        
        ep1 = moving_ep.intersect(epochs['wake'].loc[[0]])
        ep2 = moving_ep.intersect(epochs['wake'].loc[[1]])
    
#%% Reverse the positions
    
        revpos = pd.DataFrame()
    
        for i in range(len(ep1)):
            pos = rot_pos[['x', 'z']].restrict(ep1[i]).as_dataframe()
            pos[:] = pos[::-1]
            revpos = pd.concat([revpos, pos])
         
        revpos = nap.TsdFrame(revpos)
    
#%% Compute shuffles for spatial info
       
    # for i in range(100): 
    #     spk = shuffleByCircularSpikes(pyr2, ep1)
    #     pf, binsxy = nap.compute_2d_tuning_curves(group = spk, features = rot_pos[['x', 'z']], ep = ep1, nb_bins=24)
    #     SI = nap.compute_2d_mutual_info(pf, rot_pos[['x', 'z']], ep = ep1)
    #     shuffs.extend(SI['SI'].tolist())
    
        # pf, binsxy = nap.compute_2d_tuning_curves(group = pyr2, features = revpos[['x', 'z']], ep = ep1, nb_bins=24)
        # SI = nap.compute_2d_mutual_info(pf, rot_pos[['x', 'z']], ep = ep1)
        # shuffs.extend(SI['SI'].tolist())
        
#%% Compute shuffle for stability 

        oddep = ep1[1::2]
        evenep = ep1[::2]
        
        pf1, binsxy = nap.compute_2d_tuning_curves(group = pyr2, features = revpos[['x', 'z']], ep = oddep, nb_bins=24)  
        pf2, binsxy = nap.compute_2d_tuning_curves(group = pyr2, features = revpos[['x', 'z']], ep = evenep, nb_bins=24)  
               
               
        for k in pyr2:
        
            good = np.logical_and(np.isfinite(pf1[k].flatten()), np.isfinite(pf2[k].flatten()))
            corr, p = scipy.stats.pearsonr(pf1[k].flatten()[good], pf2[k].flatten()[good]) 
            shuffs.append(corr)


        
            
#%% SI shuffle values 
    
# np.percentile(shuffs,95) = #  1.4782333798084752 (for data up to B3807)

#1.50350668555229 for reverse position shuffle up to B3807 (square + circle)
#1.52 for reverse position shuffle up to B3807 (square only)

#%% Stability shuffle values 

# 0.13912361655213026 for reverse position shuffle up to B3807 (square only)
