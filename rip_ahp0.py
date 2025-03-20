#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 09:57:29 2025

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
from scipy.stats import mannwhitneyu, pearsonr, norm

#%% 

def fisher_z(r):
    return 0.5 * np.log((1 + r) / (1 - r))

def compare_correlations(r1, n1, r2, n2):
    z1 = fisher_z(r1)
    z2 = fisher_z(r2)
    
    # Standard error
    se = np.sqrt(1/(n1 - 3) + 1/(n2 - 3))
    
    # Z-score for the difference
    z = (z1 - z2) / se
    
    # Two-tailed p-value
    p_value = 2 * (1 - norm.cdf(abs(z)))
    
    return z, p_value

#%% 

data_directory = '/media/dhruv/Expansion/Processed'
datasets = np.genfromtxt(os.path.join(data_directory,'dataset_DM.list'), delimiter = '\n', dtype = str, comments = '#')
ripplechannels = np.genfromtxt(os.path.join(data_directory,'ripplechannel.list'), delimiter = '\n', dtype = str, comments = '#')

fsamp = 1250

PSD_rip_wt = pd.DataFrame()
PSD_rip_ko = pd.DataFrame()

peakfreq_wt = []
peakfreq_ko = []

lor_pyr_wt = []
lor_pyr_ko = []
lor_fs_wt = []
lor_fs_ko = []

hor_pyr_wt = []
hor_pyr_ko = []
hor_fs_wt = []
hor_fs_ko = []


all_xc_pyr_wt = pd.DataFrame()
all_xc_fs_wt = pd.DataFrame()

all_xc_pyr_ko = pd.DataFrame()
all_xc_fs_ko = pd.DataFrame()

for r,s in enumerate(datasets):
    print(s)
    name = s.split('-')[0]
    path = os.path.join(data_directory, s)
    
    data = ntm.load_session(path, 'neurosuite')
    data.load_neurosuite_xml(path)
    epochs = data.epochs
    
    if name == 'B2613' or name == 'B2618' or name == 'B2627' or name == 'B2628' or name == 'B3805' or name == 'B3813':
        isWT = 0
    else: isWT = 1 
    
    sp2 = np.load(os.path.join(path, 'spikedata_0.55.npz'), allow_pickle = True)
    time_support = nap.IntervalSet(sp2['start'], sp2['end'])
    tsd = nap.Tsd(t=sp2['t'], d=sp2['index'], time_support = time_support)
    spikes = tsd.to_tsgroup()
    spikes.set_info(group = sp2['group'], location = sp2['location'], celltype = sp2['celltype'], tr2pk = sp2['tr2pk'])
        
    lfp = nap.load_eeg(path + '/' + s + '.eeg', channel = int(ripplechannels[r]), n_channels = 32, frequency = fsamp)
    
    file = os.path.join(path, s +'.sws.evt')
    sws_ep = nap.IntervalSet(data.read_neuroscope_intervals(name = 'SWS', path2file = file))    
        
    file = os.path.join(path, s +'.evt.py.rip')
    rip_ep = data.read_neuroscope_intervals(name = 'rip', path2file = file)
    
    with open(os.path.join(path, 'riptsd.pickle'), 'rb') as pickle_file:
        rip_tsd = pickle.load(pickle_file)
            
#%% Ripple Xcorr and minima of AHP

    spikes_by_celltype = spikes.getby_category('celltype')
    rip = nap.Ts(rip_tsd.index.values)
    
    if 'pyr' in spikes._metadata['celltype'].values:
        pyr = spikes_by_celltype['pyr']
        
    if 'fs' in spikes._metadata['celltype'].values:
        fs = spikes_by_celltype['fs']
        
    keep = []    
    for i in pyr.index:
        if pyr.restrict(nap.IntervalSet(epochs['wake'][0]))._metadata['rate'][i] > 0.5:
            keep.append(i)

    pyr2 = pyr[keep]
        
    if len(pyr2) > 0 and len(fs) > 0:
        
        xc_pyr = nap.compute_eventcorrelogram(pyr2, rip, binsize = 0.005, windowsize = 0.2 , ep = nap.IntervalSet(sws_ep), norm = True)
        
        maxr_pyr = xc_pyr.mean(axis=1).max()
        minr_pyr = xc_pyr.mean(axis=1)[0:0.105].min()
     
    
        if isWT == 1:
            all_xc_pyr_wt = pd.concat([all_xc_pyr_wt, xc_pyr], axis = 1)
            
        else: all_xc_pyr_ko = pd.concat([all_xc_pyr_ko, xc_pyr], axis = 1)
                  
    ###FS
        xc_fs = nap.compute_eventcorrelogram(fs, rip, binsize = 0.005, windowsize = 0.2 , ep = nap.IntervalSet(sws_ep), norm = True)
      
        minr_fs = xc_fs.mean(axis=1)[0:0.105].min()
        maxr_fs = xc_fs.mean(axis=1).max()
        
        if isWT == 1:
            all_xc_fs_wt = pd.concat([all_xc_fs_wt, xc_fs], axis = 1)
            lor_pyr_wt.append(minr_pyr)
            lor_fs_wt.append(minr_fs)
            hor_pyr_wt.append(maxr_pyr)
            hor_fs_wt.append(maxr_fs)
            
        else: 
            all_xc_fs_ko = pd.concat([all_xc_fs_ko, xc_fs], axis = 1)
            lor_pyr_ko.append(minr_pyr)
            lor_fs_ko.append(minr_fs)
            hor_pyr_ko.append(maxr_pyr)
            hor_fs_ko.append(maxr_fs)
              
#%% Restrict LFP
    
        lfp_rip = lfp.restrict(nap.IntervalSet(rip_ep))
        lfp_z = scipy.stats.zscore(lfp_rip)
    
#%% Power Spectral Density and peak frequency of ripple
          
        freqs, P_xx = scipy.signal.welch(lfp_z, fs = fsamp)
        
        ix2 = np.where((freqs>=100) & (freqs <= 200))
        peakfreq = freqs[ix2][np.argmax(P_xx[ix2])]
               
        if isWT == 1:
            PSD_rip_wt = pd.concat([PSD_rip_wt, pd.Series(P_xx)], axis = 1)
            peakfreq_wt.append(peakfreq)
            
        else: 
            PSD_rip_ko = pd.concat([PSD_rip_ko, pd.Series(P_xx)], axis = 1)
            peakfreq_ko.append(peakfreq)    

#%% 

plt.figure()
plt.subplot(121)
plt.title('PYR')
plt.scatter(peakfreq_wt, lor_pyr_wt, color = 'royalblue', label = 'WT')
plt.scatter(peakfreq_ko, lor_pyr_ko, color = 'indianred', label = 'KO')
plt.xlabel('SWR peak freq (Hz)')
plt.ylabel('Rate at minima of ripple AHP')
plt.legend(loc = 'upper right')
plt.gca().set_box_aspect(1)
plt.subplot(122)
plt.title('FS')
plt.scatter(peakfreq_wt, lor_fs_wt, color = 'royalblue', label = 'WT')
plt.scatter(peakfreq_ko, lor_fs_ko, color = 'indianred', label = 'KO')
plt.xlabel('SWR peak freq (Hz)')
plt.ylabel('Rate at minima of ripple AHP')
plt.legend(loc = 'upper right')
plt.gca().set_box_aspect(1)

plt.figure()
plt.subplot(121)
plt.title('PYR')
plt.scatter(peakfreq_wt, hor_pyr_wt, color = 'royalblue', label = 'WT')
plt.scatter(peakfreq_ko, hor_pyr_ko, color = 'indianred', label = 'KO')
plt.xlabel('SWR peak freq (Hz)')
plt.ylabel('Rate at maxima of ripple Xcorr')
plt.legend(loc = 'upper right')
plt.gca().set_box_aspect(1)
plt.subplot(122)
plt.title('FS')
plt.scatter(peakfreq_wt, hor_fs_wt, color = 'royalblue', label = 'WT')
plt.scatter(peakfreq_ko, hor_fs_ko, color = 'indianred', label = 'KO')
plt.xlabel('SWR peak freq (Hz)')
plt.ylabel('Rate at maxima of ripple Xcorr')
plt.legend(loc = 'upper right')
plt.gca().set_box_aspect(1)

#%%

r_pyr_wt, p_pyr_wt = pearsonr(peakfreq_wt, lor_pyr_wt)
r_pyr_ko, p_pyr_ko = pearsonr(peakfreq_ko, lor_pyr_ko)

r_fs_wt, p_fs_wt = pearsonr(peakfreq_wt, lor_fs_wt)
r_fs_ko, p_fs_ko = pearsonr(peakfreq_ko, lor_fs_ko)

a = []
a.extend(peakfreq_wt)
a.extend(peakfreq_ko)

b = []
b.extend(lor_pyr_wt)
b.extend(lor_pyr_ko)

c = []
c.extend(lor_fs_wt)
c.extend(lor_fs_ko)

r_pyr_pooled, p_pyr_pooled = pearsonr(a,b)
r_fs_pooled, p_fs_pooled = pearsonr(a,c)

z_pyr_12, p_pyr_12 = compare_correlations(r_pyr_wt, len(lor_pyr_wt), r_pyr_ko, len(lor_pyr_ko))
z_pyr_13, p_pyr_13 = compare_correlations(r_pyr_wt, len(lor_pyr_wt), r_pyr_pooled, len(c))
z_pyr_23, p_pyr_23 = compare_correlations(r_pyr_ko, len(lor_pyr_ko), r_pyr_pooled, len(c))

z_fs_12, p_fs_12 = compare_correlations(r_fs_wt, len(lor_fs_wt), r_fs_ko, len(lor_fs_ko))
z_fs_13, p_fs_13 = compare_correlations(r_fs_wt, len(lor_fs_wt), r_fs_pooled, len(c))
z_fs_23, p_fs_23 = compare_correlations(r_fs_ko, len(lor_fs_ko), r_fs_pooled, len(c))