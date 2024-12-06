#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 10:12:07 2024

@author: dhruv
"""

import numpy as np 
import pandas as pd 
import os, sys
import matplotlib.pyplot as plt 
import nwbmatic as ntm
import pynapple as nap
from scipy.signal import find_peaks, hilbert
from functions_DM import *    

#%% 

# data_directory = '/media/dhruv/Expansion/Processed'
data_directory = '/media/adrien/Expansion/Processed'
datasets = np.genfromtxt(os.path.join(data_directory,'dataset_DM.list'), delimiter = '\n', dtype = str, comments = '#')
ripplechannels = np.genfromtxt(os.path.join(data_directory,'ripplechannel.list'), delimiter = '\n', dtype = str, comments = '#')

fs = 1250

asym_wake_wt = []
asym_wake_ko = []

asym_rem_wt = []
asym_rem_ko = []

for r,s in enumerate(datasets):
    print(s)
    name = s.split('-')[0]
    path = os.path.join(data_directory, s)
    
    data = ntm.load_session(path, 'neurosuite')
    data.load_neurosuite_xml(path)
    epochs = data.epochs
    position = data.position
    
    if name == 'B2613' or name == 'B2618' or name == 'B2628' or name == 'B3805' or name == 'B3813':
        isWT = 0
    else: isWT = 1 
    
    lfp = nap.load_eeg(path + '/' + s + '.eeg', channel = int(ripplechannels[r]), n_channels = 32, frequency = fs)
            
    file = os.path.join(path, s +'.rem.evt')
    rem_ep = data.read_neuroscope_intervals(name = 'REM', path2file = file)
          
    
#%% Compute speed during wake 
       
    speedbinsize = np.diff(position.index.values)[0]
    
    time_bins = np.arange(position.index[0], position.index[-1] + speedbinsize, speedbinsize)
    index = np.digitize(position.index.values, time_bins)
    tmp = position.as_dataframe().groupby(index).mean()
    tmp.index = time_bins[np.unique(index)-1]+(speedbinsize)/2
    distance = np.sqrt(np.power(np.diff(tmp['x']), 2) + np.power(np.diff(tmp['z']), 2)) * 100 #in cm
    speed = nap.Tsd(t = tmp.index.values[0:-1]+ speedbinsize/2, d = distance/speedbinsize) # in cm/s
 
    moving_ep = nap.IntervalSet(speed.threshold(2).time_support) #Epochs in which speed is > 2 cm/s
        
#%% 
    
    lfp_filt_theta = bandpass_filter_zerophase(lfp.restrict(moving_ep), 6, 9, fs)
    power_theta = nap.Tsd(lfp_filt_theta.index.values, np.abs(hilbert(lfp_filt_theta.values)))
    
    power_thresh = power_theta.threshold(np.percentile(power_theta.values, 90))
    
### Wake epoch with power threshold and movement 
    
    wake = power_thresh.restrict(moving_ep).time_support
    wake = wake.drop_short_intervals(0.1)
       
    
    lfp_wake = lfp.restrict(nap.IntervalSet(wake))
    lfp_rem = lfp.restrict(nap.IntervalSet(rem_ep))    
    
    lfp_filt_wake = bandpass_filter_zerophase(lfp_wake, 1, 80, 1250)
    lfp_filt_rem = bandpass_filter_zerophase(lfp_rem, 1, 80, 1250)
    
    peaks_w = find_peaks(lfp_filt_wake, distance = 100)[0]
    troughs_w = find_peaks(-lfp_filt_wake, distance = 100)[0]
    
    peaks_r = find_peaks(lfp_filt_rem, distance = 100)[0]
    troughs_r = find_peaks(-lfp_filt_rem, distance = 100)[0]
    
#%% 

### Check peaks
    
    # plt.figure()
    # plt.plot(lfp_wake)
    # plt.plot(lfp_filt_wake)
    # plt.plot(lfp_filt_wake[peaks_w], 'o')
    # plt.plot(lfp_filt_wake[troughs_w], 'o')
    
    # plt.figure()
    # plt.plot(lfp_rem)
    # plt.plot(lfp_filt_rem)
    # plt.plot(lfp_filt_rem[peaks_r], 'o')
    # plt.plot(lfp_filt_rem[troughs_r], 'o')
    
    
#%% 

    nCycles_w = min(len(peaks_w), len(troughs_w))    
    nCycles_r = min(len(peaks_r), len(troughs_r))    

    asc_w = nap.IntervalSet(start = lfp_filt_wake.index.values[troughs_w[0:nCycles_w]], 
                            end = lfp_filt_wake.index.values[peaks_w[0:nCycles_w]])[0:-1]
    des_w = nap.IntervalSet(start = lfp_filt_wake.index.values[peaks_w[0:nCycles_w-1]], 
                            end = lfp_filt_wake.index.values[troughs_w[1:nCycles_w]])[0:-1]
        
    asc_r = nap.IntervalSet(start = lfp_filt_rem.index.values[troughs_r[0:nCycles_r]], 
                            end = lfp_filt_rem.index.values[peaks_r[0:nCycles_r]])[0:-1]
    des_r = nap.IntervalSet(start = lfp_filt_rem.index.values[peaks_r[0:nCycles_r-1]], 
                            end = lfp_filt_rem.index.values[troughs_r[1:nCycles_r]])[0:-1]
        
    
    dur_asc_w = asc_w['end'] - asc_w['start'] 
    dur_des_w = des_w['end'] - des_w['start'] 
    
    dur_asc_r = asc_r['end'] - asc_r['start'] 
    dur_des_r = des_r['end'] - des_r['start'] 
               
    asymmetry_wake = np.log10(dur_asc_w[0:min(len(dur_asc_w), len(dur_des_w))] / dur_des_w[0:min(len(dur_asc_w), len(dur_des_w))])
    asymmetry_rem = np.log10(dur_asc_r[0:min(len(dur_asc_r), len(dur_des_r))] /dur_des_r[0:min(len(dur_asc_r), len(dur_des_r))])
    
    if isWT == 1:
        asym_wake_wt.extend(asymmetry_wake)
        asym_rem_wt.extend(asymmetry_rem)
    else: 
        asym_wake_ko.extend(asymmetry_wake)
        asym_rem_ko.extend(asymmetry_rem)
    
#%% Plotting

bins = np.linspace(-1, 1, 50)

asym_wake_wt_counts,_ = np.histogram(asym_wake_wt, bins)     
p_wake_wt = asym_wake_wt_counts/sum(asym_wake_wt_counts)

asym_wake_ko_counts,_ = np.histogram(asym_wake_ko, bins)     
p_wake_ko = asym_wake_ko_counts/sum(asym_wake_ko_counts)

asym_rem_wt_counts,_ = np.histogram(asym_rem_wt, bins)     
p_rem_wt = asym_rem_wt_counts/sum(asym_rem_wt_counts)

asym_rem_ko_counts,_ = np.histogram(asym_rem_ko, bins)     
p_rem_ko = asym_rem_ko_counts/sum(asym_rem_ko_counts)

plt.figure()
plt.suptitle('Theta Asymmetry (Wake)')
plt.subplot(121)
plt.title('WT')
plt.stairs(p_wake_wt, bins, color = 'royalblue')
plt.axvline(0, color = 'silver', linestyle = '--')
plt.xlim([-1, 1])
plt.ylabel('Fraction of theta cycles')
plt.xlabel('Asymmetry Index')
plt.gca().set_box_aspect(1)
plt.subplot(122)
plt.title('KO')
plt.stairs(p_wake_ko, bins, color = 'indianred')
plt.axvline(0, color = 'silver', linestyle = '--')
plt.xlim([-1, 1])
plt.xlabel('Asymmetry Index')
plt.ylabel('Fraction of theta cycles')
plt.gca().set_box_aspect(1)

plt.figure()
plt.suptitle('Theta Asymmetry (REM)')
plt.subplot(121)
plt.title('WT')
plt.stairs(p_rem_wt, bins, color = 'royalblue')
plt.axvline(0, color = 'silver', linestyle = '--')
plt.xlim([-1, 1])
plt.ylabel('% theta cycles')
plt.xlabel('Asymmetry Index')
plt.gca().set_box_aspect(1)
plt.subplot(122)
plt.title('KO')
plt.stairs(p_rem_ko, bins, color = 'indianred')
plt.axvline(0, color = 'silver', linestyle = '--')
plt.xlim([-1, 1])
plt.xlabel('Asymmetry Index')
plt.ylabel('Fraction of theta cycles')
plt.gca().set_box_aspect(1)

    
    