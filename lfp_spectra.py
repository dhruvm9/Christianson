#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 16:12:26 2023

@author: dhruv
"""
import numpy as np 
import pandas as pd
import nwbmatic as ntm
import pynapple as nap 
import scipy.io
import os, sys
import seaborn as sns
import matplotlib.pyplot as plt 
import pynacollada as pyna
from scipy.stats import mannwhitneyu
from scipy.signal import hilbert

#%% 

# data_directory = '/media/dhruv/Expansion/Processed'
data_directory = '/media/dhruv/Expansion/Processed/CA3'
datasets = np.genfromtxt(os.path.join(data_directory,'dataset_DM.list'), delimiter = '\n', dtype = str, comments = '#')
ripplechannels = np.genfromtxt(os.path.join(data_directory,'ripplechannel.list'), delimiter = '\n', dtype = str, comments = '#')

fs = 1250

PSD_wake_wt = pd.DataFrame()
PSD_sws_wt = pd.DataFrame()
PSD_rem_wt = pd.DataFrame()

PSD_wake_ko = pd.DataFrame()
PSD_sws_ko = pd.DataFrame()
PSD_rem_ko = pd.DataFrame()

for r,s in enumerate(datasets):
    print(s)
    name = s.split('-')[0]
    path = os.path.join(data_directory, s)
    
    data = ntm.load_session(path, 'neurosuite')
    data.load_neurosuite_xml(path)
    epochs = data.epochs
    position = data.position
    
    if name == 'B2613' or name == 'B2618' or name == 'B2627' or name == 'B2628':
        isWT = 0
    else: isWT = 1 
    
    lfp = nap.load_eeg(path + '/' + s + '.eeg', channel = int(ripplechannels[r]), n_channels = 32, frequency = fs)
    
    file = os.path.join(path, s +'.sws.evt')
    sws_ep = data.read_neuroscope_intervals(name = 'SWS', path2file = file)
    
    file = os.path.join(path, s +'.rem.evt')
    rem_ep = data.read_neuroscope_intervals(name = 'REM', path2file = file)
    
#%% Moving epoch 

    speedbinsize = np.diff(position.index.values)[0]
    
    time_bins = np.arange(position.index[0], position.index[-1] + speedbinsize, speedbinsize)
    index = np.digitize(position.index.values, time_bins)
    tmp = position.as_dataframe().groupby(index).mean()
    tmp.index = time_bins[np.unique(index)-1]+(speedbinsize)/2
    distance = np.sqrt(np.power(np.diff(tmp['x']), 2) + np.power(np.diff(tmp['z']), 2)) * 100 #in cm
    speed = pd.Series(index = tmp.index.values[0:-1]+ speedbinsize/2, data = distance/speedbinsize) # in cm/s
    speed2 = speed.rolling(window = 25, win_type='gaussian', center=True, min_periods=1).mean(std=10) #Smooth over 200ms 
    speed2 = nap.Tsd(speed2)
    
    moving_ep = nap.IntervalSet(speed2.threshold(2).time_support) #Epochs in which speed is > 2 cm/s
        

#%% Restrict LFP
    
    lfp_wake = lfp.restrict(moving_ep)
    lfp_w_z = scipy.stats.zscore(lfp_wake)
    
    lfp_sws = lfp.restrict(sws_ep)
    lfp_s_z = scipy.stats.zscore(lfp_sws)
    
    lfp_rem = lfp.restrict(rem_ep)
    lfp_r_z = scipy.stats.zscore(lfp_rem)

#%% Filter LFP and do power ratios

    # lfp_filt_delta = pyna.eeg_processing.bandpass_filter(lfp, 1, 4, 1250)
    # lfp_filt_theta = pyna.eeg_processing.bandpass_filter(lfp, 6, 9, 1250)
    # lfp_filt_gammalow = pyna.eeg_processing.bandpass_filter(lfp, 30, 50, 1250)
    # lfp_filt_gammahigh = pyna.eeg_processing.bandpass_filter(lfp, 70, 90, 1250)

    # delta_power = nap.Tsd(lfp_filt_delta.index.values, np.abs(hilbert(lfp_filt_delta.values)))
    # theta_power = nap.Tsd(lfp_filt_theta.index.values, np.abs(hilbert(lfp_filt_theta.values)))
    # gammalow_power = nap.Tsd(lfp_filt_gammalow.index.values, np.abs(hilbert(lfp_filt_gammalow.values)))
    # gammahigh_power = nap.Tsd(lfp_filt_gammalow.index.values, np.abs(hilbert(lfp_filt_gammahigh.values)))
    
    


#%% Power Spectral Density 
    
    freqs, P_wake = scipy.signal.welch(lfp_wake, fs = fs)
    _, P_sws = scipy.signal.welch(lfp_sws, fs = fs)
    _, P_rem = scipy.signal.welch(lfp_rem, fs = fs)
    
    freqs, P_wake = scipy.signal.welch(lfp_w_z, fs = fs)
    _, P_sws = scipy.signal.welch(lfp_s_z, fs = fs)
    _, P_rem = scipy.signal.welch(lfp_r_z, fs = fs)
   
        
    if isWT == 1:
        PSD_wake_wt = pd.concat([PSD_wake_wt, pd.Series(P_wake)], axis = 1)
        PSD_sws_wt = pd.concat([PSD_sws_wt, pd.Series(P_sws)], axis = 1)
        PSD_rem_wt = pd.concat([PSD_rem_wt, pd.Series(P_rem)], axis = 1)

    else: 
        PSD_wake_ko = pd.concat([PSD_wake_ko, pd.Series(P_wake)], axis = 1)
        PSD_sws_ko = pd.concat([PSD_sws_ko, pd.Series(P_sws)], axis = 1)
        PSD_rem_ko = pd.concat([PSD_rem_ko, pd.Series(P_rem)], axis = 1)
    
#%% Average spectrum 

# ix = np.where(freqs<=100)
ix = np.where(freqs<=300)

##Wake 
plt.figure()
plt.subplot(131)
plt.title('Wake')
plt.semilogx(freqs[ix], 10*np.log10(PSD_wake_wt.iloc[ix].mean(axis=1)), 'o-', label = 'WT', color = 'royalblue')
# err = 10*np.log10(PSD_wake_wt.iloc[ix].sem(axis=1))
# plt.fill_between(freqs[ix],
#                   10*np.log10(PSD_wake_wt.iloc[ix].mean(axis=1))-err, 
#                   10*np.log10(PSD_wake_wt.iloc[ix].mean(axis=1))+err, color = 'royalblue', alpha = 0.2)

plt.semilogx(freqs[ix], 10*np.log10(PSD_wake_ko.iloc[ix].mean(axis=1)), 'o-', label = 'KO', color = 'indianred')
# err = 10*np.log10(PSD_wake_ko.iloc[ix].sem(axis=1))
# plt.fill_between(freqs[ix],
#                  10*np.log10(PSD_wake_ko.iloc[ix].mean(axis=1))-err, 
#                  10*np.log10(PSD_wake_ko.iloc[ix].mean(axis=1))+err, color = 'indianred', alpha = 0.2)
plt.xlabel('Freq (Hz)')
plt.ylabel('Power spectral density (dB/Hz)')
# plt.ylim([20, 60])
plt.ylim([-50, -10])
plt.legend(loc = 'upper right')
plt.grid(True)

##NREM
plt.subplot(132)
plt.title('NREM')
plt.semilogx(freqs[ix], 10*np.log10(PSD_sws_wt.iloc[ix].mean(axis=1)), 'o-', label = 'WT', color = 'royalblue')
plt.semilogx(freqs[ix], 10*np.log10(PSD_sws_ko.iloc[ix].mean(axis=1)), 'o-', label = 'KO', color = 'indianred')
plt.xlabel('Freq (Hz)')
plt.ylabel('Power spectral density (dB/Hz)')
# plt.ylim([20, 60])
plt.ylim([-50, -10])
plt.legend(loc = 'upper right')
plt.grid(True)

##REM
plt.subplot(133)
plt.title('REM')
plt.semilogx(freqs[ix], 10*np.log10(PSD_rem_wt.iloc[ix].mean(axis=1)), 'o-', label = 'WT', color = 'royalblue')
plt.semilogx(freqs[ix], 10*np.log10(PSD_rem_ko.iloc[ix].mean(axis=1)), 'o-', label = 'KO', color = 'indianred')
plt.xlabel('Freq (Hz)')
# plt.ylim([20, 60])
plt.ylim([-50, -10])
plt.ylabel('Power spectral density (dB/Hz)')
plt.legend(loc = 'upper right')
plt.grid(True)

