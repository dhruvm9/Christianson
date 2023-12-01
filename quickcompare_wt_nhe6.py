#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 15:56:11 2023

@author: dhruv
"""

import numpy as np 
import pandas as pd 
import scipy.io
import os, sys
import time 
import matplotlib.pyplot as plt 
import pynapple as nap
import seaborn as sns
from scipy.fft import fft, ifft
from scipy.stats import wilcoxon, pearsonr
from scipy.signal import hilbert, fftconvolve
import matplotlib.cm as cm
import matplotlib.colors as colors
import math 
import pickle

#%%
   
data_directory = '/media/DataDhruv/Recordings/Christianson/noExplo'

datasets = np.genfromtxt(os.path.join(data_directory,'dataset_DM.list'), delimiter = '\n', dtype = str, comments = '#')

wt = 'B2607-230724_230724_120138'
nhe6 = 'B2608-230725_230725_102409'
   
wtpath = os.path.join(data_directory, wt)
nhe6path = os.path.join(data_directory, nhe6)

fs = 1250
ripplechannel = 5

lfp_wt = nap.load_eeg(wtpath + '/' + wt + '.eeg', channel = [0,1,2,3,4,5], n_channels = 6, frequency = fs)
lfp_nhe6 = nap.load_eeg(nhe6path + '/' + nhe6 + '.eeg', channel = [0,1,2,3,4,5], n_channels = 6, frequency = fs)

file = os.path.join(wtpath + '/' + wt + '.sws.evt')
if os.path.exists(file):
    tmp = np.genfromtxt(file)[:,0]
    tmp = tmp.reshape(len(tmp)//2,2)/1000
    sws_ep_wt = nap.IntervalSet(start = tmp[:,0], end = tmp[:,1], time_units = 's')
    
file = os.path.join(wtpath + '/' + wt + '.rem.evt')
if os.path.exists(file):
    tmp = np.genfromtxt(file)[:,0]
    tmp = tmp.reshape(len(tmp)//2,2)/1000
    rem_ep_wt = nap.IntervalSet(start = tmp[:,0], end = tmp[:,1], time_units = 's')

file = os.path.join(nhe6path + '/' + nhe6 + '.sws.evt')
if os.path.exists(file):
    tmp = np.genfromtxt(file)[:,0]
    tmp = tmp.reshape(len(tmp)//2,2)/1000
    sws_ep_nhe6 = nap.IntervalSet(start = tmp[:,0], end = tmp[:,1], time_units = 's')
    
file = os.path.join(nhe6path + '/' + nhe6 + '.rem.evt')
if os.path.exists(file):
    tmp = np.genfromtxt(file)[:,0]
    tmp = tmp.reshape(len(tmp)//2,2)/1000
    rem_ep_nhe6 = nap.IntervalSet(start = tmp[:,0], end = tmp[:,1], time_units = 's')


lfpsig_wt = lfp_wt[ripplechannel]  
lfpsig_nhe6 = lfp_nhe6[ripplechannel]  
    
   
#%%         
 
lfp_sws_wt = lfpsig_wt.restrict(sws_ep_wt)
lfp_rem_wt = lfpsig_wt.restrict(rem_ep_wt)

lfp_sws_nhe6 = lfpsig_nhe6.restrict(sws_ep_nhe6)
lfp_rem_nhe6 = lfpsig_nhe6.restrict(rem_ep_nhe6)

plt.figure()
plt.title('NREM')
f, Pxx = scipy.signal.welch(lfp_sws_wt, fs = fs)
plt.semilogx(f[f<=100], 10*np.log10(Pxx[f<=100]), 'o-', label = 'WT')
f, Pxx = scipy.signal.welch(lfp_sws_nhe6, fs = fs)
plt.semilogx(f[f<=100], 10*np.log10(Pxx[f<=100]), 'o-', label = 'NHE6')
plt.xlabel('Freq (Hz)')
plt.ylabel('Power spectral density (dB/Hz)')
plt.legend(loc = 'upper right')
plt.grid(True)

plt.figure()
plt.title('REM')
f, Pxx = scipy.signal.welch(lfp_rem_wt, fs = fs)
plt.semilogx(f[f<=100], 10*np.log10(Pxx[f<=100]), 'o-', label = 'WT')
f, Pxx = scipy.signal.welch(lfp_rem_nhe6, fs = fs)
plt.semilogx(f[f<=100], 10*np.log10(Pxx[f<=100]), 'o-', label = 'NHE6')
plt.xlabel('Freq (Hz)')
plt.ylabel('Power spectral density (dB/Hz)')
plt.legend(loc = 'upper right')
plt.grid(True)

      
plt.figure()
plt.suptitle('NREM')
plt.subplot(211)
plt.title('WT')
plt.specgram(lfp_sws_wt, Fs = fs, cmap = 'jet', vmin = 0, vmax = 50)
plt.subplot(212)
plt.title('NHE6')
plt.specgram(lfp_sws_nhe6, Fs = fs, cmap = 'jet', vmin = 0, vmax = 50)

plt.figure()
plt.suptitle('REM')
plt.subplot(211)
plt.title('WT')
plt.specgram(lfp_rem_wt, Fs = fs, cmap = 'jet', vmin = 0, vmax = 50)
plt.subplot(212)
plt.title('NHE6')
plt.specgram(lfp_rem_nhe6, Fs = fs, cmap = 'jet', vmin = 0, vmax = 50)


   
 

    
