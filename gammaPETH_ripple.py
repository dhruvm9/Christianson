#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 10:23:23 2024

@author: adrien
"""

import numpy as np 
import pandas as pd 
import os, sys
import matplotlib.pyplot as plt 
import nwbmatic as ntm
import pynapple as nap
import pynacollada as pyna
import pickle
from scipy.signal import hilbert
import matplotlib.cm as cm

#%% 

data_directory = '/media/adrien/Expansion/Processed'
datasets = np.genfromtxt(os.path.join(data_directory,'dataset_DM.list'), delimiter = '\n', dtype = str, comments = '#')
ripplechannels = np.genfromtxt(os.path.join(data_directory,'ripplechannel.list'), delimiter = '\n', dtype = str, comments = '#')

fs = 1250

for r,s in enumerate(datasets):
    print(s)
    name = s.split('-')[0]
    path = os.path.join(data_directory, s)
    
    data = ntm.load_session(path, 'neurosuite')
    data.load_neurosuite_xml(path)
    epochs = data.epochs
    
    if name == 'B2613' or name == 'B2618':
        isWT = 0
    else: isWT = 1 
    
    lfp = nap.load_eeg(path + '/' + s + '.eeg', channel = int(ripplechannels[r]), n_channels = 32, frequency = fs)
    
    file = os.path.join(path, s +'.sws.evt')
    sws_ep = data.read_neuroscope_intervals(name = 'SWS', path2file = file)
    
    file = os.path.join(path, s +'.rem.evt')
    rem_ep = data.read_neuroscope_intervals(name = 'REM', path2file = file)
    
    file = os.path.join(path, s +'.evt.py.rip')
    rip_ep = data.read_neuroscope_intervals(name = 'rip', path2file = file)
    
    with open(os.path.join(path, 'riptsd.pickle'), 'rb') as pickle_file:
        rip_tsd = pickle.load(pickle_file)
        
#%% 
    
    fmin = 10
    fmax = 300
    nfreqs = 100
    ncyc = 3 #5
    si = 1/fs
           
    freqs = np.logspace(np.log10(fmin),np.log10(fmax),nfreqs)
    
    nfreqs = len(freqs)
    
    wavespec = pd.DataFrame(index = lfp.index.values, columns = freqs)
    powerspec = pd.DataFrame(index = lfp.index.values, columns = freqs)
        
    for f in range(len(freqs)):
         wavelet = MorletWavelet(freqs[f],ncyc,si)
         tmpspec = fftconvolve(lfp.values, wavelet, mode = 'same')
         wavespec[freqs[f]] = tmpspec
         temppower = abs(wavespec[freqs[f]]) #**2
         powerspec[freqs[f]] =  temppower #(temppower.values/np.median(temppower.values))
     


