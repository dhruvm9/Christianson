#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 17:19:59 2023

@author: dhruv
"""

import numpy as np 
import pandas as pd 
import os, sys
import matplotlib.pyplot as plt 
import nwbmatic as ntm
import pynapple as nap
import pickle
import matplotlib.cm as cm
import matplotlib.colors as colors
from scipy.signal import fftconvolve

#%% 

def MorletWavelet(f, ncyc, si):
    
    #Parameters
    s = ncyc/(2*np.pi*f)    #SD of the gaussian
    tbound = (4*s);   #time bounds - at least 4SD on each side, 0 in center
    tbound = si*np.floor(tbound/si)
    t = np.arange(-tbound,tbound,si) #time
    
    #Wavelet
    sinusoid = np.exp(2*np.pi*f*t*-1j)
    gauss = np.exp(-(t**2)/(2*(s**2)))
    
    A = 1
    wavelet = A * sinusoid * gauss
    wavelet = wavelet / np.linalg.norm(wavelet)
    return wavelet 

#%% 

data_directory = '/media/adrien/Expansion/Processed'
datasets = np.genfromtxt(os.path.join(data_directory,'dataset_DM.list'), delimiter = '\n', dtype = str, comments = '#')
ripplechannels = np.genfromtxt(os.path.join(data_directory,'ripplechannel.list'), delimiter = '\n', dtype = str, comments = '#')

fs = 1250

all_pspec_median_wt = pd.DataFrame()
all_pspec_median_ko = pd.DataFrame()

all_pspec_z_wt = pd.DataFrame()
all_pspec_z_ko = pd.DataFrame()

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
     
    fmin = 100
    fmax = 300
    nfreqs = 100
    ncyc = 3 #5
    si = 1/fs
           
    freqs = np.logspace(np.log10(fmin),np.log10(fmax),nfreqs)
    
    nfreqs = len(freqs)
    
    wavespec = nap.TsdFrame(t = lfp.index.values, columns = freqs)
    powerspec = nap.TsdFrame(t = lfp.index.values, columns = freqs)
        
    for f in range(len(freqs)):
         wavelet = MorletWavelet(freqs[f],ncyc,si)
         tmpspec = fftconvolve(lfp.values, wavelet, mode = 'same')
         wavespec[freqs[f]] = tmpspec
         temppower = abs(wavespec[freqs[f]]) #**2
         powerspec[freqs[f]] =  temppower #(temppower.values/np.median(temppower.values))
         
#%%
       
    rip = nap.Tsd(rip_ep['start'].values)
      
    realigned = powerspec.index[powerspec.index.get_indexer(rip.index.values, method='nearest')]
    
    pspec_median = pd.DataFrame()
    pspec_z = pd.DataFrame()
    
    for i in range(len(powerspec.columns)):
        tmp = nap.compute_perievent(powerspec[powerspec.columns[i]], nap.Ts(realigned.values) , minmax = (-0.25, 0.25), time_unit = 's')
           
        peth_all = []
        for j in range(len(tmp)):
            peth_all.append(tmp[j].as_series())
            
        trials = pd.concat(peth_all, axis = 1, join = 'outer')
        
        z = ((trials - trials.mean()) / trials.std()).mean(axis = 1)    
        pspec_z[freqs[i]] = z
        
        mdn = (trials/trials.median()).mean(axis = 1)
        pspec_median[freqs[i]] = mdn
     
#%% By genotype

    if isWT == 1: 
        all_pspec_median_wt = pd.concat((pspec_median, all_pspec_median_wt))
        all_pspec_z_wt = pd.concat((pspec_z, all_pspec_z_wt))
   
    else: 
       all_pspec_median_ko = pd.concat((pspec_median, all_pspec_median_ko))
       all_pspec_z_ko = pd.concat((pspec_z, all_pspec_z_ko))
       
#%% Taking mean 

specgram_z_wt = all_pspec_z_wt.groupby(all_pspec_z_wt.index).mean()
specgram_z_ko = all_pspec_z_ko.groupby(all_pspec_z_ko.index).mean()

specgram_m_wt = all_pspec_median_wt.groupby(all_pspec_median_wt.index).mean()
specgram_m_ko = all_pspec_median_ko.groupby(all_pspec_median_ko.index).mean()

#%% Save 

# specgram_z_wt.to_pickle(data_directory + '/specgram_z_wt.pkl')
# specgram_z_ko.to_pickle(data_directory + '/specgram_z_ko.pkl')

# specgram_m_wt.to_pickle(data_directory + '/specgram_m_wt.pkl')
# specgram_m_ko.to_pickle(data_directory + '/specgram_m_ko.pkl')

#%% Plotting 

# ## Z-scored 

labels = [100, 150, 200, 250, 300]
# norm = colors.TwoSlopeNorm(vmin=specgram_z_wt[-0.2:0.2].values.min(),vcenter=0, vmax = specgram_z_wt[-0.2:0.2].values.max())
norm = colors.TwoSlopeNorm(vmin = -0.3, vcenter = 0, vmax = 2.2)
       
fig, ax = plt.subplots()
plt.title('Z-scored spectrogram (WT)')
cax = ax.imshow(specgram_z_wt[0:0.06].T, aspect = 'auto', cmap = 'jet', interpolation='bilinear', 
            origin = 'lower',
            extent = [specgram_z_wt[0:0.06].index.values[0], 
                      specgram_z_wt[0:0.06].index.values[-1],
                      np.log10(specgram_z_wt.columns[0]),
                      np.log10(specgram_z_wt.columns[-1])], 
            norm = norm
            )
plt.xlabel('Time from SWR (s)')
plt.xticks([0, 0.025, 0.06])
plt.ylabel('Freq (Hz)')
plt.yticks(np.log10(labels), labels = labels)
cbar = fig.colorbar(cax, label = 'Power (z)', ticks = [-0.3, 0, 2.2])
plt.axvline(0, color = 'k',linestyle = '--')
plt.gca().set_box_aspect(1)

       
fig, ax = plt.subplots()
plt.title('Z-scored spectrogram (KO)')
cax = ax.imshow(specgram_z_ko[0:0.06].T, aspect = 'auto', cmap = 'jet', interpolation='bilinear', 
            origin = 'lower',
            extent = [specgram_z_ko[0:0.06].index.values[0], 
                      specgram_z_ko[0:0.06].index.values[-1],
                      np.log10(specgram_z_ko.columns[0]),
                      np.log10(specgram_z_ko.columns[-1])], 
            norm = norm
            )
plt.xlabel('Time from SWR (s)')
plt.xticks([0, 0.025, 0.06])
plt.ylabel('Freq (Hz)')
plt.yticks(np.log10(labels), labels = labels)
cbar = fig.colorbar(cax, label = 'Power (z)', ticks = [-0.3, 0, 2.2])
plt.axvline(0, color = 'k',linestyle = '--')
plt.gca().set_box_aspect(1)


# ## Median Normalized

labels = [100, 150, 200, 250, 300]
# norm = colors.TwoSlopeNorm(vmin=specgram_m_wt[-0.2:0.2].values.min(), vmax = specgram_m_wt[-0.2:0.2].values.max())
norm = colors.TwoSlopeNorm(vmin = 1.05, vcenter = 3.25 , vmax = 5.45)
       
fig, ax = plt.subplots()
plt.title('Median normalized spectrogram (WT)')
cax = ax.imshow(specgram_m_wt[0:0.06].T, aspect = 'auto', cmap = 'jet', interpolation='bilinear', 
            origin = 'lower',
            extent = [specgram_m_wt[0:0.06].index.values[0], 
                      specgram_m_wt[0:0.06].index.values[-1],
                      np.log10(specgram_m_wt.columns[0]),
                      np.log10(specgram_m_wt.columns[-1])], 
            norm = norm
            )
plt.xlabel('Time from SWR (s)')
plt.xticks([0, 0.025, 0.06])
plt.ylabel('Freq (Hz)')
plt.yticks(np.log10(labels), labels = labels)
cbar = fig.colorbar(cax, label = 'Power (median normalized)')
plt.axvline(0, color = 'k',linestyle = '--')
plt.gca().set_box_aspect(1)


fig, ax = plt.subplots()
plt.title('Median normalized spectrogram (KO)')
cax = ax.imshow(specgram_m_ko[0:0.06].T, aspect = 'auto', cmap = 'jet', interpolation='bilinear', 
            origin = 'lower',
            extent = [specgram_m_ko[0:0.06].index.values[0], 
                      specgram_m_ko[0:0.06].index.values[-1],
                      np.log10(specgram_m_ko.columns[0]),
                      np.log10(specgram_m_ko.columns[-1])], 
            norm = norm
            )
plt.xlabel('Time from SWR (s)')
plt.xticks([0, 0.025, 0.06])
plt.ylabel('Freq (Hz)')
plt.yticks(np.log10(labels), labels = labels)
cbar = fig.colorbar(cax, label = 'Power (median normalized)')
plt.axvline(0, color = 'k',linestyle = '--')
plt.gca().set_box_aspect(1)
       
