#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 12:22:55 2024

@author: dhruv
"""

############## CHECK IF MINMAX FUNCTION FOR PHASE ##################


import numpy as np 
import pandas as pd 
import os, sys
import matplotlib.pyplot as plt 
import nwbmatic as ntm
import pynapple as nap
import pynacollada as pyna
import pickle
from scipy.signal import hilbert, fftconvolve
from matplotlib.backends.backend_pdf import PdfPages    
import matplotlib.colors as colors

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


def rounder(values):
    def f(x):
        idx = np.argmin(np.abs(values - x))
        return values[idx]
    return np.frompyfunc(f, 1, 1)


#%% 

# data_directory = '/media/dhruv/Expansion/Processed'
data_directory = '/media/adrien/Expansion/Processed'
datasets = np.genfromtxt(os.path.join(data_directory,'dataset_DM.list'), delimiter = '\n', dtype = str, comments = '#')
# datasets = np.genfromtxt(os.path.join(data_directory,'dataset_test.list'), delimiter = '\n', dtype = str, comments = '#')
ripplechannels = np.genfromtxt(os.path.join(data_directory,'ripplechannel.list'), delimiter = '\n', dtype = str, comments = '#')

fs = 1250

darr_wt = np.zeros((len(datasets),40,100))
darr_ko = np.zeros((len(datasets),40,100))

all_pspec_z_wt = pd.DataFrame()
all_pspec_z_ko = pd.DataFrame()

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
    
    file = os.path.join(path, s +'.evt.py.rip')
    rip_ep = data.read_neuroscope_intervals(name = 'rip', path2file = file)
    
    with open(os.path.join(path, 'riptsd.pickle'), 'rb') as pickle_file:
        rip_tsd = pickle.load(pickle_file)
        
#%% Load spikes 

    sp2 = np.load(os.path.join(path, 'spikedata_0.55.npz'), allow_pickle = True)
    time_support = nap.IntervalSet(sp2['start'], sp2['end'])
    tsd = nap.Tsd(t=sp2['t'], d=sp2['index'], time_support = time_support)
    spikes = tsd.to_tsgroup()
    spikes.set_info(group = sp2['group'], location = sp2['location'], celltype = sp2['celltype'], tr2pk = sp2['tr2pk'])
    
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

    downsample = 1
    binned_angles = np.linspace(0, 360, 40)

    # lfp_wake = lfp.restrict(moving_ep)
    lfp_wake = lfp.restrict(rem_ep)
    # lfp_wake = lfp
    
    lfp_filt_theta_wake = pyna.eeg_processing.bandpass_filter(lfp_wake, 6, 9, 1250)
    lfp_filt_gamma_wake = pyna.eeg_processing.bandpass_filter(lfp_wake, 30, 150, 1250)
           
    h_wake = nap.Tsd(t = lfp_filt_theta_wake.index.values, d = hilbert(lfp_filt_theta_wake))
        
    phase_wake = nap.Tsd(t = lfp_filt_theta_wake.index.values, d = (np.angle(h_wake.values, deg=True) + 360) % (360))
    phase_wake = phase_wake[::downsample]
            
    # tmp = rounder(binned_angles)(phase_wake.values)
    
    # binned_phase = nap.Tsd(t = phase_wake.index.values, d = tmp)
    
    
#%%

    fmin = 30
    fmax = 150
    nfreqs = 100
    ncyc = 9
    si = 1/fs
        
    freqs = np.logspace(np.log10(fmin),np.log10(fmax),nfreqs)
    
    nfreqs = len(freqs)
    
    wavespec = pd.DataFrame(index = lfp_wake.index.values[::downsample], columns = freqs, dtype = np.float64 )
    powerspec = pd.DataFrame(index = lfp_wake.index.values[::downsample],columns = freqs, dtype = np.float64)
        
    for f in range(len(freqs)):
         wavelet = MorletWavelet(freqs[f],ncyc,si)
         tmpspec = fftconvolve(lfp_wake.values, wavelet, mode = 'same')
         wavespec[freqs[f]] = tmpspec [::downsample]
         temppower = abs(wavespec[freqs[f]]) #**2
         
         ### Z-score
         powerspec[freqs[f]] = ((temppower.values - temppower.mean()) / temppower.std()) #(temppower.values/np.median(temppower.values)) 
                  
         ###Modified Z-score
         # absdiff = abs(temppower.values - temppower.median())        
         # mad = np.median(absdiff)
         # powerspec[freqs[f]] = (0.6745* (temppower.values - temppower.mean())) / mad
    
    gamma_power = nap.TsdFrame(powerspec)
    
    ###Change the EPOCH
    # phasepref_wake = nap.compute_1d_tuning_curves_continuous(gamma_power, phase_wake, 40, moving_ep, minmax = (0, 2 * np.pi))
    phasepref_wake = nap.compute_1d_tuning_curves_continuous(gamma_power, phase_wake, 40, rem_ep, minmax = (0, 2 * np.pi))
    
    # tmp2 = powerspec.groupby(powerspec.index).mean()
    
#%% 
    

    # norm = colors.TwoSlopeNorm(vmin = -1, vcenter = 0, vmax = 1)
    if isWT == 0: 
    
        plt.figure()
        plt.title(s)
        # plt.imshow(tmp2.T, aspect = 'auto', interpolation='bilinear', origin = 'lower', extent = [0, 360, 30, 150], cmap = 'seismic')
        # plt.imshow(tmp2.T, aspect = 'auto', interpolation='none', origin = 'lower', extent = [0, 360, 30, 150], cmap = 'seismic')
        plt.imshow(phasepref_wake.T, aspect = 'auto', interpolation='bilinear', origin = 'lower', extent = [0, 360, 30, 150], cmap = 'seismic')
        plt.xlabel('Theta phase (deg)')
        plt.ylabel('Frequency (Hz)')
        plt.colorbar()
    
#%%

    if isWT == 1: 
        # all_pspec_z_wt = pd.concat((phasepref_wake, all_pspec_z_wt))
        darr_wt[r,:,:] = phasepref_wake
    else:     
        # all_pspec_z_ko = pd.concat((phasepref_wake, all_pspec_z_ko))
        darr_ko[r,:,:] = phasepref_wake
        
#%% 
        
# specgram_z_wt = all_pspec_z_wt.groupby(all_pspec_z_wt.index).mean()
# specgram_z_ko = all_pspec_z_ko.groupby(all_pspec_z_ko.index).mean()

specgram_z_wt = np.mean(darr_wt, axis = 0)
specgram_z_ko = np.mean(darr_ko, axis = 0)

norm = colors.TwoSlopeNorm(vmin = -0.04, vcenter = 0, vmax = 0.04)

plt.figure()
# plt.suptitle('Z-scored spectrogram')
# plt.subplot(121)
plt.title('WT')
plt.imshow(specgram_z_wt.T, aspect = 'auto', interpolation='bilinear', origin = 'lower', extent = [0, 360, 30, 150], cmap = 'seismic', norm = norm)
plt.xlabel('Theta phase (deg)')
plt.ylabel('Frequency (Hz)')
plt.colorbar()
plt.gca().set_box_aspect(1)

plt.figure()
# plt.subplot(122)
plt.title('KO')
plt.imshow(specgram_z_ko.T, aspect = 'auto', interpolation='bilinear', origin = 'lower', extent = [0, 360, 30, 150], cmap = 'seismic', norm = norm)
plt.xlabel('Theta phase (deg)')
plt.ylabel('Frequency (Hz)')
plt.colorbar()
plt.gca().set_box_aspect(1)

#%% 

# import pickle 
# specgram_z_wt.to_pickle(data_directory + '/theta_wake_specgram_z_wt.pkl')
# specgram_z_ko.to_pickle(data_directory + '/theta_wake_specgram_z_ko.pkl')

