#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 13:19:12 2023

@author: dhruv
"""

import numpy as np 
import pandas as pd 
import os, sys
import matplotlib.pyplot as plt 
import nwbmatic as ntm
import pynapple as nap
import pynacollada as pyna
import pickle
from scipy.signal import hilbert, fftconvolve
from scipy.stats import circmean
from pingouin import circ_r
import matplotlib.colors as colors
from matplotlib.backends.backend_pdf import PdfPages    

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

def multipage(filename, figs=None, dpi=200):
    pp = PdfPages(filename)
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()
    
def smoothAngularTuningCurves(tuning_curves, sigma=2):
    from scipy.ndimage import gaussian_filter1d
    
    tmp = np.concatenate((tuning_curves.values, tuning_curves.values, tuning_curves.values))
    tmp = gaussian_filter1d(tmp, sigma=sigma, axis=0)

    return pd.DataFrame(index = tuning_curves.index,
        data = tmp[tuning_curves.shape[0]:tuning_curves.shape[0]*2], 
        columns = tuning_curves.columns
        )
    
    
#%% 

# data_directory = '/media/dhruv/Expansion/Processed'
data_directory = '/media/adrien/Expansion/Processed'
datasets = np.genfromtxt(os.path.join(data_directory,'dataset_DM.list'), delimiter = '\n', dtype = str, comments = '#')
# datasets = np.genfromtxt(os.path.join(data_directory,'dataset_test.list'), delimiter = '\n', dtype = str, comments = '#')
ripplechannels = np.genfromtxt(os.path.join(data_directory,'ripplechannel.list'), delimiter = '\n', dtype = str, comments = '#')

fs = 1250

darr_wt_wake = np.zeros((len(datasets),40,100))
darr_wt_rem = np.zeros((len(datasets),40,100))

darr_ko_wake = np.zeros((len(datasets),40,100))
darr_ko_rem = np.zeros((len(datasets),40,100))


for r,s in enumerate(datasets):
    print(s)
    name = s.split('-')[0]
    path = os.path.join(data_directory, s)
    
    data = ntm.load_session(path, 'neurosuite')
    data.load_neurosuite_xml(path)
    epochs = data.epochs
    position = data.position
    
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
        
#%% Load spikes 

    sp2 = np.load(os.path.join(path, 'spikedata_0.55.npz'), allow_pickle = True)
    time_support = nap.IntervalSet(sp2['start'], sp2['end'])
    tsd = nap.Tsd(t=sp2['t'], d=sp2['index'], time_support = time_support)
    spikes = tsd.to_tsgroup()
    spikes.set_info(group = sp2['group'], location = sp2['location'], celltype = sp2['celltype'], tr2pk = sp2['tr2pk'])
            
#%% 
    
    spikes_by_celltype = spikes.getby_category('celltype')
    if 'pyr' in spikes._metadata['celltype'].values:
        pyr = spikes_by_celltype['pyr']
    else: pyr = []
    
    if 'fs' in spikes._metadata['celltype'].values:
        pv = spikes_by_celltype['fs']
    else: pv = []
    
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
    
    lfp_wake = lfp.restrict(moving_ep)
    # lfp_wake = lfp.restrict(rem_ep)   

    lfp_filt_theta_wake = pyna.eeg_processing.bandpass_filter(lfp_wake, 30, 150, 1250)
    # lfp_filt_theta_wake = pyna.eeg_processing.bandpass_filter(lfp_wake, 6, 9, 1250)
    
    # lfp_filt_theta_rem = pyna.eeg_processing.bandpass_filter(lfp_rem, 30, 150, 1250)
        
    h_wake = nap.Tsd(t = lfp_filt_theta_wake.index.values, d = hilbert(lfp_filt_theta_wake))
    # h_rem = nap.Tsd(t = lfp_filt_theta_rem.index.values, d = hilbert(lfp_filt_theta_rem))
    
    phase_wake = nap.Tsd(t = lfp_filt_theta_wake.index.values, d = (np.angle(h_wake.values) + 2 * np.pi) % (2 * np.pi))
    # phase_rem = nap.Tsd(t = lfp_filt_theta_rem.index.values, d = (np.angle(h_rem.values) + 2 * np.pi) % (2 * np.pi))
  
#%% 
    
    if len(pv) > 0 and len(pyr) > 0:      
        phasepref_wake_pyr = nap.compute_1d_tuning_curves(pyr, phase_wake, 40, moving_ep) 
        phasepref_wake_pyr = smoothAngularTuningCurves(phasepref_wake_pyr, sigma=3)
        
        phasepref_wake_pv = nap.compute_1d_tuning_curves(pv, phase_wake, 40, moving_ep) 
        phasepref_wake_pv = smoothAngularTuningCurves(phasepref_wake_pv, sigma=3)
    
#%% Session spectra and tuning curves

        plt.figure()
        plt.title(s + '_wake_pyr')
        for i,n in enumerate(pyr):
            plt.subplot(9,8,n+1, projection='polar')
            plt.plot(phasepref_wake_pyr[n], color = 'b')        
            
        plt.figure()
        plt.title(s + '_wake_pv')
        for i,n in enumerate(pv):
            plt.subplot(9,8,n+1, projection='polar')
            plt.plot(phasepref_wake_pv[n], color = 'r')   
            
    del pyr, pv

multipage(data_directory + '/' + 'GammaPhaseplots_wake.pdf', dpi=250)

    # plt.figure()
    # plt.title(s)
    # # plt.imshow(tmp2.T, aspect = 'auto', interpolation='bilinear', origin = 'lower', extent = [0, 360, 30, 150], cmap = 'seismic')
    # # plt.imshow(tmp2.T, aspect = 'auto', interpolation='none', origin = 'lower', extent = [0, 360, 30, 150], cmap = 'seismic')
    # plt.imshow(phasepref_wake.T, aspect = 'auto', interpolation='bilinear', origin = 'lower', extent = [0, 360, 30, 150], cmap = 'seismic')
    # plt.xlabel('Theta phase (deg)')
    # plt.ylabel('Frequency (Hz)')
    # plt.colorbar()

#%% Genotype 

    # if isWT == 1: 
    #     # all_pspec_z_wt = pd.concat((phasepref_wake, all_pspec_z_wt))
    #     darr_wt_wake[r,:,:] = phasepref_wake
    #     # darr_wt_rem[r,:,:] = phasepref_rem
    # else:     
    #     # all_pspec_z_ko = pd.concat((phasepref_wake, all_pspec_z_ko))
    #     darr_ko_wake[r,:,:] = phasepref_wake
    #     # darr_ko_rem[r,:,:] = phasepref_rem
    
#%% 

# specgram_z_wake_wt = np.mean(darr_wt_wake, axis = 0)
# specgram_z_wake_ko = np.mean(darr_ko_wake, axis = 0)

# # specgram_z_rem_wt = np.mean(darr_wt_rem, axis = 0)
# # specgram_z_rem_ko = np.mean(darr_ko_rem, axis = 0)

# norm = colors.TwoSlopeNorm(vmin = -0.007, vcenter = 0, vmax = 0.007)

# plt.figure()
# # plt.suptitle('Z-scored spectrogram')
# # plt.subplot(121)
# plt.title('WT (wake)')
# plt.imshow(specgram_z_wake_wt.T, aspect = 'auto', interpolation='bilinear', origin = 'lower', extent = [0, 360, 30, 150], cmap = 'seismic', norm = norm)
# plt.xlabel('Gamma phase (deg)')
# plt.ylabel('Frequency (Hz)')
# plt.colorbar()
# plt.gca().set_box_aspect(1)

# plt.figure()
# # plt.subplot(122)
# plt.title('KO (wake)')
# plt.imshow(specgram_z_wake_ko.T, aspect = 'auto', interpolation='bilinear', origin = 'lower', extent = [0, 360, 30, 150], cmap = 'seismic', norm = norm)
# plt.xlabel('Gamma phase (deg)')
# plt.ylabel('Frequency (Hz)')
# plt.colorbar()
# plt.gca().set_box_aspect(1)


# plt.figure()
# # plt.suptitle('Z-scored spectrogram')
# # plt.subplot(121)
# plt.title('WT (REM)')
# plt.imshow(specgram_z_rem_wt.T, aspect = 'auto', interpolation='bilinear', origin = 'lower', extent = [0, 360, 30, 150], cmap = 'seismic')
# plt.xlabel('Theta phase (deg)')
# plt.ylabel('Frequency (Hz)')
# plt.colorbar()
# plt.gca().set_box_aspect(1)

# plt.figure()
# # plt.subplot(122)
# plt.title('KO (REM)')
# plt.imshow(specgram_z_rem_ko.T, aspect = 'auto', interpolation='bilinear', origin = 'lower', extent = [0, 360, 30, 150], cmap = 'seismic')
# plt.xlabel('Theta phase (deg)')
# plt.ylabel('Frequency (Hz)')
# plt.colorbar()
# plt.gca().set_box_aspect(1)


#%%     

    # bins = np.linspace(0, 2*np.pi, 40)    
    # widths = np.diff(bins)
       
    # if len(pv) > 0 and len(pyr) > 0:        
        
    #     spikephase_wake_ex = {}
    #     plt.figure() 
    #     plt.suptitle(s + ' EX cells')
        
    #     cmeans_pyr = []
    #     vlens_pyr = []
    
    #     for i, j in enumerate(pyr.index):
    #         spikephase_wake_ex[j] = pyr[j].value_from(phase_wake) 
    #         c = circmean(spikephase_wake_ex[j])
    #         cmeans_pyr.append(c)
    #         veclength = circ_r(spikephase_wake_ex[j])
    #         vlens_pyr.append(veclength)
    #         n, bins = np.histogram(spikephase_wake_ex[j], bins)
    #         area = n / spikephase_wake_ex[j].size
    #         radius = (area/np.pi) ** .5
            
    #         # plt.subplot(10, 6, i+1)
    #         # plt.hist(spikephase_wake[j])
            
    #         ax = plt.subplot(10,6,i+1, projection='polar')
    #         ax.bar(bins[:-1], radius, width=widths, color = 'lightsteelblue')
    #         ax.set_yticks([])
       
    
            
    #     spikephase_wake_fs = {}
    #     plt.figure() 
    #     plt.suptitle(s + ' FS cells')
        
    #     cmeans_fs = []
    #     vlens_fs = []
    
    #     for i, j in enumerate(pv.index):
    #         spikephase_wake_fs[j] = pv[j].value_from(phase_wake) 
    #         c = circmean(spikephase_wake_fs[j])
    #         cmeans_fs.append(c)
    #         veclength = circ_r(spikephase_wake_fs[j])
    #         vlens_fs.append(veclength)
    #         n, bins = np.histogram(spikephase_wake_fs[j], bins)
    #         area = n / spikephase_wake_fs[j].size
    #         radius = (area/np.pi) ** .5
            
    #         # plt.subplot(10, 6, i+1)
    #         # plt.hist(spikephase_wake[j])
            
    #         ax = plt.subplot(10,6,i+1, projection='polar')
    #         ax.bar(bins[:-1], radius, width=widths, color = 'lightcoral')
    #         ax.set_yticks([])
               
#%% 
    
    # multipage(data_directory + '/' + 'Phaseplots.pdf', dpi=250)
        
        
