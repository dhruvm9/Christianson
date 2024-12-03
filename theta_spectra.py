#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 12:41:04 2024

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

data_directory = '/media/dhruv/Expansion/Processed'
# data_directory = '/media/dhruv/Expansion/Processed/CA3'
datasets = np.genfromtxt(os.path.join(data_directory,'dataset_DM.list'), delimiter = '\n', dtype = str, comments = '#')
ripplechannels = np.genfromtxt(os.path.join(data_directory,'ripplechannel.list'), delimiter = '\n', dtype = str, comments = '#')

fs = 1250

PSD_wake_wt = pd.DataFrame()
PSD_sws_wt = pd.DataFrame()
PSD_rem_wt = pd.DataFrame()

PSD_wake_ko = pd.DataFrame()
PSD_sws_ko = pd.DataFrame()
PSD_rem_ko = pd.DataFrame()

peakfreq_wake_wt = []
peakfreq_wake_ko = []
peakfreq_rem_wt = []
peakfreq_rem_ko = []

for r,s in enumerate(datasets):
    print(s)
    name = s.split('-')[0]
    path = os.path.join(data_directory, s)
    
    data = ntm.load_session(path, 'neurosuite')
    data.load_neurosuite_xml(path)
    epochs = data.epochs
    position = data.position
    
    if name == 'B2613' or name == 'B2618' or name == 'B2627' or name == 'B2628' or name == 'B3805' or name == 'B3813':
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

#%% Power Spectral Density 
    
    freqs, P_wake = scipy.signal.welch(lfp_wake, fs = fs, nfft = 5000)
    _, P_sws = scipy.signal.welch(lfp_sws, fs = fs, nfft = 5000)
    _, P_rem = scipy.signal.welch(lfp_rem, fs = fs, nfft = 5000)
    
    freqs, P_wake = scipy.signal.welch(lfp_w_z, fs = fs, nfft = 5000)
    _, P_sws = scipy.signal.welch(lfp_s_z, fs = fs, nfft = 5000)
    _, P_rem = scipy.signal.welch(lfp_r_z, fs = fs, nfft = 5000)
    
    ix2 = np.where((freqs>=5) & (freqs <= 12))
    peakfreq_wake = freqs[ix2][np.argmax(P_wake[ix2])]
    peakfreq_rem = freqs[ix2][np.argmax(P_rem[ix2])]
           
            
    if isWT == 1:
        PSD_wake_wt = pd.concat([PSD_wake_wt, pd.Series(P_wake)], axis = 1)
        PSD_sws_wt = pd.concat([PSD_sws_wt, pd.Series(P_sws)], axis = 1)
        PSD_rem_wt = pd.concat([PSD_rem_wt, pd.Series(P_rem)], axis = 1)
        peakfreq_wake_wt.append(peakfreq_wake)
        peakfreq_rem_wt.append(peakfreq_rem)

    else: 
        PSD_wake_ko = pd.concat([PSD_wake_ko, pd.Series(P_wake)], axis = 1)
        PSD_sws_ko = pd.concat([PSD_sws_ko, pd.Series(P_sws)], axis = 1)
        PSD_rem_ko = pd.concat([PSD_rem_ko, pd.Series(P_rem)], axis = 1)
        peakfreq_wake_ko.append(peakfreq_wake)
        peakfreq_rem_ko.append(peakfreq_rem)


    # ix2 = np.where((freqs>=5) & (freqs <=12))
    # peakfreq = freqs[ix2][np.argmax(P_wake[ix2])]
           
    # if isWT == 1:
    #     PSD_rip_wt = pd.concat([PSD_rip_wt, pd.Series(P_xx)], axis = 1)
    #     peakfreq_wt.append(peakfreq)
        
    # else: 
    #     PSD_rip_ko = pd.concat([PSD_rip_ko, pd.Series(P_xx)], axis = 1)
    #     peakfreq_ko.append(peakfreq)


#%% Average spectrum 

# ix = np.where(freqs<=100)
ix = np.where(freqs<=15)

##Wake 
plt.figure()
plt.subplot(131)
plt.title('Wake')
plt.semilogx(freqs[ix], 10*np.log10(PSD_wake_wt.iloc[ix].mean(axis=1)), 'o-', label = 'WT', color = 'royalblue')
# plt.semilogx(freqs[ix], 10*np.log10(PSD_wake_wt.iloc[ix]), 'o-', label = 'WT', color = 'royalblue')
# plt.semilogx(freqs[ix], 10*np.log10(PSD_wake_wt.iloc[ix]), 'o-', label = 'WT', color = 'royalblue')
# err = 10*np.log10(PSD_wake_wt.iloc[ix].sem(axis=1))
# plt.fill_between(freqs[ix],
#                   10*np.log10(PSD_wake_wt.iloc[ix].mean(axis=1))-err, 
#                   10*np.log10(PSD_wake_wt.iloc[ix].mean(axis=1))+err, color = 'royalblue', alpha = 0.2)

plt.semilogx(freqs[ix], 10*np.log10(PSD_wake_ko.iloc[ix].mean(axis=1)), 'o-', label = 'KO', color = 'indianred')
# plt.semilogx(freqs[ix], 10*np.log10(PSD_wake_ko.iloc[ix]), 'o-', label = 'KO', color = 'indianred')
# plt.semilogx(freqs[ix], 10*np.log10(PSD_wake_ko.iloc[ix]), 'o-', label = 'KO', color = 'indianred')
# err = 10*np.log10(PSD_wake_ko.iloc[ix].sem(axis=1))
# plt.fill_between(freqs[ix],
#                  10*np.log10(PSD_wake_ko.iloc[ix].mean(axis=1))-err, 
#                  10*np.log10(PSD_wake_ko.iloc[ix].mean(axis=1))+err, color = 'indianred', alpha = 0.2)
plt.xlabel('Freq (Hz)')
plt.ylabel('Power spectral density (dB/Hz)')
# plt.ylim([20, 60])
# plt.ylim([-50, -10])
plt.ylim([-23, -11])
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
# plt.ylim([-50, -10])
plt.ylim([-23, -11])
plt.legend(loc = 'upper right')
plt.grid(True)

##REM
plt.subplot(133)
plt.title('REM')
plt.semilogx(freqs[ix], 10*np.log10(PSD_rem_wt.iloc[ix].mean(axis=1)), 'o-', label = 'WT', color = 'royalblue')
plt.semilogx(freqs[ix], 10*np.log10(PSD_rem_ko.iloc[ix].mean(axis=1)), 'o-', label = 'KO', color = 'indianred')
plt.xlabel('Freq (Hz)')
# plt.ylim([20, 60])
# plt.ylim([-50, -10])
plt.ylim([-23, -11])
plt.ylabel('Power spectral density (dB/Hz)')
plt.legend(loc = 'upper right')
plt.grid(True)

#%% 

wt = np.array(['WT' for x in range(len(peakfreq_wake_wt))])
ko = np.array(['KO' for x in range(len(peakfreq_wake_ko))])

genotype = np.hstack([wt, ko])

allpeaks = []
allpeaks.extend(peakfreq_wake_wt)
allpeaks.extend(peakfreq_wake_ko)

summ = pd.DataFrame(data = [allpeaks, genotype], index = ['freq', 'genotype']).T

plt.figure()
plt.title('Peak Frequency in Wake')
sns.set_style('white')
palette = ['royalblue', 'indianred']
ax = sns.violinplot( x = summ['genotype'], y=summ['freq'].astype(float) , data = summ, dodge=False,
                    palette = palette,cut = 2,
                    scale="width", inner=None)
ax.tick_params(bottom=True, left=True)
xlim = ax.get_xlim()
ylim = ax.get_ylim()
for violin in ax.collections:
    x0, y0, width, height = violin.get_paths()[0].get_extents().bounds
    violin.set_clip_path(plt.Rectangle((x0, y0), width / 2, height, transform=ax.transData))
sns.boxplot(x = summ['genotype'], y=summ['freq'].astype(float) , data = summ, saturation=1, showfliers=False,
            width=0.3, boxprops={'zorder': 3, 'facecolor': 'none'}, ax=ax)
old_len_collections = len(ax.collections)
sns.stripplot(x = summ['genotype'], y = summ['freq'].astype(float), data = summ, color = 'k', dodge=False, ax=ax)
# sns.swarmplot(x = durdf['genotype'], y = durdf['dur'].astype(float), data = durdf, color = 'k', dodge=False, ax=ax)
for dots in ax.collections[old_len_collections:]:
    dots.set_offsets(dots.get_offsets())
ax.set_xlim(xlim)
ax.set_ylim(ylim)
plt.ylabel('Frequency (Hz)')
ax.set_box_aspect(1)

wt = np.array(['WT' for x in range(len(peakfreq_rem_wt))])
ko = np.array(['KO' for x in range(len(peakfreq_rem_ko))])

genotype = np.hstack([wt, ko])

allpeaks = []
allpeaks.extend(peakfreq_rem_wt)
allpeaks.extend(peakfreq_rem_ko)

summ = pd.DataFrame(data = [allpeaks, genotype], index = ['freq', 'genotype']).T

plt.figure()
plt.title('Peak Frequency in REM')
sns.set_style('white')
palette = ['royalblue', 'indianred']
ax = sns.violinplot( x = summ['genotype'], y=summ['freq'].astype(float) , data = summ, dodge=False,
                    palette = palette,cut = 2,
                    scale="width", inner=None)
ax.tick_params(bottom=True, left=True)
xlim = ax.get_xlim()
ylim = ax.get_ylim()
for violin in ax.collections:
    x0, y0, width, height = violin.get_paths()[0].get_extents().bounds
    violin.set_clip_path(plt.Rectangle((x0, y0), width / 2, height, transform=ax.transData))
sns.boxplot(x = summ['genotype'], y=summ['freq'].astype(float) , data = summ, saturation=1, showfliers=False,
            width=0.3, boxprops={'zorder': 3, 'facecolor': 'none'}, ax=ax)
old_len_collections = len(ax.collections)
sns.stripplot(x = summ['genotype'], y = summ['freq'].astype(float), data = summ, color = 'k', dodge=False, ax=ax)
# sns.swarmplot(x = durdf['genotype'], y = durdf['dur'].astype(float), data = durdf, color = 'k', dodge=False, ax=ax)
for dots in ax.collections[old_len_collections:]:
    dots.set_offsets(dots.get_offsets())
ax.set_xlim(xlim)
ax.set_ylim(ylim)
plt.ylabel('Frequency (Hz)')
ax.set_box_aspect(1)


t_w, p_w = mannwhitneyu(peakfreq_wake_wt, peakfreq_wake_ko)
t_r, p_r = mannwhitneyu(peakfreq_rem_wt, peakfreq_rem_ko)


#%% 

#     fmin = 1
#     fmax = 9
#     nfreqs = 100
#     freqs = np.logspace(np.log10(fmin),np.log10(fmax),nfreqs)
    
#     filter_bank = nap.generate_morlet_filterbank(freqs, fs, gaussian_width=1.5, window_length=1.0)
    
#     fig, ax = plt.subplots(1, constrained_layout=True, figsize=(10, 7))
#     for f_i in range(filter_bank.shape[1]):
#         ax.plot(filter_bank[:, f_i].real() + f_i * 1.5)
#         ax.text(-6.8, 1.5 * f_i, f"{np.round(freqs[f_i], 2)}Hz", va="center", ha="left")

#     ax.set_yticks([])
#     ax.set_xlim(-5, 5)
#     ax.set_xlabel("Time (s)")


    
#     ncyc = 3 #5
#     si = 1/fs
           
    
    
#     nfreqs = len(freqs)
    
#     wavespec = pd.DataFrame(index = lfp.index.values, columns = freqs)
#     powerspec = pd.DataFrame(index = lfp.index.values, columns = freqs)
        
#     for f in range(len(freqs)):
#          wavelet = MorletWavelet(freqs[f],ncyc,si)
#          tmpspec = fftconvolve(lfp.values, wavelet, mode = 'same')
#          wavespec[freqs[f]] = tmpspec
#          temppower = abs(wavespec[freqs[f]]) #**2
#          powerspec[freqs[f]] =  temppower #(temppower.values/np.median(temppower.values))
         
# #%% 
    
#     plt.figure()
#     plt.suptitle('NREM')
#     plt.title('WT')
#     plt.specgram(lfp.restrict(moving_ep), Fs = fs, cmap = 'jet', vmin = 0, vmax = 50)
#     plt.colorbar()
   