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
from scipy.signal import hilbert
from scipy.stats import circmean
from pingouin import circ_r
from matplotlib.backends.backend_pdf import PdfPages    

#%% 

def multipage(filename, figs=None, dpi=200):
    pp = PdfPages(filename)
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()
    
#%% 

data_directory = '/media/dhruv/Expansion/Processed'
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
        
#%% Load spikes 

    sp2 = np.load(os.path.join(path, 'spikedata.npz'), allow_pickle = True)
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
        
#%% 
    
    lfp_wake = lfp.restrict(nap.IntervalSet(epochs['wake']))
    lfp_rem = lfp.restrict(nap.IntervalSet(rem_ep))    

    lfp_filt_theta_wake = pyna.eeg_processing.bandpass_filter(lfp_wake, 5, 12, 1250)
    lfp_filt_theta_rem = pyna.eeg_processing.bandpass_filter(lfp_rem, 5, 12, 1250)
        
    h_wake = nap.Tsd(t = lfp_filt_theta_wake.index.values, d = hilbert(lfp_filt_theta_wake))
    h_rem = nap.Tsd(t = lfp_filt_theta_rem.index.values, d = hilbert(lfp_filt_theta_rem))
    
    phase_wake = nap.Tsd(t = lfp_filt_theta_wake.index.values, d = (np.angle(h_wake.values) + 2 * np.pi) % (2 * np.pi))
          
    bins = np.linspace(0, 2*np.pi, 30)    
    widths = np.diff(bins)
       
    if len(pv) > 0 and len(pyr) > 0:        
        
        spikephase_wake_ex = {}
        plt.figure() 
        plt.suptitle(s + ' EX cells')
        
        cmeans_pyr = []
        vlens_pyr = []
    
        for i, j in enumerate(pyr.index):
            spikephase_wake_ex[j] = pyr[j].value_from(phase_wake) 
            c = circmean(spikephase_wake_ex[j])
            cmeans_pyr.append(c)
            veclength = circ_r(spikephase_wake_ex[j])
            vlens_pyr.append(veclength)
            n, bins = np.histogram(spikephase_wake_ex[j], bins)
            area = n / spikephase_wake_ex[j].size
            radius = (area/np.pi) ** .5
            
            # plt.subplot(10, 6, i+1)
            # plt.hist(spikephase_wake[j])
            
            ax = plt.subplot(10,6,i+1, projection='polar')
            ax.bar(bins[:-1], radius, width=widths, color = 'lightsteelblue')
            ax.set_yticks([])
       
    
            
        spikephase_wake_fs = {}
        plt.figure() 
        plt.suptitle(s + ' FS cells')
        
        cmeans_fs = []
        vlens_fs = []
    
        for i, j in enumerate(pv.index):
            spikephase_wake_fs[j] = pv[j].value_from(phase_wake) 
            c = circmean(spikephase_wake_fs[j])
            cmeans_fs.append(c)
            veclength = circ_r(spikephase_wake_fs[j])
            vlens_fs.append(veclength)
            n, bins = np.histogram(spikephase_wake_fs[j], bins)
            area = n / spikephase_wake_fs[j].size
            radius = (area/np.pi) ** .5
            
            # plt.subplot(10, 6, i+1)
            # plt.hist(spikephase_wake[j])
            
            ax = plt.subplot(10,6,i+1, projection='polar')
            ax.bar(bins[:-1], radius, width=widths, color = 'lightcoral')
            ax.set_yticks([])
               
#%% 
    
    # multipage(data_directory + '/' + 'Phaseplots.pdf', dpi=250)
        
        
