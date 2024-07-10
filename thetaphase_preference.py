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
import warnings
import seaborn as sns
from scipy.signal import hilbert, fftconvolve
from pingouin import circ_r, circ_mean, circ_rayleigh
from scipy.stats import mannwhitneyu
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

def shuffleByCircularSpikes(spikes, ep):
    shuffled = {}
    for n in spikes.keys():
        
        for j in range(len(ep)):
            spk = spikes[n].restrict(ep[j])
            shift = np.random.uniform(0, (ep[j]['end'][0] - ep[j]['start'][0]))
            spk_shifted = (spk.index.values + shift) % (ep[j]['end'][0] - ep[j]['start'][0]) + ep[j]['start'][0]
            
            if  j == 0:
                shuffled[n] = spk_shifted
            else:
                shuffled[n] = np.append(shuffled[n], spk_shifted)
    
    shuffled = nap.TsGroup(shuffled)
    return shuffled
                
def circular_hist(ax, x, bins=16, density=True, offset=0, gaps=True):
    """
    Produce a circular histogram of angles on ax.

    Parameters
    ----------
    ax : matplotlib.axes._subplots.PolarAxesSubplot
        axis instance created with subplot_kw=dict(projection='polar').

    x : array
        Angles to plot, expected in units of radians.

    bins : int, optional
        Defines the number of equal-width bins in the range. The default is 16.

    density : bool, optional
        If True plot frequency proportional to area. If False plot frequency
        proportional to radius. The default is True.

    offset : float, optional
        Sets the offset for the location of the 0 direction in units of
        radians. The default is 0.

    gaps : bool, optional
        Whether to allow gaps between bins. When gaps = False the bins are
        forced to partition the entire [-pi, pi] range. The default is True.

    Returns
    -------
    n : array or list of arrays
        The number of values in each bin.

    bins : array
        The edges of the bins.

    patches : `.BarContainer` or list of a single `.Polygon`
        Container of individual artists used to create the histogram
        or list of such containers if there are multiple input datasets.
    """
    # Wrap angles to [-pi, pi)
    x = (x+np.pi) % (2*np.pi) - np.pi

    # Force bins to partition entire circle
    if not gaps:
        bins = np.linspace(-np.pi, np.pi, num=bins+1)

    # Bin data and record counts
    n, bins = np.histogram(x, bins=bins)

    # Compute width of each bin
    widths = np.diff(bins)

    # By default plot frequency proportional to area
    if density:
        # Area to assign each bin
        area = n / x.size
        # Calculate corresponding bin radius
        radius = (area/np.pi) ** .5
    # Otherwise plot frequency proportional to radius
    else:
        radius = n

    # Plot data on ax
    patches = ax.bar(bins[:-1], radius, zorder=1, align='edge', width=widths,
                     edgecolor='C0', fill=False, linewidth=1)

    # Set the direction of the zero angle
    ax.set_theta_offset(offset)

    # Remove ylabels for area plots (they are mostly obstructive)
    if density:
        ax.set_yticks([])

    return n, bins, patches
 
    
#%% 

warnings.filterwarnings("ignore")

data_directory = '/media/dhruv/Expansion/Processed'
# data_directory = '/media/adrien/Expansion/Processed'

# data_directory = '/media/adrien/Expansion/Processed'
datasets = np.genfromtxt(os.path.join(data_directory,'dataset_DM.list'), delimiter = '\n', dtype = str, comments = '#')
# datasets = np.genfromtxt(os.path.join(data_directory,'dataset_test.list'), delimiter = '\n', dtype = str, comments = '#')
ripplechannels = np.genfromtxt(os.path.join(data_directory,'ripplechannel.list'), delimiter = '\n', dtype = str, comments = '#')
# ripplechannels = np.genfromtxt(os.path.join(data_directory,'ripplechannel_test.list'), delimiter = '\n', dtype = str, comments = '#')

fs = 1250

darr_wt_wake = np.zeros((len(datasets),40,100))
darr_wt_rem = np.zeros((len(datasets),40,100))

darr_ko_wake = np.zeros((len(datasets),40,100))
darr_ko_rem = np.zeros((len(datasets),40,100))

mrl_pyr_wt = []
mrl_pyr_ko = []

mrl_pv_wt = []
mrl_pv_ko = []

means_pyr_wt = []
means_pyr_ko = []

means_pv_wt = []
means_pv_ko = []

p_pyr_wt = []
p_pyr_ko = []

p_pv_wt = []
p_pv_ko = []

tokeep_pyr_wt = []
tokeep_pv_wt = []

tokeep_pyr_ko = []
tokeep_pv_ko = []

fracsig_pyr_wt = []
fracsig_pv_wt = []

fracsig_pyr_ko = []
fracsig_pv_ko = []


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
    
    # with open(os.path.join(path, 'riptsd.pickle'), 'rb') as pickle_file:
    #     rip_tsd = pickle.load(pickle_file)
        
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
    speed = pd.Series(index = tmp.index.values[0:-1]+ speedbinsize/2, data = distance/speedbinsize) # in cm/s
    speed2 = speed.rolling(window = 25, win_type='gaussian', center=True, min_periods=1).mean(std=10) #Smooth over 200ms 
    speed2 = nap.Tsd(speed2)
    moving_ep = nap.IntervalSet(speed2.threshold(2).time_support) #Epochs in which speed is > 2 cm/s
    
    # print(len(moving_ep))
    
    # sys.exit()
        
#%% 

    ep = rem_ep
    downsample = 2
    
    lfpsig = lfp.restrict(ep)   

    # lfp_filt_theta_wake = pyna.eeg_processing.bandpass_filter(lfp_wake, 30, 150, 1250)
       
    lfp_filt_theta = pyna.eeg_processing.bandpass_filter(lfpsig, 6, 9, 1250)
           
    h_power = nap.Tsd(t = lfp_filt_theta.index.values, d = hilbert(lfp_filt_theta))
     
    phase = nap.Tsd(t = lfp_filt_theta.index.values, d = (np.angle(h_power.values) + 2 * np.pi) % (2 * np.pi))
    phase = phase[::downsample]
      
#%% Compute phase preference
    
    if len(pv) > 0 and len(pyr) > 0:      
        phasepref_pyr = nap.compute_1d_tuning_curves(pyr, phase, 40, ep) 
        phasepref_pyr = smoothAngularTuningCurves(phasepref_pyr, sigma=3)
        
        phasepref_pv = nap.compute_1d_tuning_curves(pv, phase, 40, ep) 
        phasepref_pv = smoothAngularTuningCurves(phasepref_pv, sigma=3)
        
        sess_mrl_pyr = []        
        sess_mean_pyr = []
        
        sess_mrl_pv = []  
        sess_mean_pv = []
        
        sess_tokeep_pyr = []
        sess_tokeep_pv = []
             
        
        shu_mrl_pyr  = {}
        shu_mrl_pv = {}
        
        sess_tokeep_pyr = []
        shu_threshold_pyr = {}
        
        sess_tokeep_pv = []
        shu_threshold_pv = {}

#%% Computing shuffles PYR 
    
        for k in range(100):
            shu_pyr = shuffleByCircularSpikes(pyr, ep)    
            phasepref_shu_pyr = nap.compute_1d_tuning_curves(shu_pyr, phase, 40, ep)  
            phasepref_shu_pyr = smoothAngularTuningCurves(phasepref_shu_pyr, sigma=3)
        
            for ii in phasepref_shu_pyr.columns:
                MRL = circ_r(phasepref_shu_pyr.index.values, w = phasepref_shu_pyr[ii])
                
                if k == 0:
                    shu_mrl_pyr[ii] = MRL
                else: 
                    shu_mrl_pyr[ii] = np.append(shu_mrl_pyr[ii], MRL)
               
        
        for ii in phasepref_pyr.columns:
            shu_threshold_pyr[ii] = np.percentile(shu_mrl_pyr[ii], 95)
            MRL = circ_r(phasepref_pyr.index.values, w = phasepref_pyr[ii])
            meanbin = circ_mean(phasepref_pyr.index.values, w = phasepref_pyr[ii])
            
            # plt.figure()
            # plt.title(MRL > shu_threshold_pyr[ii])
            # plt.hist(shu_mrl_pyr[ii])
            # plt.axvline(MRL)
            
            sess_mrl_pyr.append(MRL)
            sess_mean_pyr.append(meanbin)
            
                            
            if isWT == 1:
                mrl_pyr_wt.append(MRL)    
                means_pyr_wt.append(meanbin)
                
                if MRL > shu_threshold_pyr[ii]:
                    tokeep_pyr_wt.append(True)
                    sess_tokeep_pyr.append(True)
                else:
                    tokeep_pyr_wt.append(False)
                    sess_tokeep_pyr.append(False)
             
                                                   
            else:
                mrl_pyr_ko.append(MRL)    
                means_pyr_ko.append(meanbin)
           
                if MRL > shu_threshold_pyr[ii]:
                    tokeep_pyr_ko.append(True)
                    sess_tokeep_pyr.append(True)
                else:
                    tokeep_pyr_ko.append(False)
                    sess_tokeep_pyr.append(False)
     
#%% Computing shuffles FS
    
        for k in range(100):
            shu_pv = shuffleByCircularSpikes(pv, ep)    
            phasepref_shu_pv = nap.compute_1d_tuning_curves(shu_pv, phase, 40, ep)  
            phasepref_shu_pv = smoothAngularTuningCurves(phasepref_shu_pv, sigma=3)
        
            for ii in phasepref_shu_pv.columns:
                MRL = circ_r(phasepref_shu_pv.index.values, w = phasepref_shu_pv[ii])
                
                if k == 0:
                    shu_mrl_pv[ii] = MRL
                else: 
                    shu_mrl_pv[ii] = np.append(shu_mrl_pv[ii], MRL)
               
        
        for ii in phasepref_pv.columns:
            shu_threshold_pv[ii] = np.percentile(shu_mrl_pv[ii], 95)
            MRL = circ_r(phasepref_pv.index.values, w = phasepref_pv[ii])
            meanbin = circ_mean(phasepref_pv.index.values, w = phasepref_pv[ii])
            
            # plt.figure()
            # plt.title(MRL > shu_threshold_pv[ii])
            # plt.hist(shu_mrl_pv[ii])
            # plt.axvline(MRL)
            
            sess_mrl_pv.append(MRL)
            sess_mean_pv.append(meanbin)
            
                            
            if isWT == 1:
                mrl_pv_wt.append(MRL)    
                means_pv_wt.append(meanbin)
                
                if MRL > shu_threshold_pv[ii]:
                    tokeep_pv_wt.append(True)
                    sess_tokeep_pv.append(True)
                else:
                    tokeep_pv_wt.append(False)
                    sess_tokeep_pv.append(False)
     
                
            else:
                mrl_pv_ko.append(MRL)    
                means_pv_ko.append(meanbin)
           
                if MRL > shu_threshold_pv[ii]:
                    tokeep_pv_ko.append(True)
                    sess_tokeep_pv.append(True)
                else:
                    tokeep_pv_ko.append(False)
                    sess_tokeep_pv.append(False)
       

#%%                 
                
        # fig, ax = plt.subplots(1,2, subplot_kw=dict(projection = 'polar'))
        # fig.suptitle(s)
        # ax[0].set_title('PYR')
        # ax[0].plot(sess_mean_pyr, sess_mrl_pyr, 'o', color = 'b')
        # ax[1].set_title('FS')
        # ax[1].plot(sess_mean_pv, sess_mrl_pv, 'o', color = 'r')
  
        
#%% Determine fraction of significantly coupled cells per session

    if isWT == 1:
        fracsig_pyr_wt.append(len(np.array(sess_mrl_pyr)[sess_tokeep_pyr])/len(sess_mrl_pyr))
        fracsig_pv_wt.append(len(np.array(sess_mrl_pv)[sess_tokeep_pv])/len(sess_mrl_pv))
        
    else:
        fracsig_pyr_ko.append(len(np.array(sess_mrl_pyr)[sess_tokeep_pyr])/len(sess_mrl_pyr))
        fracsig_pv_ko.append(len(np.array(sess_mrl_pv)[sess_tokeep_pv])/len(sess_mrl_pv))
        
#%% Delete variables before next iteration     
    
    del pyr, pv
        
#%% Out of loop plotting 
            
# fig, ax = plt.subplots(1,2, subplot_kw=dict(projection = 'polar'))
# fig.suptitle('WT')
# ax[0].set_title('PYR')
# ax[0].plot(means_pyr_wt, mrl_pyr_wt, 'o', color = 'b')
# ax[1].set_title('FS')
# ax[1].plot(means_pv_wt, mrl_pv_wt, 'o', color = 'r')
    
# fig, ax = plt.subplots(1,2, subplot_kw=dict(projection = 'polar'))
# fig.suptitle('KO')
# ax[0].set_title('PYR')
# ax[0].plot(means_pyr_ko, mrl_pyr_ko, 'o', color = 'b')
# ax[1].set_title('FS')
# ax[1].plot(means_pv_ko, mrl_pv_ko, 'o', color = 'r')

fig, ax = plt.subplots(1,2, subplot_kw=dict(projection = 'polar'))
plt.suptitle('WT')
ax[0].set_title('PYR')
ax[0].plot(np.array(means_pyr_wt)[np.array(tokeep_pyr_wt)], np.array(mrl_pyr_wt)[np.array(tokeep_pyr_wt)], 'o', color = 'lightsteelblue')
ax[1].set_title('FS')
ax[1].plot(np.array(means_pv_wt)[np.array(tokeep_pv_wt)], np.array(mrl_pv_wt)[np.array(tokeep_pv_wt)], 'o', color = 'lightcoral')


fig, ax = plt.subplots(1,2, subplot_kw=dict(projection = 'polar'))
plt.suptitle('KO')
ax[0].set_title('PYR')
ax[0].plot(np.array(means_pyr_ko)[np.array(tokeep_pyr_ko)], np.array(mrl_pyr_ko)[np.array(tokeep_pyr_ko)], 'o', color = 'royalblue')
ax[1].set_title('FS')
ax[1].plot(np.array(means_pv_ko)[np.array(tokeep_pv_ko)], np.array(mrl_pv_ko)[np.array(tokeep_pv_ko)], 'o', color = 'indianred')


# fig, ax = plt.subplots(1,2, subplot_kw=dict(projection = 'polar'))
# ax[0].set_title('PYR')
# ax[0].plot(np.array(means_pyr_wt)[np.array(tokeep_pyr_wt)], np.array(mrl_pyr_wt)[np.array(tokeep_pyr_wt)], 'o', color = 'royalblue', label = 'WT')
# ax[0].plot(np.array(means_pyr_ko)[np.array(tokeep_pyr_ko)], np.array(mrl_pyr_ko)[np.array(tokeep_pyr_ko)], 'o', color = 'k', label = 'KO')
# ax[0].legend(loc = 'upper right')
# ax[1].set_title('FS')
# ax[1].plot(np.array(means_pv_wt)[np.array(tokeep_pv_wt)], np.array(mrl_pv_wt)[np.array(tokeep_pv_wt)], 'o', color = 'indianred', label = 'WT')
# ax[1].plot(np.array(means_pv_ko)[np.array(tokeep_pv_ko)], np.array(mrl_pv_ko)[np.array(tokeep_pv_ko)], 'o', color = 'k', label = 'KO')
# ax[1].legend(loc = 'upper right')


#%% Organize fraction of significantly coupled cells

wt1 = np.array(['WT' for x in range(len(fracsig_pyr_wt))])
wt2 = np.array(['WT' for x in range(len(fracsig_pv_wt))])

ko1 = np.array(['KO' for x in range(len(fracsig_pyr_ko))])
ko2 = np.array(['KO' for x in range(len(fracsig_pv_ko))])

genotype = np.hstack([wt1, ko1, wt2, ko2])

ex = np.array(['PYR' for x in range(len(fracsig_pyr_wt))])
inh = np.array(['FS' for x in range(len(fracsig_pv_wt))])

ex2 = np.array(['PYR' for x in range(len(fracsig_pyr_ko))])
inh2 = np.array(['FS' for x in range(len(fracsig_pv_ko))])

ctype = np.hstack([ex, ex2, inh, inh2])

sigfracs = []
sigfracs.extend(fracsig_pyr_wt)
sigfracs.extend(fracsig_pyr_ko)
sigfracs.extend(fracsig_pv_wt)
sigfracs.extend(fracsig_pv_ko)

infos = pd.DataFrame(data = [sigfracs, ctype, genotype], index = ['frac', 'celltype', 'genotype']).T

#%% Plot fraction of significantly coupled cells

plt.figure()
plt.subplot(121)
plt.title('PYR')
sns.set_style('white')
palette = ['lightsteelblue', 'lightcoral']
ax = sns.violinplot( x = infos[infos['celltype'] == 'PYR']['genotype'], y=infos[infos['celltype'] == 'PYR']['frac'].astype(float) , data = infos, dodge=False,
                    palette = palette,cut = 2,
                    scale="width", inner=None)
ax.tick_params(bottom=True, left=True)
xlim = ax.get_xlim()
ylim = ax.get_ylim()
for violin in ax.collections:
    x0, y0, width, height = violin.get_paths()[0].get_extents().bounds
    violin.set_clip_path(plt.Rectangle((x0, y0), width / 2, height, transform=ax.transData))
sns.boxplot(x = infos[infos['celltype'] == 'PYR']['genotype'], y=infos[infos['celltype'] == 'PYR']['frac'] , data = infos, saturation=1, showfliers=False,
            width=0.3, boxprops={'zorder': 3, 'facecolor': 'none'}, ax=ax)
old_len_collections = len(ax.collections)
sns.stripplot(x = infos[infos['celltype'] == 'PYR']['genotype'], y=infos[infos['celltype'] == 'PYR']['frac'], data = infos, color = 'k', dodge=False, ax=ax)

for dots in ax.collections[old_len_collections:]:
    dots.set_offsets(dots.get_offsets())
ax.set_xlim(xlim)
ax.set_ylim(ylim)
plt.ylabel('Fraction of significantly coupled cells')
ax.set_box_aspect(1)

plt.subplot(122)
plt.title('FS')
sns.set_style('white')
palette = ['royalblue', 'indianred']
ax = sns.violinplot( x = infos[infos['celltype'] == 'FS']['genotype'], y=infos[infos['celltype'] == 'FS']['frac'].astype(float) , data = infos, dodge=False,
                    palette = palette,cut = 2,
                    scale="width", inner=None)
ax.tick_params(bottom=True, left=True)
xlim = ax.get_xlim()
ylim = ax.get_ylim()
for violin in ax.collections:
    x0, y0, width, height = violin.get_paths()[0].get_extents().bounds
    violin.set_clip_path(plt.Rectangle((x0, y0), width / 2, height, transform=ax.transData))
sns.boxplot(x = infos[infos['celltype'] == 'FS']['genotype'], y=infos[infos['celltype'] == 'FS']['frac'] , data = infos, saturation=1, showfliers=False,
            width=0.3, boxprops={'zorder': 3, 'facecolor': 'none'}, ax=ax)
old_len_collections = len(ax.collections)
sns.stripplot(x = infos[infos['celltype'] == 'FS']['genotype'], y=infos[infos['celltype'] == 'FS']['frac'], data = infos, color = 'k', dodge=False, ax=ax)

for dots in ax.collections[old_len_collections:]:
    dots.set_offsets(dots.get_offsets())
ax.set_xlim(xlim)
ax.set_ylim(ylim)
plt.ylabel('Fraction of significantly coupled cells')
ax.set_box_aspect(1)

#%% Stats 

t_pyr, p_pyr = mannwhitneyu(fracsig_pyr_wt, fracsig_pyr_ko)
t_pv, p_pv = mannwhitneyu(fracsig_pv_wt, fracsig_pv_ko)


#%%             
            # fig, ax = plt.subplots(1, 2, subplot_kw=dict(projection='polar'))
            # circular_hist(ax[0], np.array(sess_mean_pv))
            # # Visualise by radius of bins
            # circular_hist(ax[1], np.array(sess_mean_pv), offset=np.pi/2, density=False)
            
            
fig, ax = plt.subplots(1,2, subplot_kw=dict(projection = 'polar'))
fig.suptitle('WT')
ax[0].set_title('PYR')
circular_hist(ax[0], np.array(means_pyr_wt)[np.array(tokeep_pyr_wt)], bins = 16)
ax[1].set_title('FS')
circular_hist(ax[1], np.array(means_pv_wt)[np.array(tokeep_pv_wt)], bins = 16)
    
fig, ax = plt.subplots(1,2, subplot_kw=dict(projection = 'polar'))
fig.suptitle('KO')
ax[0].set_title('PYR')
circular_hist(ax[0], np.array(means_pyr_ko)[np.array(tokeep_pyr_ko)], bins = 16)
ax[1].set_title('FS')
circular_hist(ax[1], np.array(means_pv_ko)[np.array(tokeep_pv_ko)], bins = 16)

    
#%% Session spectra and tuning curves

        # plt.figure()
        # plt.title(s + '_wake_pyr')
        # for i,n in enumerate(pyr):
        #     plt.subplot(9,8,n+1, projection='polar')
        #     plt.plot(phasepref_wake_pyr[n], color = 'b')        
            
        # plt.figure()
        # plt.title(s + '_wake_pv')
        # for i,n in enumerate(pv):
        #     plt.subplot(9,8,n+1, projection='polar')
        #     plt.plot(phasepref_wake_pv[n], color = 'r')   
            
    

# multipage(data_directory + '/' + 'GammaPhaseplots_wake.pdf', dpi=250)

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
        
        
