#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 16:57:45 2024

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
datasets = np.genfromtxt(os.path.join(data_directory,'dataset_DM_rippletravel.list'), delimiter = '\n', dtype = str, comments = '#')

ripplechannels = np.genfromtxt(os.path.join(data_directory,'ripplechannel.list'), delimiter = '\n', dtype = str, comments = '#')


fs = 1250

darr_wt_wake = np.zeros((len(datasets),40,100))
darr_wt_rem = np.zeros((len(datasets),40,100))

darr_ko_wake = np.zeros((len(datasets),40,100))
darr_ko_rem = np.zeros((len(datasets),40,100))

mrl_sup_wt = []
mrl_sup_ko = []

mrl_deep_wt = []
mrl_deep_ko = []

means_sup_wt = []
means_sup_ko = []

means_deep_wt = []
means_deep_ko = []

p_sup_wt = []
p_sup_ko = []

p_deep_wt = []
p_deep_ko = []

tokeep_sup_wt = []
tokeep_deep_wt = []

tokeep_sup_ko = []
tokeep_deep_ko = []

fracsig_sup_wt = []
fracsig_deep_wt = []

fracsig_sup_ko = []
fracsig_deep_ko = []


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
    
            
#%% Load spikes 

    sp2 = np.load(os.path.join(path, 'pyr_layers.npz'), allow_pickle = True)
    time_support = nap.IntervalSet(sp2['start'], sp2['end'])
    tsd = nap.Tsd(t=sp2['t'], d=sp2['index'], time_support = time_support)
    spikes = tsd.to_tsgroup()
    spikes.set_info(group = sp2['group'], location = sp2['location'], celltype = sp2['celltype'], layer = sp2['layer'])
            
#%% 
    
    spikes_by_layer = spikes.getby_category('layer')
    
    if 'sup' in spikes._metadata['layer'].values:
        sup = spikes_by_layer['sup']
    else: sup = []
        
    if 'deep' in spikes._metadata['layer'].values:
        deep = spikes_by_layer['deep']
    else: deep = []
    
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

    # ep = rem_ep
    ep = moving_ep
    
    downsample = 2
    
    lfpsig = lfp.restrict(ep)   

    # lfp_filt_theta_wake = pyna.eeg_processing.bandpass_filter(lfp_wake, 30, 150, 1250)
       
    lfp_filt_theta = pyna.eeg_processing.bandpass_filter(lfpsig, 6, 9, 1250)
           
    h_power = nap.Tsd(t = lfp_filt_theta.index.values, d = hilbert(lfp_filt_theta))
     
    phase = nap.Tsd(t = lfp_filt_theta.index.values, d = (np.angle(h_power.values) + 2 * np.pi) % (2 * np.pi))
    phase = phase[::downsample]
      
#%% Compute phase preference
    
    if len(sup) > 0 and len(deep) > 0:      
        phasepref_sup = nap.compute_1d_tuning_curves(sup, phase, 40, ep) 
        phasepref_sup = smoothAngularTuningCurves(phasepref_sup, sigma=3)
        
        phasepref_deep = nap.compute_1d_tuning_curves(deep, phase, 40, ep) 
        phasepref_deep = smoothAngularTuningCurves(phasepref_deep, sigma=3)
        
        sess_mrl_sup = []        
        sess_mean_sup = []
        
        sess_mrl_deep = []  
        sess_mean_deep = []
        
        sess_tokeep_sup = []
        sess_tokeep_deep = []
             
        
        shu_mrl_sup  = {}
        shu_mrl_deep = {}
        
        sess_tokeep_sup = []
        shu_threshold_sup = {}
        
        sess_tokeep_deep = []
        shu_threshold_deep = {}

#%% Computing shuffles Sup 
    
        for k in range(100):
            shu_sup = shuffleByCircularSpikes(sup, ep)    
            phasepref_shu_sup = nap.compute_1d_tuning_curves(shu_sup, phase, 40, ep)  
            phasepref_shu_sup = smoothAngularTuningCurves(phasepref_shu_sup, sigma=3)
        
            for ii in phasepref_shu_sup.columns:
                MRL = circ_r(phasepref_shu_sup.index.values, w = phasepref_shu_sup[ii])
                
                if k == 0:
                    shu_mrl_sup[ii] = MRL
                else: 
                    shu_mrl_sup[ii] = np.append(shu_mrl_sup[ii], MRL)
               
        
        for ii in phasepref_sup.columns:
            shu_threshold_sup[ii] = np.percentile(shu_mrl_sup[ii], 95)
            MRL = circ_r(phasepref_sup.index.values, w = phasepref_sup[ii])
            meanbin = circ_mean(phasepref_sup.index.values, w = phasepref_sup[ii])
            
            # plt.figure()
            # plt.title(MRL > shu_threshold_pyr[ii])
            # plt.hist(shu_mrl_pyr[ii])
            # plt.axvline(MRL)
            
            sess_mrl_sup.append(MRL)
            sess_mean_sup.append(meanbin)
            
                            
            if isWT == 1:
                mrl_sup_wt.append(MRL)    
                means_sup_wt.append(meanbin)
                
                if MRL > shu_threshold_sup[ii]:
                    tokeep_sup_wt.append(True)
                    sess_tokeep_sup.append(True)
                else:
                    tokeep_sup_wt.append(False)
                    sess_tokeep_sup.append(False)
             
                                                   
            else:
                mrl_sup_ko.append(MRL)    
                means_sup_ko.append(meanbin)
           
                if MRL > shu_threshold_sup[ii]:
                    tokeep_sup_ko.append(True)
                    sess_tokeep_sup.append(True)
                else:
                    tokeep_sup_ko.append(False)
                    sess_tokeep_sup.append(False)
     
#%% Computing shuffles deep
    
        for k in range(100):
            shu_deep = shuffleByCircularSpikes(deep, ep)    
            phasepref_shu_deep = nap.compute_1d_tuning_curves(shu_deep, phase, 40, ep)  
            phasepref_shu_deep = smoothAngularTuningCurves(phasepref_shu_deep, sigma=3)
        
            for ii in phasepref_shu_deep.columns:
                MRL = circ_r(phasepref_shu_deep.index.values, w = phasepref_shu_deep[ii])
                
                if k == 0:
                    shu_mrl_deep[ii] = MRL
                else: 
                    shu_mrl_deep[ii] = np.append(shu_mrl_deep[ii], MRL)
               
        
        for ii in phasepref_deep.columns:
            shu_threshold_deep[ii] = np.percentile(shu_mrl_deep[ii], 95)
            MRL = circ_r(phasepref_deep.index.values, w = phasepref_deep[ii])
            meanbin = circ_mean(phasepref_deep.index.values, w = phasepref_deep[ii])
            
            # plt.figure()
            # plt.title(MRL > shu_threshold_pv[ii])
            # plt.hist(shu_mrl_pv[ii])
            # plt.axvline(MRL)
            
            sess_mrl_deep.append(MRL)
            sess_mean_deep.append(meanbin)
            
                            
            if isWT == 1:
                mrl_deep_wt.append(MRL)    
                means_deep_wt.append(meanbin)
                
                if MRL > shu_threshold_deep[ii]:
                    tokeep_deep_wt.append(True)
                    sess_tokeep_deep.append(True)
                else:
                    tokeep_deep_wt.append(False)
                    sess_tokeep_deep.append(False)
     
                
            else:
                mrl_deep_ko.append(MRL)    
                means_deep_ko.append(meanbin)
           
                if MRL > shu_threshold_deep[ii]:
                    tokeep_deep_ko.append(True)
                    sess_tokeep_deep.append(True)
                else:
                    tokeep_deep_ko.append(False)
                    sess_tokeep_deep.append(False)
       

#%%                 
                
        # fig, ax = plt.subplots(1,2, subplot_kw=dict(projection = 'polar'))
        # fig.suptitle(s)
        # ax[0].set_title('PYR')
        # ax[0].plot(sess_mean_pyr, sess_mrl_pyr, 'o', color = 'b')
        # ax[1].set_title('FS')
        # ax[1].plot(sess_mean_pv, sess_mrl_pv, 'o', color = 'r')
  
        
#%% Determine fraction of significantly coupled cells per session

    if isWT == 1:
        fracsig_sup_wt.append(len(np.array(sess_mrl_sup)[sess_tokeep_sup])/len(sess_mrl_sup))
        fracsig_deep_wt.append(len(np.array(sess_mrl_deep)[sess_tokeep_deep])/len(sess_mrl_deep))
        
    else:
        fracsig_sup_ko.append(len(np.array(sess_mrl_sup)[sess_tokeep_sup])/len(sess_mrl_sup))
        fracsig_deep_ko.append(len(np.array(sess_mrl_deep)[sess_tokeep_deep])/len(sess_mrl_deep))
        
#%% Delete variables before next iteration     
    
    del sup, deep
        
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
ax[0].set_title('Sup cells')
ax[0].plot(np.array(means_sup_wt)[np.array(tokeep_sup_wt)], np.array(mrl_sup_wt)[np.array(tokeep_sup_wt)], 'o', color = 'lightsteelblue')
ax[1].set_title('Deep cells')
ax[1].plot(np.array(means_deep_wt)[np.array(tokeep_deep_wt)], np.array(mrl_deep_wt)[np.array(tokeep_deep_wt)], 'o', color = 'lightcoral')


fig, ax = plt.subplots(1,2, subplot_kw=dict(projection = 'polar'))
plt.suptitle('KO')
ax[0].set_title('Sup cells')
ax[0].plot(np.array(means_sup_ko)[np.array(tokeep_sup_ko)], np.array(mrl_sup_ko)[np.array(tokeep_sup_ko)], 'o', color = 'royalblue')
ax[1].set_title('Deep cells')
ax[1].plot(np.array(means_deep_ko)[np.array(tokeep_deep_ko)], np.array(mrl_deep_ko)[np.array(tokeep_deep_ko)], 'o', color = 'indianred')


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

wt1 = np.array(['WT' for x in range(len(fracsig_sup_wt))])
wt2 = np.array(['WT' for x in range(len(fracsig_deep_wt))])

ko1 = np.array(['KO' for x in range(len(fracsig_sup_ko))])
ko2 = np.array(['KO' for x in range(len(fracsig_deep_ko))])

genotype = np.hstack([wt1, ko1, wt2, ko2])

ex = np.array(['Sup' for x in range(len(fracsig_sup_wt))])
inh = np.array(['Deep' for x in range(len(fracsig_deep_wt))])

ex2 = np.array(['Sup' for x in range(len(fracsig_sup_ko))])
inh2 = np.array(['Deep' for x in range(len(fracsig_deep_ko))])

ctype = np.hstack([ex, ex2, inh, inh2])

sigfracs = []
sigfracs.extend(fracsig_sup_wt)
sigfracs.extend(fracsig_sup_ko)
sigfracs.extend(fracsig_deep_wt)
sigfracs.extend(fracsig_deep_ko)

infos = pd.DataFrame(data = [sigfracs, ctype, genotype], index = ['frac', 'celltype', 'genotype']).T

#%% Plot fraction of significantly coupled cells

plt.figure()
plt.subplot(121)
plt.title('Sup')
sns.set_style('white')
palette = ['lightsteelblue', 'lightcoral']
ax = sns.violinplot( x = infos[infos['celltype'] == 'Sup']['genotype'], y=infos[infos['celltype'] == 'Sup']['frac'].astype(float) , data = infos, dodge=False,
                    palette = palette,cut = 2,
                    scale="width", inner=None)
ax.tick_params(bottom=True, left=True)
xlim = ax.get_xlim()
ylim = ax.get_ylim()
for violin in ax.collections:
    x0, y0, width, height = violin.get_paths()[0].get_extents().bounds
    violin.set_clip_path(plt.Rectangle((x0, y0), width / 2, height, transform=ax.transData))
sns.boxplot(x = infos[infos['celltype'] == 'Sup']['genotype'], y=infos[infos['celltype'] == 'Sup']['frac'] , data = infos, saturation=1, showfliers=False,
            width=0.3, boxprops={'zorder': 3, 'facecolor': 'none'}, ax=ax)
old_len_collections = len(ax.collections)
sns.stripplot(x = infos[infos['celltype'] == 'Sup']['genotype'], y=infos[infos['celltype'] == 'Sup']['frac'], data = infos, color = 'k', dodge=False, ax=ax)

for dots in ax.collections[old_len_collections:]:
    dots.set_offsets(dots.get_offsets())
ax.set_xlim(xlim)
ax.set_ylim(ylim)
plt.ylabel('Fraction of significantly coupled cells')
ax.set_box_aspect(1)

plt.subplot(122)
plt.title('Deep')
sns.set_style('white')
palette = ['royalblue', 'indianred']
ax = sns.violinplot( x = infos[infos['celltype'] == 'Deep']['genotype'], y=infos[infos['celltype'] == 'Deep']['frac'].astype(float) , data = infos, dodge=False,
                    palette = palette,cut = 2,
                    scale="width", inner=None)
ax.tick_params(bottom=True, left=True)
xlim = ax.get_xlim()
ylim = ax.get_ylim()
for violin in ax.collections:
    x0, y0, width, height = violin.get_paths()[0].get_extents().bounds
    violin.set_clip_path(plt.Rectangle((x0, y0), width / 2, height, transform=ax.transData))
sns.boxplot(x = infos[infos['celltype'] == 'Deep']['genotype'], y=infos[infos['celltype'] == 'Deep']['frac'] , data = infos, saturation=1, showfliers=False,
            width=0.3, boxprops={'zorder': 3, 'facecolor': 'none'}, ax=ax)
old_len_collections = len(ax.collections)
sns.stripplot(x = infos[infos['celltype'] == 'Deep']['genotype'], y=infos[infos['celltype'] == 'Deep']['frac'], data = infos, color = 'k', dodge=False, ax=ax)

for dots in ax.collections[old_len_collections:]:
    dots.set_offsets(dots.get_offsets())
ax.set_xlim(xlim)
ax.set_ylim(ylim)
plt.ylabel('Fraction of significantly coupled cells')
ax.set_box_aspect(1)

#%% Stats 

t_sup, p_sup = mannwhitneyu(fracsig_sup_wt, fracsig_sup_ko)
t_deep, p_deep = mannwhitneyu(fracsig_deep_wt, fracsig_deep_ko)


#%%             
            # fig, ax = plt.subplots(1, 2, subplot_kw=dict(projection='polar'))
            # circular_hist(ax[0], np.array(sess_mean_pv))
            # # Visualise by radius of bins
            # circular_hist(ax[1], np.array(sess_mean_pv), offset=np.pi/2, density=False)
            
            
fig, ax = plt.subplots(1,2, subplot_kw=dict(projection = 'polar'))
fig.suptitle('WT')
ax[0].set_title('Sup cells')
circular_hist(ax[0], np.array(means_sup_wt)[np.array(tokeep_sup_wt)], bins = 16)
ax[1].set_title('Deep cells')
circular_hist(ax[1], np.array(means_deep_wt)[np.array(tokeep_deep_wt)], bins = 16)
    
fig, ax = plt.subplots(1,2, subplot_kw=dict(projection = 'polar'))
fig.suptitle('KO')
ax[0].set_title('Sup cells')
circular_hist(ax[0], np.array(means_sup_ko)[np.array(tokeep_sup_ko)], bins = 16)
ax[1].set_title('Deep cells')
circular_hist(ax[1], np.array(means_deep_ko)[np.array(tokeep_deep_ko)], bins = 16)
   
