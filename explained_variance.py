#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 11:39:03 2024

@author: dhruv
"""

import numpy as np 
import pandas as pd
import nwbmatic as ntm
import pynapple as nap 
import seaborn as sns
import os, sys
import matplotlib.pyplot as plt 
from scipy.stats import pearsonr, wilcoxon
import time
import pickle
import warnings 

#%% 

warnings.filterwarnings("ignore")

data_directory = '/media/dhruv/Expansion/Processed'
datasets = np.genfromtxt(os.path.join(data_directory,'dataset_DM.list'), delimiter = '\n', dtype = str, comments = '#')
ripplechannels = np.genfromtxt(os.path.join(data_directory,'ripplechannel.list'), delimiter = '\n', dtype = str, comments = '#')

Fs = 1250 

ev_wt_10 = []
ev_ko_10 = []

rev_wt_10 = []
rev_ko_10 = [] 

ev_wt_20 = []
ev_ko_20 = []

rev_wt_20 = []
rev_ko_20 = [] 

ev_wt_rest = []
ev_ko_rest = []

rev_wt_rest = []
rev_ko_rest = [] 

ncells_wt = []
ncells_ko = []

evdiff_10_wt = []
evdiff_20_wt = []
evdiff_rest_wt = []

evdiff_10_ko = []
evdiff_20_ko = []
evdiff_rest_ko = []

pre_dur = []
post_dur = []


for s in datasets:
    print(s)
    
    # t = time.time()
    name = s.split('-')[0]
    path = os.path.join(data_directory, s)
    
    data = ntm.load_session(path, 'neurosuite')
    data.load_neurosuite_xml(path)
    epochs = data.epochs
    position = data.position
    
    if name == 'B2613' or name == 'B2618' or name == 'B2627' or name == 'B2628' or name == 'B3805' or name == 'B3813':
        isWT = 0
    else: isWT = 1 
    
    file = os.path.join(path, s +'.evt.py.rip')
    rip_ep = data.read_neuroscope_intervals(name = 'rip', path2file = file)
    
    file = os.path.join(path, s +'.sws.evt')
    sws_ep = nap.IntervalSet(data.read_neuroscope_intervals(name = 'SWS', path2file = file))
    
    pre_dur.append(sws_ep.intersect(epochs['sleep'][0]).tot_length())
    post_dur.append(sws_ep.intersect(epochs['sleep'][1]).tot_length())
    
    # file = os.path.join(path, s +'.rem.evt')
    # rem_ep = nap.IntervalSet(data.read_neuroscope_intervals(name = 'REM', path2file = file))
    
    # with open(os.path.join(path, 'riptsd_5sd.pickle'), 'rb') as pickle_file:
    #     rip_tsd = pickle.load(pickle_file)
    
    # print(time.time() - t)

#%% Load classified spikes 

    # t = time.time()
    sp2 = np.load(os.path.join(path, 'spikedata_0.55.npz'), allow_pickle = True)
    time_support = nap.IntervalSet(sp2['start'], sp2['end'])
    tsd = nap.Tsd(t=sp2['t'], d=sp2['index'], time_support = time_support)
    spikes = tsd.to_tsgroup()
    spikes.set_info(group = sp2['group'], location = sp2['location'], celltype = sp2['celltype'], tr2pk = sp2['tr2pk'])
    # print(time.time() - t)
    
#%% Get only high-firing PYR cells for EV
    
    spikes_by_celltype = spikes.getby_category('celltype')
    
    if 'pyr' in spikes._metadata['celltype'].values:
        pyr = spikes_by_celltype['pyr']
    
    keep = []
    
    for i in pyr.index:
        if pyr.restrict(nap.IntervalSet(epochs['wake'][0]))._metadata['rate'][i] > 0.5:
            keep.append(i)
              
            
    pyr2 = pyr[keep]
    
    # keep2 = []
    # for i in pyr2.index:
    #     rw = pyr2.restrict(epochs['wake'].loc[[0]])._metadata['rate'][i]
    #     rn = pyr2.restrict(sws_ep)._metadata['rate'][i]
    #     rr = pyr2.restrict(rem_ep)._metadata['rate'][i]
        
    #     if (rn/rw >= 0.7) and (rr/rw >= 0.7):
    #         keep2.append(i)

    # pyr3 = pyr2[keep2]
        
#%% Compute speed during wake 

    # if len(pyr3) >= 10:
    
    if (len(pyr2)) >= 10:
        print('yes!')
        
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

#%% Bin all data 

        binsize = 0.05 #50ms 
        binned_spikes = pyr2.count(bin_size = binsize)
        
#%% Create sub-epochs 
    
    ###PRE
    
    ###First 10 min 
    
        # sub_nrem_ep = nap.IntervalSet(start = epochs['sleep'].loc[[0]].intersect(sws_ep).loc[[0]]['start'], end = epochs['sleep'].loc[[0]].intersect(sws_ep).loc[[0]]['end'])
        # tStart = epochs['sleep'].loc[[0]].intersect(sws_ep).loc[[0]]['start']
        # tEnd = epochs['sleep'].loc[[0]].intersect(sws_ep).loc[[0]]['end']
        
        # epcount = 0
        
        # sub_nrem_length = tEnd - tStart
        
        # while sub_nrem_length.values[0] < 600:
        #     sub_nrem_ep = sub_nrem_ep.union(nap.IntervalSet(start = tStart, end = tEnd))
            
        #     epcount += 1
                        
        #     tStart = epochs['sleep'].loc[[0]].intersect(sws_ep).loc[[epcount]]['start']
        #     tEnd = epochs['sleep'].loc[[0]].intersect(sws_ep).loc[[epcount]]['end']
        #     sub_nrem_length = sub_nrem_ep.tot_length() + (tEnd - tStart)
            
        # remainder_time = 600 - sub_nrem_ep.tot_length()
        # sub_nrem_ep = sub_nrem_ep.union(nap.IntervalSet(start = tStart, end = tStart + remainder_time))
        
        # ripp_pre = sub_nrem_ep.intersect(rip_ep)
    
    ### Last 20 min 
        sub_nrem_ep = nap.IntervalSet(start = epochs['sleep'][0].intersect(sws_ep)[-1]['start'], end = epochs['sleep'][0].intersect(sws_ep)[-1]['end'])
        tStart = epochs['sleep'][0].intersect(sws_ep)[-1]['start']
        tEnd = epochs['sleep'][0].intersect(sws_ep)[-1]['end']
        
        epcount = -1
        
        sub_nrem_length = tEnd - tStart
        
        while sub_nrem_length < 600:
            sub_nrem_ep = sub_nrem_ep.union(nap.IntervalSet(start = tStart, end = tEnd))
            
            epcount -= 1
                        
            tStart = epochs['sleep'][0].intersect(sws_ep)[epcount]['start']
            tEnd = epochs['sleep'][0].intersect(sws_ep)[epcount]['end']
            sub_nrem_length = sub_nrem_ep.tot_length() + (tEnd - tStart)
            
        remainder_time = 600 - sub_nrem_ep.tot_length()
        sub_nrem_ep = sub_nrem_ep.union(nap.IntervalSet(start = tEnd - remainder_time, end = tEnd))
        
        ripp_pre = sub_nrem_ep.intersect(rip_ep)
    
    ### All PRE 
        # ripp_pre = epochs['sleep'].loc[[0]].intersect(sws_ep).intersect(nap.IntervalSet(rip_ep))
        
   ###POST (first 20 min)
           
        sub_nrem_ep = nap.IntervalSet(start = epochs['sleep'][1].intersect(sws_ep)[0]['start'], end = epochs['sleep'][1].intersect(sws_ep)[0]['end'])
        tStart = epochs['sleep'][1].intersect(sws_ep)[0]['start']
        tEnd = epochs['sleep'][1].intersect(sws_ep)[0]['end']
        
        epcount = 0
        
        sub_nrem_length = tEnd - tStart
        
        while sub_nrem_length[0] < 600: #600:
            sub_nrem_ep = sub_nrem_ep.union(nap.IntervalSet(start = tStart, end = tEnd))
            
            epcount += 1
                        
            tStart = epochs['sleep'][1].intersect(sws_ep)[epcount]['start']
            tEnd = epochs['sleep'][1].intersect(sws_ep)[epcount]['end']
            sub_nrem_length = sub_nrem_ep.tot_length() + (tEnd - tStart)
            
        remainder_time = 600 - sub_nrem_ep.tot_length() ###CHECK 
        sub_nrem_ep = sub_nrem_ep.union(nap.IntervalSet(start = tStart, end = tStart + remainder_time))
        
        ripp_post_10 = sub_nrem_ep.intersect(nap.IntervalSet(rip_ep))
        
        # ripp_post_10 = epochs['sleep'].loc[[1]].intersect(sws_ep).intersect(nap.IntervalSet(rip_ep))
    
    ###POST (10-20 min)
                  
    
        # sub_nrem_ep2 = nap.IntervalSet(start = sub_nrem_ep.iloc[-1]['end'], end = tEnd)
        # tStart = sub_nrem_ep.iloc[-1]['end']
        # tEnd = tEnd
                       
        # sub_nrem_length = tEnd - tStart
        
        # while sub_nrem_length.values[0] < 600:
        #     sub_nrem_ep2 = sub_nrem_ep2.union(nap.IntervalSet(start = tStart, end = tEnd))
            
        #     epcount += 1
                        
        #     tStart = epochs['sleep'].loc[[1]].intersect(sws_ep).loc[[epcount]]['start'] 
        #     tEnd = epochs['sleep'].loc[[1]].intersect(sws_ep).loc[[epcount]]['end']
        #     sub_nrem_length = sub_nrem_ep.tot_length() - tStart + tEnd
            
        # remainder_time = 600 - sub_nrem_ep2.tot_length()
        # sub_nrem_ep2 = sub_nrem_ep2.union(nap.IntervalSet(start = tStart, end = tStart + remainder_time))
        
        # ripp_post_20 = sub_nrem_ep2.intersect(nap.IntervalSet(rip_ep))
        
    ###POST (20+ min)
         
        # ripp_post_rest = rip_ep.intersect(nap.IntervalSet(start = sub_nrem_ep2.iloc[-1]['end'], end = epochs['sleep'].loc[[1]].intersect(sws_ep).iloc[-1]['end']))
        # ripp_post_rest = rip_ep.intersect(nap.IntervalSet(start = sub_nrem_ep.iloc[-1]['end'], end = epochs['sleep'].loc[[1]].intersect(sws_ep).iloc[-1]['end']))
          
        
#%% 
      
        
        # t = time.time()
        
        # print(time.time() - t)
                    

        binned_pre2 = binned_spikes.restrict(nap.IntervalSet(ripp_pre)).as_dataframe()
        binned_wake2 = binned_spikes.restrict(nap.IntervalSet(moving_ep)).as_dataframe()
        binned_post2_10 = binned_spikes.restrict(nap.IntervalSet(ripp_post_10)).as_dataframe()
                
        todrop = []
        for i in binned_pre2.columns:
            if ((binned_pre2[i] == 0).all() == True) or ((binned_wake2[i] == 0).all() == True) or ((binned_post2_10[i] == 0).all() == True):
                todrop.append(i)
        
        binned_pre2 = binned_pre2.drop(todrop, axis=1)
        binned_wake2 = binned_wake2.drop(todrop, axis=1)
        binned_post2_10 = binned_post2_10.drop(todrop, axis=1)
        
        # binned_post2_20 = binned_spikes.restrict(nap.IntervalSet(ripp_post_20)).as_dataframe()
        # binned_post2_rest = binned_spikes.restrict(nap.IntervalSet(ripp_post_rest)).as_dataframe()

        
        # t = time.time()
        Cpre =  np.corrcoef(binned_pre2.T) 
        Cwake = np.corrcoef(binned_wake2.T)
        Cpost_10 = np.corrcoef(binned_post2_10.T)
        # Cpost_20 = np.corrcoef(binned_post2_20.T)
        # Cpost_rest = np.corrcoef(binned_post2_rest.T)
        # print(time.time() - t)
        
        ix = np.triu_indices(Cpre.shape[1])
        
        uptr_pre = Cpre[ix].flatten() 
        uptr_wake = Cwake[ix].flatten() 
        uptr_post_10 = Cpost_10[ix].flatten()
        # uptr_post_20 = Cpost_20[ix].flatten()
        # uptr_post_rest = Cpost_rest[ix].flatten()
        
        Rwpost_10, _ = pearsonr(uptr_wake, uptr_post_10)
        # Rwpost_20, _ = pearsonr(uptr_wake, uptr_post_20)
        # Rwpost_rest, _ = pearsonr(uptr_wake, uptr_post_rest)
        Rwpre, _ = pearsonr(uptr_wake, uptr_pre)
        Rprepost_10, _ = pearsonr(uptr_pre, uptr_post_10)
        # Rprepost_20, _ = pearsonr(uptr_pre, uptr_post_20)
        # Rprepost_rest, _ = pearsonr(uptr_pre, uptr_post_rest)
        
        EV_10 = ((Rwpost_10 - (Rwpre * Rprepost_10)) / np.sqrt((1 - Rwpre**2) * (1 - Rprepost_10**2)))**2
        REV_10 = ((Rwpre - (Rwpost_10 * Rprepost_10)) / np.sqrt((1 - Rwpost_10**2) * (1 - Rprepost_10**2)))**2
        
        # EV_20 = ((Rwpost_20 - (Rwpre * Rprepost_20)) / np.sqrt((1 - Rwpre**2) * (1 - Rprepost_20**2)))**2
        # REV_20 = ((Rwpre - (Rwpost_20 * Rprepost_20)) / np.sqrt((1 - Rwpost_20**2) * (1 - Rprepost_20**2)))**2
        
        # EV_rest = ((Rwpost_rest - (Rwpre * Rprepost_rest)) / np.sqrt((1 - Rwpre**2) * (1 - Rprepost_rest**2)))**2
        # REV_rest = ((Rwpre - (Rwpost_rest * Rprepost_rest)) / np.sqrt((1 - Rwpost_rest**2) * (1 - Rprepost_rest**2)))**2
               
        
        # if np.isnan(EV_10) == False and np.isnan(REV_10) == False and np.isnan(EV_20) == False and np.isnan(REV_20) == False and np.isnan(EV_rest) == False and np.isnan(REV_rest) == False:
        # if np.isnan(EV_10) == False and np.isnan(REV_10) == False  and np.isnan(EV_rest) == False and np.isnan(REV_rest) == False:
        if np.isnan(EV_10) == False and np.isnan(REV_10) == False  :
        
            if isWT == 1:
                
                evdiff_10_wt.append(EV_10 - REV_10)
                # evdiff_20_wt.append(EV_20 - REV_20)
                # evdiff_rest_wt.append(EV_rest - REV_rest)
                
                ev_wt_10.append(EV_10)
                rev_wt_10.append(REV_10)
                # ev_wt_20.append(EV_20)
                # rev_wt_20.append(REV_20)
                # ev_wt_rest.append(EV_rest)
                # rev_wt_rest.append(REV_rest)
                
                
                ncells_wt.append(len(pyr))
            
            else:
                
                evdiff_10_ko.append(EV_10 - REV_10)
                # evdiff_20_ko.append(EV_20 - REV_20)
                # evdiff_rest_ko.append(EV_rest - REV_rest)
                
                ev_ko_10.append(EV_10)
                rev_ko_10.append(REV_10)
                # ev_ko_20.append(EV_20)
                # rev_ko_20.append(REV_20)
                # ev_ko_rest.append(EV_rest)
                # rev_ko_rest.append(REV_rest)

                
                ncells_ko.append(len(pyr))
                
                # sys.exit()
                
#%% Organize EV and REV to plot 

# e1 = np.array(['ev_wt' for x in range(len(ev_wt))])
# r1 = np.array(['rev_wt' for x in range(len(rev_wt))])

# e2 = np.array(['ev_ko' for x in range(len(ev_ko))])
# r2 = np.array(['rev_ko' for x in range(len(rev_ko))])

# types = np.hstack([e1, r1, e2, r2])

# allEV = []
# allEV.extend(ev_wt)
# allEV.extend(rev_wt)
# allEV.extend(ev_ko)
# allEV.extend(rev_ko)

# summ = pd.DataFrame(data = [allEV, types], index = ['EV', 'type']).T

#%% Organize and plot EV differences 

e1 = np.array(['ev_10_wt' for x in range(len(evdiff_10_wt))])
# e2 = np.array(['ev_20_wt' for x in range(len(evdiff_20_wt))])
# e3 = np.array(['ev_rest_wt' for x in range(len(evdiff_rest_wt))])

e4 = np.array(['ev_10_ko' for x in range(len(evdiff_10_ko))])
# e5 = np.array(['ev_20_ko' for x in range(len(evdiff_20_ko))])
# e6 = np.array(['ev_rest_ko' for x in range(len(evdiff_rest_ko))])

# types = np.hstack([e1, e4, e2, e5, e3, e6])
types = np.hstack([e1, e4])

allEV = []
allEV.extend(evdiff_10_wt)
allEV.extend(evdiff_10_ko)
# allEV.extend(evdiff_20_wt)
# allEV.extend(evdiff_20_ko)
# allEV.extend(evdiff_rest_wt)
# allEV.extend(evdiff_rest_ko)

summ = pd.DataFrame(data = [allEV, types], index = ['EV', 'type']).T

###Plotting 

plt.figure()
plt.title('Reactivation')
sns.set_style('white')
palette = ['royalblue', 'indianred', 'royalblue', 'indianred','royalblue', 'indianred'] 
ax = sns.violinplot( x = summ['type'], y=summ['EV'].astype(float) , data = summ, dodge=False,
                    palette = palette,cut = 2,
                    scale="width", inner=None)
ax.tick_params(bottom=True, left=True)
xlim = ax.get_xlim()
ylim = ax.get_ylim()
for violin in ax.collections:
    x0, y0, width, height = violin.get_paths()[0].get_extents().bounds
    violin.set_clip_path(plt.Rectangle((x0, y0), width / 2, height, transform=ax.transData))
sns.boxplot(x = summ['type'], y=summ['EV'].astype(float) , data = summ, saturation=1, showfliers=False,
            width=0.3, boxprops={'zorder': 3, 'facecolor': 'none'}, ax=ax)
old_len_collections = len(ax.collections)
sns.stripplot(x = summ['type'], y = summ['EV'].astype(float), data = summ, color = 'k', dodge=False, ax=ax)
# sns.swarmplot(x = wakedf['type'], y = wakedf['rate'].astype(float), data = wakedf, color = 'k', dodge=False, ax=ax)
for dots in ax.collections[old_len_collections:]:
    dots.set_offsets(dots.get_offsets())
ax.set_xlim(xlim)
ax.set_ylim(ylim)
plt.ylabel('EV - REV')
ax.set_box_aspect(1)



#%% Plotting 

label = ['EV', 'REV']
x = [0, 0.35]# the label locations
width = 0.3  # the width of the bars

plt.figure()
plt.suptitle('Reactivation - last 10 min pre; all post')
plt.subplot(121)
plt.boxplot(ev_wt_10, positions=[0], showfliers=False, patch_artist=True, boxprops=dict(facecolor='royalblue', color='royalblue'),
            capprops=dict(color='royalblue'),
            whiskerprops=dict(color='royalblue'),
            medianprops=dict(color='white', linewidth = 2))
plt.boxplot(rev_wt_10, positions=[0.3], showfliers=False, patch_artist=True, boxprops=dict(facecolor='lightsteelblue', color='lightsteelblue'),
            capprops=dict(color='lightsteelblue'),
            whiskerprops=dict(color='lightsteelblue'),
            medianprops=dict(color='white', linewidth = 2))

plt.xticks([0, 0.3],['EV', 'REV'])
plt.ylim([-0.1,1])
plt.title('WT')
plt.ylabel('Explained Variance')
pval = np.vstack([(ev_wt_10), (rev_wt_10)])
plt.plot(x, np.vstack(pval), 'o-', color = 'k', zorder = 3, markersize = 3, linewidth = 1 )
plt.gca().set_box_aspect(1)

plt.subplot(122)
plt.boxplot(ev_ko_10, positions=[0], showfliers=False, patch_artist=True, boxprops=dict(facecolor='indianred', color='indianred'),
            capprops=dict(color='indianred'),
            whiskerprops=dict(color='indianred'),
            medianprops=dict(color='white', linewidth = 2))
plt.boxplot(rev_ko_10, positions=[0.3], showfliers=False, patch_artist=True, boxprops=dict(facecolor='lightcoral', color='lightcoral'),
            capprops=dict(color='lightcoral'),
            whiskerprops=dict(color='lightcoral'),
            medianprops=dict(color='white', linewidth = 2))

plt.xticks([0, 0.3],['EV', 'REV'])
plt.title('KO')
plt.ylabel('Explained Variance')
plt.ylim([-0.1,1])
p2 = np.vstack([(ev_ko_10), (rev_ko_10)])
plt.plot(x, np.vstack(p2), 'o-', color = 'k', zorder = 3, markersize = 3, linewidth = 1 )
plt.gca().set_box_aspect(1)


# plt.figure()
# plt.suptitle('Reactivation - 10-20 min post')
# plt.subplot(121)
# plt.boxplot(ev_wt_20, positions=[0], showfliers=False, patch_artist=True, boxprops=dict(facecolor='royalblue', color='royalblue'),
#             capprops=dict(color='royalblue'),
#             whiskerprops=dict(color='royalblue'),
#             medianprops=dict(color='white', linewidth = 2))
# plt.boxplot(rev_wt_20, positions=[0.3], showfliers=False, patch_artist=True, boxprops=dict(facecolor='lightsteelblue', color='lightsteelblue'),
#             capprops=dict(color='lightsteelblue'),
#             whiskerprops=dict(color='lightsteelblue'),
#             medianprops=dict(color='white', linewidth = 2))

# plt.xticks([0, 0.3],['EV', 'REV'])
# plt.ylim([-0.1,1])
# plt.title('WT')
# plt.ylabel('Explained Variance')
# pval = np.vstack([(ev_wt_20), (rev_wt_20)])
# plt.plot(x, np.vstack(pval), 'o-', color = 'k', zorder = 3, markersize = 3, linewidth = 1 )
# plt.gca().set_box_aspect(1)

# plt.subplot(122)
# plt.boxplot(ev_ko_20, positions=[0], showfliers=False, patch_artist=True, boxprops=dict(facecolor='indianred', color='indianred'),
#             capprops=dict(color='indianred'),
#             whiskerprops=dict(color='indianred'),
#             medianprops=dict(color='white', linewidth = 2))
# plt.boxplot(rev_ko_20, positions=[0.3], showfliers=False, patch_artist=True, boxprops=dict(facecolor='lightcoral', color='lightcoral'),
#             capprops=dict(color='lightcoral'),
#             whiskerprops=dict(color='lightcoral'),
#             medianprops=dict(color='white', linewidth = 2))

# plt.xticks([0, 0.3],['EV', 'REV'])
# plt.title('KO')
# plt.ylabel('Explained Variance')
# plt.ylim([-0.1,1])
# p2 = np.vstack([(ev_ko_20), (rev_ko_20)])
# plt.plot(x, np.vstack(p2), 'o-', color = 'k', zorder = 3, markersize = 3, linewidth = 1 )
# plt.gca().set_box_aspect(1)


# plt.figure()
# plt.suptitle('Reactivation - 20+ min post')
# plt.subplot(121)
# plt.boxplot(ev_wt_rest, positions=[0], showfliers=False, patch_artist=True, boxprops=dict(facecolor='royalblue', color='royalblue'),
#             capprops=dict(color='royalblue'),
#             whiskerprops=dict(color='royalblue'),
#             medianprops=dict(color='white', linewidth = 2))
# plt.boxplot(rev_wt_rest, positions=[0.3], showfliers=False, patch_artist=True, boxprops=dict(facecolor='lightsteelblue', color='lightsteelblue'),
#             capprops=dict(color='lightsteelblue'),
#             whiskerprops=dict(color='lightsteelblue'),
#             medianprops=dict(color='white', linewidth = 2))

# plt.xticks([0, 0.3],['EV', 'REV'])
# plt.ylim([-0.1,1])
# plt.title('WT')
# plt.ylabel('Explained Variance')
# pval = np.vstack([(ev_wt_rest), (rev_wt_rest)])
# plt.plot(x, np.vstack(pval), 'o-', color = 'k', zorder = 3, markersize = 3, linewidth = 1 )
# plt.gca().set_box_aspect(1)

# plt.subplot(122)
# plt.boxplot(ev_ko_rest, positions=[0], showfliers=False, patch_artist=True, boxprops=dict(facecolor='indianred', color='indianred'),
#             capprops=dict(color='indianred'),
#             whiskerprops=dict(color='indianred'),
#             medianprops=dict(color='white', linewidth = 2))
# plt.boxplot(rev_ko_rest, positions=[0.3], showfliers=False, patch_artist=True, boxprops=dict(facecolor='lightcoral', color='lightcoral'),
#             capprops=dict(color='lightcoral'),
#             whiskerprops=dict(color='lightcoral'),
#             medianprops=dict(color='white', linewidth = 2))

# plt.xticks([0, 0.3],['EV', 'REV'])
# plt.title('KO')
# plt.ylabel('Explained Variance')
# plt.ylim([-0.1,1])
# p2 = np.vstack([(ev_ko_rest), (rev_ko_rest)])
# plt.plot(x, np.vstack(p2), 'o-', color = 'k', zorder = 3, markersize = 3, linewidth = 1 )
# plt.gca().set_box_aspect(1)

#%% Stats 

# z_wt, p_wt = wilcoxon(ev_wt, rev_wt)
# z_ko, p_ko = wilcoxon(ev_ko, rev_ko)

#%% 

evdiff_wt = [i - j for i, j in zip(ev_wt_10, rev_wt_10)]
evdiff_ko = [i - j for i, j in zip(ev_ko_10, rev_ko_10)]

plt.figure()
plt.scatter(ncells_wt, evdiff_wt, color = 'royalblue', label = 'WT')
plt.scatter(ncells_ko, evdiff_ko, color = 'indianred', label = 'KO')
plt.xlabel('#PYR cells')
plt.ylabel('EV - REV')
plt.legend(loc = 'upper right')