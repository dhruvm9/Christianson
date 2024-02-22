#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 13:33:10 2024

@author: dhruv
"""

import numpy as np 
import pandas as pd
import nwbmatic as ntm
import pynapple as nap 
import pickle
import scipy.io
import os, sys
import seaborn as sns
import matplotlib.pyplot as plt 
from scipy.stats import mannwhitneyu
from itertools import combinations

#%% 

def rotate_via_numpy(xy, radians):
    """xy is a tuple or array """
    x, y = xy
    c, s = np.cos(radians), np.sin(radians)
    j = np.array([[c, s], [-s, c]])
    m = np.dot(j, [x, y])

    return float(m.T[0]), float(m.T[1])

#%% 

data_directory = '/media/dhruv/Expansion/Processed'
datasets = np.genfromtxt(os.path.join(data_directory,'dataset_DM.list'), delimiter = '\n', dtype = str, comments = '#')

zerolag_pre_wt = []
zerolag_post_wt = []

zerolag_pre_ko = []
zerolag_post_ko = []

for s in datasets[1:]:
    print(s)
    name = s.split('-')[0]
       
    path = os.path.join(data_directory, s)
    
    if name == 'B2613' or name == 'B2618':
        isWT = 0
    else: isWT = 1 

    # sp2 = np.load(os.path.join(path, 'spikedata.npz'), allow_pickle = True)
    sp2 = np.load(os.path.join(path, 'spikedata_0.55.npz'), allow_pickle = True)
    time_support = nap.IntervalSet(sp2['start'], sp2['end'])
    tsd = nap.Tsd(t=sp2['t'], d=sp2['index'], time_support = time_support)
    spikes = tsd.to_tsgroup()
    spikes.set_info(group = sp2['group'], location = sp2['location'], celltype = sp2['celltype'], tr2pk = sp2['tr2pk'])
    
    data = ntm.load_session(path, 'neurosuite')
    epochs = data.epochs
    position = data.position
    
    file = os.path.join(path, s +'.sws.evt')
    sws_ep = nap.IntervalSet(data.read_neuroscope_intervals(name = 'SWS', path2file = file))
    
#%% Rotate position 

    rot_pos = []
        
    xypos = np.array(position[['x', 'z']])
      
    for i in range(len(xypos)):
        newx, newy = rotate_via_numpy(xypos[i], 1.05)
        rot_pos.append((newx, newy))
        
    rot_pos = nap.TsdFrame(t = position.index.values, d = rot_pos, columns = ['x', 'z'])
    
#%% Compute speed during wake 

    sleep_pre_ep = sws_ep.intersect(epochs['sleep'].loc[[0]])
    sleep_post_ep = sws_ep.intersect(epochs['sleep'].loc[[1]])
    
    spikes_by_celltype = spikes.getby_category('celltype')
    
    if 'pyr' in spikes._metadata['celltype'].values:
        pyr = spikes_by_celltype['pyr']
        
        keep = []
        
        for i in pyr.index:
            if pyr.restrict(nap.IntervalSet(epochs['wake'].loc[[0]]))._metadata['rate'][i] > 0.5:
                keep.append(i)
    
        pyr2 = pyr[keep]
    
        if len(pyr2) > 2:
            
            speedbinsize = np.diff(rot_pos.index.values)[0]
            
            time_bins = np.arange(rot_pos.index[0], rot_pos.index[-1] + speedbinsize, speedbinsize)
            index = np.digitize(rot_pos.index.values, time_bins)
            tmp = rot_pos.as_dataframe().groupby(index).mean()
            tmp.index = time_bins[np.unique(index)-1]+(speedbinsize)/2
            distance = np.sqrt(np.power(np.diff(tmp['x']), 2) + np.power(np.diff(tmp['z']), 2)) * 100 #in cm
            speed = nap.Tsd(t = tmp.index.values[0:-1]+ speedbinsize/2, d = distance/speedbinsize) # in cm/s
         
            moving_ep = nap.IntervalSet(speed.threshold(2).time_support) #Epochs in which speed is > 2 cm/s
            ep = moving_ep.intersect(epochs['wake'].loc[[0]])
        
#%% Compute place fields 

            placefields, binsxy = nap.compute_2d_tuning_curves(group = pyr2, features = rot_pos, ep = ep, nb_bins=20)
            
            pf_centre = []
            
            for i in pyr2.keys(): 
                placefields[i][np.isnan(placefields[i])] = 0
                placefields[i] = scipy.ndimage.gaussian_filter(placefields[i], 1)
                pf_centre.append(np.unravel_index(np.argmax(placefields[i], axis=None), placefields[i].shape))
            
            pairs = list(combinations(range(len(pyr2)), 2))
            
            # keep_pairs = []
            
            # for i,j in pairs:
            #     diff = tuple(np.subtract(pf_centre[i], pf_centre[j]))
            #     if abs(diff[0]) + abs(diff[1]) <= 10:
            #         keep_pairs.append(tuple([i,j]))
                
            # print(keep_pairs)
                
#%% PYR cell pair cross corrs
       
            # if len(keep_pairs) > 0:
            xc_pyr_wake = nap.compute_crosscorrelogram(pyr2, binsize = 0.01, windowsize = 0.3 , ep = nap.IntervalSet(epochs['wake']))
            # xc_pyr_wake = xc_pyr_wake[xc_pyr_wake.columns[61:]]
            # xc_pyr_wake = xc_pyr_wake[pyr2.index[keep_pairs[0:20]]]
            
            xc_pyr_pre = nap.compute_crosscorrelogram(pyr2, binsize = 0.01, windowsize = 0.3 , ep = nap.IntervalSet(sleep_pre_ep))
            
            # xc_pyr_pre = xc_pyr_pre[xc_pyr_pre.columns[61:]]
            # xc_pyr_pre = xc_pyr_pre[pyr2.index[keep_pairs[0:20]]]
            
            xc_pyr_post = nap.compute_crosscorrelogram(pyr2, binsize = 0.01, windowsize = 0.3 , ep = nap.IntervalSet(sleep_post_ep))
            
            # xc_pyr_post = xc_pyr_post[xc_pyr_post.columns[61:]]
            # xc_pyr_post = xc_pyr_post[pyr2.index[keep_pairs[0:20]]]
                
            
            if isWT == 1:
                zerolag_pre_wt.extend(xc_pyr_pre.loc[0].values)
                zerolag_post_wt.extend(xc_pyr_post.loc[0].values)    
                
            else:
                zerolag_pre_ko.extend(xc_pyr_pre.loc[0].values)
                zerolag_post_ko.extend(xc_pyr_post.loc[0].values)    
                
                
                                       
#%% 

        
        # k = 0
        # for i,n in xc_pyr_wake.columns:
        #     plt.figure()
        #     # plt.subplot(14,14,k+1)        
        #     plt.subplot(131)
        #     plt.title(str(i))
        #     plt.imshow(placefields[i], extent=(binsxy[1][0],binsxy[1][-1],binsxy[0][0],binsxy[0][-1]), cmap = 'jet')
        #     plt.subplot(132)
        #     plt.title(str(n))
        #     plt.imshow(placefields[n], extent=(binsxy[1][0],binsxy[1][-1],binsxy[0][0],binsxy[0][-1]), cmap = 'jet')
        #     plt.subplot(133)
        #     plt.plot(xc_pyr_wake[(i,n)], label = 'wake')
        #     plt.plot(xc_pyr_pre[(i,n)], label = 'pre')
        #     plt.plot(xc_pyr_post[(i,n)], label = 'post')
        #     plt.gca().set_box_aspect(1)
        #     plt.legend(loc = 'upper right')
        #     # k +=1
    
#%% 
   
    else: pyr = []
    
    del pyr
    
#%% Plotting 

# label = ['Pre_WT', 'Post_WT']
# x = [0, 0.35]# the label locations
# width = 0.3  # the width of the bars

# plt.figure()
# plt.suptitle('Zero lag Xcorr')
# plt.subplot(121)
# plt.boxplot(zerolag_pre_wt, positions=[0], showfliers=False, patch_artist=True, boxprops=dict(facecolor='royalblue', color='royalblue'),
#             capprops=dict(color='royalblue'),
#             whiskerprops=dict(color='royalblue'),
#             medianprops=dict(color='white', linewidth = 2))
# plt.boxplot(zerolag_post_wt, positions=[0.3], showfliers=False, patch_artist=True, boxprops=dict(facecolor='lightsteelblue', color='lightsteelblue'),
#             capprops=dict(color='lightsteelblue'),
#             whiskerprops=dict(color='lightsteelblue'),
#             medianprops=dict(color='white', linewidth = 2))

# plt.xticks([0, 0.3],['Pre', 'Post'])
# # plt.ylim([-0.1,1])
# plt.title('WT')
# plt.ylabel('Zero lag magnitude')
# pval = np.vstack([(zerolag_pre_wt), (zerolag_post_wt)])
# # plt.plot(x, np.vstack(pval), 'o-', color = 'k', zorder = 3, markersize = 3, linewidth = 1 )
# plt.gca().set_box_aspect(1)

# plt.subplot(122)
# plt.boxplot(zerolag_pre_ko, positions=[0], showfliers=False, patch_artist=True, boxprops=dict(facecolor='indianred', color='indianred'),
#             capprops=dict(color='indianred'),
#             whiskerprops=dict(color='indianred'),
#             medianprops=dict(color='white', linewidth = 2))
# plt.boxplot(zerolag_post_ko, positions=[0.3], showfliers=False, patch_artist=True, boxprops=dict(facecolor='lightcoral', color='lightcoral'),
#             capprops=dict(color='lightcoral'),
#             whiskerprops=dict(color='lightcoral'),
#             medianprops=dict(color='white', linewidth = 2))

# plt.xticks([0, 0.3],['Pre', 'Post'])
# plt.title('KO')
# plt.ylabel('Zero lag magnitude')
# # plt.ylim([-0.1,1])
# p2 = np.vstack([(zerolag_pre_ko), (zerolag_post_ko)])
# # plt.plot(x, np.vstack(p2), 'o-', color = 'k', zorder = 3, markersize = 3, linewidth = 1 )
# plt.gca().set_box_aspect(1)

#%% Plotting examples

#### KO 

# plt.figure()
# plt.suptitle('High wake corr')
# plt.subplot(231)
# plt.plot(xc_pyr_pre[(17,21)], label = 'pre')
# plt.plot(xc_pyr_wake[(17,21)], label = 'wake')
# plt.plot(xc_pyr_post[(17,21)], label = 'post')
# plt.legend(loc = 'upper right')
# plt.subplot(232)
# plt.plot(xc_pyr_pre[(16,17)], label = 'pre')
# plt.plot(xc_pyr_wake[(16,17)], label = 'wake')
# plt.plot(xc_pyr_post[(16,17)], label = 'post')
# plt.legend(loc = 'upper right')
# plt.subplot(233)
# plt.plot(xc_pyr_pre[(17,26)], label = 'pre')
# plt.plot(xc_pyr_wake[(17,26)], label = 'wake')
# plt.plot(xc_pyr_post[(17,26)], label = 'post')
# plt.legend(loc = 'upper right')
# plt.subplot(234)
# plt.plot(xc_pyr_pre[(21,26)], label = 'pre')
# plt.plot(xc_pyr_wake[(21,26)], label = 'wake')
# plt.plot(xc_pyr_post[(21,26)], label = 'post')
# plt.legend(loc = 'upper right')
# plt.subplot(235)
# plt.plot(xc_pyr_pre[(36,46)], label = 'pre')
# plt.plot(xc_pyr_wake[(36,46)], label = 'wake')
# plt.plot(xc_pyr_post[(36,46)], label = 'post')
# plt.legend(loc = 'upper right')
# plt.subplot(236)
# plt.plot(xc_pyr_pre[(42,48)], label = 'pre')
# plt.plot(xc_pyr_wake[(42,48)], label = 'wake')
# plt.plot(xc_pyr_post[(42,48)], label = 'post')
# plt.legend(loc = 'upper right')
# plt.tight_layout()  

# plt.figure()
# plt.suptitle('Low wake corr')
# plt.subplot(231)
# plt.plot(xc_pyr_pre[(37,44)], label = 'pre')
# plt.plot(xc_pyr_wake[(37,44)], label = 'wake')
# plt.plot(xc_pyr_post[(37,44)], label = 'post')
# plt.legend(loc = 'upper right')
# plt.subplot(232)
# plt.plot(xc_pyr_pre[(16,45)], label = 'pre')
# plt.plot(xc_pyr_wake[(16,45)], label = 'wake')
# plt.plot(xc_pyr_post[(16,45)], label = 'post')
# plt.legend(loc = 'upper right')
# plt.subplot(233)
# plt.plot(xc_pyr_pre[(17,48)], label = 'pre')
# plt.plot(xc_pyr_wake[(17,48)], label = 'wake')
# plt.plot(xc_pyr_post[(17,48)], label = 'post')
# plt.legend(loc = 'upper right')
# plt.subplot(234)
# plt.plot(xc_pyr_pre[(16,48)], label = 'pre')
# plt.plot(xc_pyr_wake[(16,48)], label = 'wake')
# plt.plot(xc_pyr_post[(16,48)], label = 'post')
# plt.legend(loc = 'upper right')
# plt.subplot(235)
# plt.plot(xc_pyr_pre[(15,16)], label = 'pre')
# plt.plot(xc_pyr_wake[(15,16)], label = 'wake')
# plt.plot(xc_pyr_post[(15,16)], label = 'post')
# plt.legend(loc = 'upper right')

#### WT 

# plt.figure()
# plt.suptitle('High wake corr')
# plt.subplot(231)
# plt.plot(xc_pyr_pre[(1,5)], label = 'pre')
# plt.plot(xc_pyr_wake[(1,5)], label = 'wake')
# plt.plot(xc_pyr_post[(1,5)], label = 'post')
# plt.legend(loc = 'upper right')
# plt.subplot(232)
# plt.plot(xc_pyr_pre[(4,5)], label = 'pre')
# plt.plot(xc_pyr_wake[(4,5)], label = 'wake')
# plt.plot(xc_pyr_post[(4,5)], label = 'post')
# plt.legend(loc = 'upper right')
# plt.subplot(233)
# plt.plot(xc_pyr_pre[(5,11)], label = 'pre')
# plt.plot(xc_pyr_wake[(5,11)], label = 'wake')
# plt.plot(xc_pyr_post[(5,11)], label = 'post')
# plt.legend(loc = 'upper right')
# plt.subplot(234)
# plt.plot(xc_pyr_pre[(11,12)], label = 'pre')
# plt.plot(xc_pyr_wake[(11,12)], label = 'wake')
# plt.plot(xc_pyr_post[(11,12)], label = 'post')
# plt.legend(loc = 'upper right')
# plt.subplot(235)
# plt.plot(xc_pyr_pre[(12,18)], label = 'pre')
# plt.plot(xc_pyr_wake[(12,18)], label = 'wake')
# plt.plot(xc_pyr_post[(12,18)], label = 'post')
# plt.legend(loc = 'upper right')

# plt.figure()
# plt.suptitle('Low wake corr')
# plt.subplot(231)
# plt.plot(xc_pyr_pre[(4,10)], label = 'pre')
# plt.plot(xc_pyr_wake[(4,10)], label = 'wake')
# plt.plot(xc_pyr_post[(4,10)], label = 'post')
# plt.legend(loc = 'upper right')
# plt.subplot(232)
# plt.plot(xc_pyr_pre[(6,9)], label = 'pre')
# plt.plot(xc_pyr_wake[(6,9)], label = 'wake')
# plt.plot(xc_pyr_post[(6,9)], label = 'post')
# plt.legend(loc = 'upper right')
# plt.subplot(233)
# plt.plot(xc_pyr_pre[(4,15)], label = 'pre')
# plt.plot(xc_pyr_wake[(4,15)], label = 'wake')
# plt.plot(xc_pyr_post[(4,15)], label = 'post')
# plt.legend(loc = 'upper right')
# plt.subplot(234)
# plt.plot(xc_pyr_pre[(9,12)], label = 'pre')
# plt.plot(xc_pyr_wake[(9,12)], label = 'wake')
# plt.plot(xc_pyr_post[(9,12)], label = 'post')
# plt.legend(loc = 'upper right')
# plt.subplot(235)
# plt.plot(xc_pyr_pre[(6,17)], label = 'pre')
# plt.plot(xc_pyr_wake[(6,17)], label = 'wake')
# plt.plot(xc_pyr_post[(6,17)], label = 'post')
# plt.legend(loc = 'upper right')
