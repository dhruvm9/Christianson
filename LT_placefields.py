#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 09:51:31 2025

@author: dhruv
"""

import numpy as np 
import pandas as pd 
import nwbmatic as ntm
import scipy.io
import pynapple as nap 
import os, sys
import matplotlib.pyplot as plt 
import seaborn as sns
import warnings
from scipy.stats import mannwhitneyu, wilcoxon, ks_2samp, kendalltau
from functions_DM import *

#%% 


warnings.filterwarnings("ignore")

data_directory = '/media/dhruv/Expansion/Processed/LinearTrack'
datasets = np.genfromtxt(os.path.join(data_directory,'dataset_DM.list'), delimiter = '\n', dtype = str, comments = '#')

npyr2_wt = []
npyr2_ko = []

npyr3_wt = []
npyr3_ko = []

allcells = []
pyrcells = []
fscells = []

fwd_corrs_wt = []
fwd_sig_wt = []

rev_corrs_wt = []
rev_sig_wt = []

fwd_corrs_ko = []
fwd_sig_ko = []

rev_corrs_ko = []
rev_sig_ko = []

sigfrac_fwd_wt = []
sigfrac_fwd_ko = []

sigfrac_rev_wt = []
sigfrac_rev_ko = []

pos_slope_fwd_wt = []
pos_slope_fwd_ko = []
neg_slope_fwd_wt = []
neg_slope_fwd_ko = []
pos_slope_rev_wt = []
pos_slope_rev_ko = []
neg_slope_rev_wt = []
neg_slope_rev_ko = []

pos_dist_fwd_wt = []
pos_dist_fwd_ko = []
neg_dist_fwd_wt = []
neg_dist_fwd_ko = []
pos_dist_rev_wt = []
pos_dist_rev_ko = []
neg_dist_rev_wt = []
neg_dist_rev_ko = []

pos_cell_fwd_wt = []
pos_cell_fwd_ko = []
neg_cell_fwd_wt = []
neg_cell_fwd_ko = []
pos_cell_rev_wt = []
pos_cell_rev_ko = []
neg_cell_rev_wt = []
neg_cell_rev_ko = []


KOmice = ['B2613', 'B2618', 'B2627', 'B2628', 'B3805', 'B3813', 'B4701', 'B4704', 'B4709']

for s in datasets:
    print(s)
    name = s.split('-')[0]
       
    path = os.path.join(data_directory, s)
    
    if name in KOmice:
        isWT = 0
    else: isWT = 1 

    spikes = nap.load_file(os.path.join(path, 'spikedata_0.55.npz'))
    
    allcells.append(len(spikes))
    
    data = ntm.load_session(path, 'neurosuite')
    epochs = data.epochs
    position = data.position
    
       
#%% Rotate position 

    rot_pos = []
        
    xypos = np.array(position[['x', 'z']])
    
    if name in ['B4701', 'B4702', 'B4704', 'B4705', 'B4707']:
        rad = 0.2
    else: rad = 0    
            
    for i in range(len(xypos)):
        newx, newy = rotate_via_numpy(xypos[i], rad)
        rot_pos.append((newx, newy))
        
    rot_pos = nap.TsdFrame(t = position.index.values, d = rot_pos, columns = ['x', 'z'])
          
#%% Plot tracking 

    # plt.figure()
    # plt.title(s)
    # plt.plot(rot_pos['x'], rot_pos['z'])
    
#%% Get cells with wake rate more than 0.5Hz
        
    spikes_by_celltype = spikes.getby_category('celltype')
    
    if 'pyr' in spikes._metadata['celltype'].values:
        pyr = spikes_by_celltype['pyr']
        pyrcells.append(len(pyr))
    
    # else: 
    #     print('No EX cells in ' + s)
    #     continue
        
    if 'fs' in spikes._metadata['celltype'].values:
        fs = spikes_by_celltype['fs']
        fscells.append(len(fs))
        
    keep = []
    
    for i in pyr.index:
        if pyr.restrict(nap.IntervalSet(epochs['wake']))._metadata['rate'][i] > 0.5:
            keep.append(i)

    pyr2 = pyr[keep]

#%% Compute speed during wake 

    if len(pyr2) >= 10:
        # print('Yes!')
        
        if isWT == 1:
            npyr2_wt.append(len(pyr2))
        else: npyr2_ko.append(len(pyr2))
        
        speedbinsize = np.diff(rot_pos.index.values)[0]
        
        time_bins = np.arange(rot_pos.index[0], rot_pos.index[-1] + speedbinsize, speedbinsize)
        index = np.digitize(rot_pos.index.values, time_bins)
        tmp = rot_pos.as_dataframe().groupby(index).mean()
        tmp.index = time_bins[np.unique(index)-1]+(speedbinsize)/2
        distance = np.sqrt(np.power(np.diff(tmp['x']), 2) + np.power(np.diff(tmp['z']), 2)) * 100 #in cm
        speed = pd.Series(index = tmp.index.values[0:-1]+ speedbinsize/2, data = distance/speedbinsize) # in cm/s
        speed2 = speed.rolling(window = 25, win_type='gaussian', center=True, min_periods=1).mean(std=10) #Smooth over 200ms 
        speed2 = nap.Tsd(speed2)
        moving_ep = nap.IntervalSet(speed2.threshold(2).time_support) #Epochs in which speed is > 2 cm/s
        wake_ep = moving_ep.intersect(epochs['wake'])
        
#%% Split forward and reverse positions
                
        pos_x = rot_pos['x'].smooth(1)
        # pos_y = scipy.ndimage.gaussian_filter(rot_pos['z'].values, 100)
        
        if name in ['B4708', 'B4709']:        
            peaks = pos_x.threshold(0.05, method = 'aboveequal')
            troughs = pos_x.threshold(-0.17, method = 'belowequal')
            
        else: 
            peaks = pos_x.threshold(0, method = 'aboveequal')
            troughs = pos_x.threshold(-0.22, method = 'belowequal')
                       
        peaks_ts = peaks.time_support
        troughs_ts = troughs.time_support
        
        pstarts = peaks_ts['start']
        pends = peaks_ts['end']
        
        tstarts = troughs_ts['start']
        tends = troughs_ts['end']
        
        fwd_start = []
        fwd_end = []

        rev_start = []
        rev_end = []
        
        i_trough = 0
        i_peak = 0
        
        while i_trough < len(tends) and i_peak < len(pstarts):
            current_trough_end = tends[i_trough]

            while i_peak < len(pstarts) and pstarts[i_peak] <= current_trough_end:
                i_peak += 1
            if i_peak >= len(pstarts):
                break  # No more peaks left
                
            next_peak_start = pstarts[i_peak]
            next_peak_end = pends[i_peak]
            
            valid_troughs = tends[tends < next_peak_start]
            if len(valid_troughs) == 0:
                i_trough += 1
                continue  # No valid trough before this peak → skip
            latest_trough_end = valid_troughs[-1]
                        
            fwd_start.append(latest_trough_end)
            fwd_end.append(next_peak_start)
            
            while i_trough < len(tstarts) and tstarts[i_trough] <= next_peak_end:
                i_trough += 1
            if i_trough >= len(tstarts):
                break  # No more troughs left
                
            next_trough_start = tstarts[i_trough]
            
            valid_peaks = pends[pends < next_trough_start]
            if len(valid_peaks) == 0:
                i_peak += 1
                continue
            latest_peak_end = valid_peaks[-1]  # last valid one
                                        
            rev_start.append(latest_peak_end)
            rev_end.append(next_trough_start)
        
        fwd = nap.IntervalSet(start = fwd_start, end = fwd_end)
        rev = nap.IntervalSet(start = rev_start, end = rev_end)

        # plt.figure()
        # plt.title(s)
        # plt.plot(pos_x)
        # # plt.plot(peaks, 'x')
        # # plt.plot(troughs, 'o')
        # plt.plot(pos_x.restrict(fwd),'o')
        # plt.plot(pos_x.restrict(rev),'o')

        # plt.figure()
        # plt.title(s)
        # plt.plot(rot_pos['x'], rot_pos['z'], color = 'silver')
        # for i in range(len(fwd)):
        #     plt.plot(rot_pos['x'].restrict(fwd[i]), rot_pos['z'].restrict(fwd[i]), color = 'k')
        # for i in range(len(rev)):
        #     plt.plot(rot_pos['x'].restrict(rev[i]), rot_pos['z'].restrict(rev[i]), color = 'r')
           
    
#%% Compute place fields

        placefields_f = nap.compute_1d_tuning_curves(group = pyr2, 
                                                            feature = rot_pos['x'], 
                                                            ep = fwd, 
                                                            nb_bins=24)  
        placefields_f = placefields_f.fillna(0)
        for i in placefields_f.columns:
            placefields_f[i] = placefields_f[i].rolling(window = 5, win_type='gaussian', center=True, min_periods=1).mean(std=3)
               
        # px1 = occupancy_prob(rot_pos, fwd, nb_bins=24, norm = True)
        # SI_f = nap.compute_2d_mutual_info(placefields_f, fwdpos[['x', 'z']], ep = wake_ep)
        
        # ###Reverse position
        placefields_r = nap.compute_1d_tuning_curves(group = pyr2, 
                                                            feature = rot_pos['x'], 
                                                            ep = rev, 
                                                            nb_bins=24)  
        
        placefields_r = placefields_r.fillna(0)
        for i in placefields_r.columns:
            placefields_r[i] = placefields_r[i].rolling(window = 5, win_type='gaussian', center=True, min_periods=1).mean(std=3)
               
        # px2 = occupancy_prob(rot_pos, rev, nb_bins=24, norm = True)
        # SI_r = nap.compute_2d_mutual_info(placefields_r, revpos[['x', 'z']], ep = wake_ep)
                
                      
        # for i in pyr2.keys(): 
            
        #     placefields_f[i][np.isnan(placefields_f[i])] = 0
        #     placefields_f[i] = scipy.ndimage.gaussian_filter(placefields_f[i], 1.5, mode = 'nearest')
        #     masked_array = np.ma.masked_where(px1 == 0, placefields_f[i]) #should work fine without it 
        #     placefields_f[i] = masked_array
            
        #     ###Reverse fields
        #     placefields_r[i][np.isnan(placefields_r[i])] = 0
        #     placefields_r[i] = scipy.ndimage.gaussian_filter(placefields_r[i], 1.5, mode = 'nearest')
        #     masked_array = np.ma.masked_where(px2 == 0, placefields_r[i]) #should work fine without it 
        #     placefields_r[i] = masked_array
        
        # if isWT == 1:
        #     allspatialinfo_wt.extend(SI_f['SI'].tolist())
        #     allspatialinfo_wt.extend(SI_r['SI'].tolist())
            
        # else: 
        #     allspatialinfo_ko.extend(SI_f['SI'].tolist())
        #     allspatialinfo_ko.extend(SI_r['SI'].tolist())
        
    
#%% Plot place fields

        ref = pyr2.keys()
        nrows = int(np.sqrt(len(ref)))
        ncols = int(len(ref)/nrows)+1
        
        if name in ['B4708', 'B4709']:
            ylim = [0.125, 0.3]
        else: ylim = [0.05, 0.2]
                    
        # plt.figure()
        # for i,n in enumerate(pyr2):
        #     plt.suptitle(s)
        #     plt.subplot(nrows, ncols, i+1)
        #     plt.plot(rot_pos['x'], rot_pos['z'], color = 'grey')
        #     spk_pos1 = pyr2[n].value_from(rot_pos.restrict(fwd))
        #     spk_pos2 = pyr2[n].value_from(rot_pos.restrict(rev))
        #     plt.plot(spk_pos1['x'], spk_pos1['z'], 'o', color = 'k', markersize = 5, alpha = 0.2)
        #     plt.plot(spk_pos2['x'], spk_pos2['z'], 'o', color = 'r', markersize = 5, alpha = 0.2)
        #     plt.ylim(ylim)
        #     plt.gca().set_box_aspect(1)
            
        # plt.figure()
        # for i,n in enumerate(pyr2):
        #     plt.suptitle(s)
        #     plt.subplot(nrows, ncols, i+1)
        #     plt.plot(placefields_f[n], color = 'k') 
        #     plt.plot(placefields_r[n], color = 'r') 
        #     plt.xlabel('X (cm)')
        #     plt.ylabel('Firing rate (Hz)')
        #     plt.gca().set_box_aspect(1)
                            
                        
#%% Max X bin of place fields

        # xmax_f = []
        # xmax_r = []
        
        # for i in pyr2: 
        #     max_index_f = np.unravel_index(np.ma.argmax(placefields_f[i]), placefields_f[i].shape)
        #     max_index_r = np.unravel_index(np.ma.argmax(placefields_r[i]), placefields_r[i].shape)
        
        xmax_f = placefields_f.values.argmax(axis=0) #placefields_f.idxmax().values
        xmax_r = placefields_r.values.argmax(axis=0) #placefields_r.idxmax().values
        
        norm = plt.Normalize()     
        colorf = plt.cm.jet(norm([i/24 for i in xmax_f]))
        colorr = plt.cm.jet(norm([i/24 for i in xmax_r]))
        
        fxmax = pd.DataFrame(index = pyr2.index, data = xmax_f)
        rxmax = pd.DataFrame(index = pyr2.index, data = xmax_r)
                
        # sorted_fxmax = fxmax.sort_values(by = 0)
        # order_fwd = sorted_fxmax.index.tolist()
        
        # plt.figure()
        # plt.title(s)
        # for i,n in enumerate(order_fwd):
        #     # offset = 0.001*i
        #     plt.plot(10*(placefields_f[n]/placefields_f[n].max()) + (4*i), color = colorf[fxmax.index.get_loc(n)])
        # plt.gca().set_box_aspect(1)
            
     
        
#%% Rank correlation during population bursts
        
        # file = os.path.join(path, s +'.evt.py.wpb')
        # pb_ep = data.read_neuroscope_intervals(name = 'wpb', path2file = file)
        
        # zone_intervals = peaks_ts.union(troughs_ts)
                          
        # pb_reward = zone_intervals.intersect(pb_ep)
               
        file = os.path.join(path, s +'.evt.py.rip')
        pb_reward = data.read_neuroscope_intervals(name = 'rip', path2file = file)
                                 
        corrs_f = []
        sig_f = []
        
        corrs_r = []
        sig_r = []
                
        for i in range(len(pb_reward)):
            spk = pyr2.restrict(nap.IntervalSet(pb_reward[i]))
            
            firstspk = []
            tokeep = []
            
            for j in spk:
                
                if spk[j].t.size != 0:
                    firstspk.append(spk[j][0].start_time())
                    tokeep.append(j)
            
            if len(tokeep) >=5 :
            
                r_fwd, p_fwd = kendalltau(firstspk, fxmax.loc[tokeep])
                r_rev, p_rev = kendalltau(firstspk, rxmax.loc[tokeep])
                
                
                if abs(r_fwd) > abs(r_rev):
                    
                    corrs_f.append(r_fwd)
                    
                    corrs_shu = []
                         
                    for k in range(100):
                        fxmax_shu = fxmax.apply(np.random.permutation)
                        rshu, pshu = kendalltau(firstspk, fxmax_shu.loc[tokeep])
                        corrs_shu.append(rshu)
                        
                    if (r_fwd > np.percentile(corrs_shu, 95)): 
                    
                        m_fwd, b_fwd = np.polyfit(firstspk - pb_reward[i]['start'], fxmax.loc[tokeep], 1)
                        ybin = m_fwd*(firstspk - pb_reward[i]['start'])
                        
                        dist = fxmax.loc[tokeep].max() - fxmax.loc[tokeep].min()
                                                
                        if isWT == 1:
                            pos_slope_fwd_wt.append(m_fwd[0]*(0.35/24)) ###35 cm maze, 24 bins
                            pos_dist_fwd_wt.append(dist.values[0]*(35/24)) ###35 cm maze, 24 bins
                            pos_cell_fwd_wt.append(len(tokeep))
                            
                        else: 
                            pos_slope_fwd_ko.append(m_fwd[0]*(0.35/24)) ###35 cm maze, 24 bins
                            pos_dist_fwd_ko.append(dist.values[0]*(35/24))
                            pos_cell_fwd_ko.append(len(tokeep))
                                            
                        # plt.figure()
                        # plt.subplot(121)
                        # plt.title('R = ' + str(round(r_fwd,2)) + ', p = ' + str(round(p_fwd,2)))
                        # plt.scatter(firstspk - pb_reward[i]['start'], fxmax.loc[tokeep], marker = '|', color = colorf[fxmax.index.get_indexer(tokeep)], s = 500)
                        # plt.plot(firstspk - pb_reward[i]['start'], ybin + b_fwd, color = 'k')
                        # plt.ylabel('Max X bin')
                        # plt.ylim([-1, 25])
                        # plt.xlabel ('Time (s)')
                        # plt.gca().set_box_aspect(1)
                        # plt.subplot(122)
                        # plt.hist(corrs_shu, bins = 10)                
                        # plt.axvline(r_fwd, color = 'k')
                        # plt.gca().set_box_aspect(1)
                                        
                        sig_f.append(1)
                    
                    elif (r_fwd < np.percentile(corrs_shu, 5)):
                        
                        m_fwd, b_fwd = np.polyfit(firstspk - pb_reward[i]['start'], fxmax.loc[tokeep], 1)
                        ybin = m_fwd*(firstspk - pb_reward[i]['start'])
                        
                        dist = fxmax.loc[tokeep].max() - fxmax.loc[tokeep].min()
                        
                        if isWT == 1:
                            neg_slope_fwd_wt.append(m_fwd[0]*(0.35/24))
                            neg_dist_fwd_wt.append(dist.values[0]*(35/24))
                            neg_cell_fwd_wt.append(len(tokeep))
                            
                        else: 
                            neg_slope_fwd_ko.append(m_fwd[0]*(0.35/24))
                            neg_dist_fwd_ko.append(dist.values[0]*(35/24))
                            neg_cell_fwd_ko.append(len(tokeep))
                                                                        
                        # plt.figure()
                        # plt.subplot(121)
                        # plt.title('R = ' + str(round(r_fwd,2)) + ', p = ' + str(round(p_fwd,2)))
                        # plt.scatter(firstspk - pb_reward[i]['start'], fxmax.loc[tokeep], marker = '|', color = colorf[fxmax.index.get_indexer(tokeep)], s = 500)
                        # plt.plot(firstspk - pb_reward[i]['start'], ybin + b_fwd, color = 'k')
                        # plt.ylabel('Max X bin')
                        # plt.ylim([-1, 25])
                        # plt.xlabel ('Time (s)')
                        # plt.gca().set_box_aspect(1)
                        # plt.subplot(122)
                        # plt.hist(corrs_shu, bins = 10)                
                        # plt.axvline(r_fwd, color = 'k')
                        # plt.gca().set_box_aspect(1)
                                        
                        sig_f.append(1)
                                               
                    else: 
                        sig_f.append(0)
                        
                elif abs(r_rev) > abs(r_fwd):
                    
                    corrs_r.append(r_rev)
                    
                    corrs_shu = []
                         
                    for k in range(100):
                        rxmax_shu = rxmax.apply(np.random.permutation)
                        rshu, pshu = kendalltau(firstspk, rxmax_shu.loc[tokeep])
                        corrs_shu.append(rshu)
                        
                    if (r_rev > np.percentile(corrs_shu, 95)): 
                    
                        m_rev, b_rev = np.polyfit(firstspk - pb_reward[i]['start'], rxmax.loc[tokeep], 1)
                        ybin = m_rev*(firstspk - pb_reward[i]['start'])
                        
                        dist = rxmax.loc[tokeep].max() - rxmax.loc[tokeep].min()
                        
                        if isWT == 1:
                            pos_slope_rev_wt.append(m_rev[0]*(0.35/24))
                            pos_dist_rev_wt.append(dist.values[0]*(35/24))
                            pos_cell_rev_wt.append(len(tokeep))
                            # if m_rev[0] < 0:
                            #     sys.exit()
                            
                        else: 
                            pos_slope_rev_ko.append(m_rev[0]*(0.35/24))
                            pos_dist_rev_ko.append(dist.values[0]*(35/24))
                            pos_cell_rev_ko.append(len(tokeep))
                        
                        # plt.figure()
                        # plt.subplot(121)
                        # plt.title('R = ' + str(round(r_rev,2)) + ', p = ' + str(round(p_rev,2)))
                        # plt.scatter(firstspk - pb_reward[i]['start'], rxmax.loc[tokeep], marker = '|', color = colorr[rxmax.index.get_indexer(tokeep)], s = 500)
                        # plt.plot(firstspk - pb_reward[i]['start'], ybin + b_rev, color = 'k')
                        # plt.ylabel('Max X bin')
                        # plt.ylim([-1, 25])
                        # plt.xlabel ('Time (s)')
                        # plt.gca().set_box_aspect(1)
                        # plt.subplot(122)
                        # plt.hist(corrs_shu, bins = 10)                
                        # plt.axvline(r_rev, color = 'k')
                        # plt.gca().set_box_aspect(1)
                       
                                        
                        sig_r.append(1)
                                      
                    
                    elif (r_rev < np.percentile(corrs_shu, 5)) :
                        
                        m_rev, b_rev = np.polyfit(firstspk - pb_reward[i]['start'], rxmax.loc[tokeep], 1)
                        ybin = m_rev*(firstspk - pb_reward[i]['start'])
                        
                        dist = rxmax.loc[tokeep].max() - rxmax.loc[tokeep].min()
                        
                        if isWT == 1:
                            neg_slope_rev_wt.append(m_rev[0]*(0.35/24))
                            neg_dist_rev_wt.append(dist.values[0]*(35/24))
                            neg_cell_rev_wt.append(len(tokeep))
                        else: 
                            neg_slope_rev_ko.append(m_rev[0]*(0.35/24))
                            neg_dist_rev_ko.append(dist.values[0]*(35/24))
                            neg_cell_rev_ko.append(len(tokeep))
                                                
                        # plt.figure()
                        # plt.subplot(121)
                        # plt.title('R = ' + str(round(r_rev,2)) + ', p = ' + str(round(p_rev,2)))
                        # plt.scatter(firstspk - pb_reward[i]['start'], rxmax.loc[tokeep], marker = '|', color = colorr[rxmax.index.get_indexer(tokeep)], s = 500)
                        # plt.plot(firstspk - pb_reward[i]['start'], ybin + b_rev, color = 'k')
                        # plt.ylabel('Max X bin')
                        # plt.ylim([-1, 25])
                        # plt.xlabel ('Time (s)')
                        # plt.gca().set_box_aspect(1)
                        # plt.subplot(122)
                        # plt.hist(corrs_shu, bins = 10)                
                        # plt.axvline(r_rev, color = 'k')
                        # plt.gca().set_box_aspect(1)
                                        
                        sig_r.append(1)
                    else: 
                        sig_r.append(0)
               
        if isWT == 1:
            fwd_corrs_wt.extend(corrs_f)
            fwd_sig_wt.extend(sig_f)
            
            rev_corrs_wt.extend(corrs_r)
            rev_sig_wt.extend(sig_r)
        
            sigfrac_fwd_wt.append((sum(s == 1 and c > 0 for s, c in zip(sig_f, corrs_f))
            + sum(s == 1 and c < 0 for s, c in zip(sig_r, corrs_r))) / (len(corrs_f) + len(corrs_r)))
            
            sigfrac_rev_wt.append((sum(s == 1 and c < 0 for s, c in zip(sig_f, corrs_f))
            + sum(s == 1 and c > 0 for s, c in zip(sig_r, corrs_r))) / (len(corrs_f) + len(corrs_r)))
            
        else:
            fwd_corrs_ko.extend(corrs_f)
            fwd_sig_ko.extend(sig_f)
            
            rev_corrs_ko.extend(corrs_r)
            rev_sig_ko.extend(sig_r)
        
            sigfrac_fwd_ko.append((sum(s == 1 and c > 0 for s, c in zip(sig_f, corrs_f))
            + sum(s == 1 and c < 0 for s, c in zip(sig_r, corrs_r))) / (len(corrs_f) + len(corrs_r)))
            
            sigfrac_rev_ko.append((sum(s == 1 and c < 0 for s, c in zip(sig_f, corrs_f))
            + sum(s == 1 and c > 0 for s, c in zip(sig_r, corrs_r))) / (len(corrs_f) + len(corrs_r)))
               
        
#%% Organize corrs from all sessions (fwd and rev separately)

# label = ['all events', 'significant events']

# plt.figure()
# plt.suptitle('Inbound (towards the centre)')
# plt.subplot(121)
# plt.title('WT')
# a = [c for s, c in zip(fwd_sig_wt, fwd_corrs_wt) if s == 1]
# x_multi = [fwd_corrs_wt, a]
# plt.hist(x_multi, 20, histtype = 'bar', label = label)
# plt.xlabel('R')
# plt.ylabel('Counts')
# plt.legend(loc = 'upper right')
# plt.gca().set_box_aspect(1)
# plt.subplot(122)
# plt.title('KO')
# a = [c for s, c in zip(fwd_sig_ko, fwd_corrs_ko) if s == 1]
# x_multi = [fwd_corrs_ko, a]
# plt.hist(x_multi, 20, histtype = 'bar', label = label)
# plt.xlabel('R')
# plt.ylabel('Counts')
# plt.legend(loc = 'upper right')
# plt.gca().set_box_aspect(1)

# plt.figure()
# plt.suptitle('Outbound (towards the edge)')
# plt.subplot(121)
# plt.title('WT')
# a = [c for s, c in zip(rev_sig_wt, rev_corrs_wt) if s == 1]
# x_multi = [rev_corrs_wt, a]
# plt.hist(x_multi, 20, histtype = 'bar', label = label)
# plt.xlabel('R')
# plt.ylabel('Counts')
# plt.legend(loc = 'upper right')
# plt.gca().set_box_aspect(1)
# plt.subplot(122)
# plt.title('KO')
# a = [c for s, c in zip(rev_sig_ko, rev_corrs_ko) if s == 1]
# x_multi = [rev_corrs_ko, a]
# plt.hist(x_multi, 20, histtype = 'bar', label = label)
# plt.xlabel('R')
# plt.ylabel('Counts')
# plt.legend(loc = 'upper right')
# plt.gca().set_box_aspect(1)
      
        
#%% Organize spatial information data 

# wt1 = np.array(['WT' for x in range(len(allspatialinfo_wt))])
# ko1 = np.array(['KO' for x in range(len(allspatialinfo_ko))])

# genotype = np.hstack([wt1, ko1])

# sinfos1 = []
# sinfos1.extend(allspatialinfo_wt)
# sinfos1.extend(allspatialinfo_ko)

# allinfos1 = pd.DataFrame(data = [sinfos1, genotype], index = ['SI', 'genotype']).T

#%% Plot spatial info 

# plt.figure()
# plt.suptitle('Spatial Information')
# plt.title('Linear Track')
# sns.set_style('white')
# palette = ['royalblue', 'indianred'] 
# ax = sns.violinplot( x = allinfos1['genotype'], y=allinfos1['SI'].astype(float) , data = allinfos1, dodge=False,
#                     palette = palette,cut = 2,
#                     scale="width", inner=None)
# ax.tick_params(bottom=True, left=True)
# xlim = ax.get_xlim()
# ylim = ax.get_ylim()
# for violin in ax.collections:
#     x0, y0, width, height = violin.get_paths()[0].get_extents().bounds
#     violin.set_clip_path(plt.Rectangle((x0, y0), width / 2, height, transform=ax.transData))
# sns.boxplot(x = allinfos1['genotype'], y=allinfos1['SI'].astype(float) , data = allinfos1, saturation=1, showfliers=False,
#             width=0.3, boxprops={'zorder': 3, 'facecolor': 'none'}, ax=ax)
# old_len_collections = len(ax.collections)
# sns.stripplot(x = allinfos1['genotype'], y = allinfos1['SI'].astype(float), data = allinfos1, color = 'k', dodge=False, ax=ax, alpha = 0.2)
# for dots in ax.collections[old_len_collections:]:
#     dots.set_offsets(dots.get_offsets())
# ax.set_xlim(xlim)
# ax.set_ylim(ylim)
# plt.ylabel('Spatial Information (bits per spike)')
# ax.set_box_aspect(1)
    
#%% Stats for SI

# t_SI, p_SI = mannwhitneyu(allspatialinfo_wt, allspatialinfo_ko)

#%% Organize replay slope data

wt1 = np.array(['WT' for x in range(len(pos_slope_fwd_wt))])
ko1 = np.array(['KO' for x in range(len(pos_slope_fwd_ko))])
wt2 = np.array(['WT' for x in range(len(neg_slope_fwd_wt))])
ko2 = np.array(['KO' for x in range(len(neg_slope_fwd_ko))])
wt3 = np.array(['WT' for x in range(len(pos_slope_rev_wt))])
ko3 = np.array(['KO' for x in range(len(pos_slope_rev_ko))])
wt4 = np.array(['WT' for x in range(len(neg_slope_rev_wt))])
ko4 = np.array(['KO' for x in range(len(neg_slope_rev_ko))])
genotype = np.hstack([wt1, ko1, wt2, ko2, wt3, ko3, wt4, ko4])

p1 = np.array(['pos' for x in range(len(pos_slope_fwd_wt))])
p2 = np.array(['pos' for x in range(len(pos_slope_fwd_ko))])
n1 = np.array(['neg' for x in range(len(neg_slope_fwd_wt))])
n2 = np.array(['neg' for x in range(len(neg_slope_fwd_ko))])
p3 = np.array(['pos' for x in range(len(pos_slope_rev_wt))])
p4 = np.array(['pos' for x in range(len(pos_slope_rev_ko))])
n3 = np.array(['neg' for x in range(len(neg_slope_rev_wt))])
n4 = np.array(['neg' for x in range(len(neg_slope_rev_ko))])
slopesign = np.hstack([p1, p2, n1, n2, p3, p4, n3, n4])

f1 = np.array(['fwd' for x in range(len(pos_slope_fwd_wt))])
f2 = np.array(['fwd' for x in range(len(pos_slope_fwd_ko))])
f3 = np.array(['fwd' for x in range(len(neg_slope_fwd_wt))])
f4 = np.array(['fwd' for x in range(len(neg_slope_fwd_ko))])
r1 = np.array(['rev' for x in range(len(pos_slope_rev_wt))])
r2 = np.array(['rev' for x in range(len(pos_slope_rev_ko))])
r3 = np.array(['rev' for x in range(len(neg_slope_rev_wt))])
r4 = np.array(['rev' for x in range(len(neg_slope_rev_ko))])
direction = np.hstack([f1, f2, f3, f4, r1, r2, r3, r4])

sinfos1 = []
sinfos1.extend(pos_slope_fwd_wt)
sinfos1.extend(pos_slope_fwd_ko)
sinfos1.extend(neg_slope_fwd_wt)
sinfos1.extend(neg_slope_fwd_ko)
sinfos1.extend(pos_slope_rev_wt)
sinfos1.extend(pos_slope_rev_ko)
sinfos1.extend(neg_slope_rev_wt)
sinfos1.extend(neg_slope_rev_ko)

allinfos1 = pd.DataFrame(data = [sinfos1, slopesign, direction, genotype], index = ['slope', 'sign', 'direction', 'genotype']).T

t_slope_f, p_slope_f = mannwhitneyu(allinfos1['slope'][(allinfos1['sign'] == 'pos') & (allinfos1['genotype'] == 'WT')].values.astype(float)
                                    , allinfos1['slope'][(allinfos1['sign'] == 'pos') & (allinfos1['genotype'] == 'KO')].values.astype(float)) 

t_slope_r, p_slope_r = mannwhitneyu(allinfos1['slope'][(allinfos1['sign'] == 'neg') & (allinfos1['genotype'] == 'WT')].values.astype(float)
                                    , allinfos1['slope'][(allinfos1['sign'] == 'neg') & (allinfos1['genotype'] == 'KO')].values.astype(float)) 

print(len(allinfos1['slope'][(allinfos1['sign'] == 'pos') & (allinfos1['genotype'] == 'WT')]), len(allinfos1['slope'][(allinfos1['sign'] == 'pos') & (allinfos1['genotype'] == 'KO')]))
print(len(allinfos1['slope'][(allinfos1['sign'] == 'neg') & (allinfos1['genotype'] == 'WT')]), len(allinfos1['slope'][(allinfos1['sign'] == 'neg') & (allinfos1['genotype'] == 'KO')]))

#%% Plot replay slope 

plt.figure()
plt.suptitle('Replay Speed')
sns.set_style('white')
palette = ['royalblue', 'indianred'] 
plt.subplot(121)
plt.title('Forward Replay')
ax = sns.violinplot( x = allinfos1['genotype'], y=allinfos1['slope'][(allinfos1['sign'] == 'pos')].astype(float), 
                    data = allinfos1, dodge=False,
                    palette = palette,cut = 2,
                    scale="width", inner=None)
ax.tick_params(bottom=True, left=True)
xlim = ax.get_xlim()
ylim = ax.get_ylim()
for violin in ax.collections:
    x0, y0, width, height = violin.get_paths()[0].get_extents().bounds
    violin.set_clip_path(plt.Rectangle((x0, y0), width / 2, height, transform=ax.transData))
sns.boxplot(x = allinfos1['genotype'], y=allinfos1['slope'][(allinfos1['sign'] == 'pos')].astype(float), 
            data = allinfos1, saturation=1, showfliers=False,
            width=0.3, boxprops={'zorder': 3, 'facecolor': 'none'}, ax=ax)
old_len_collections = len(ax.collections)
sns.stripplot(x = allinfos1['genotype'], y = allinfos1['slope'][(allinfos1['sign'] == 'pos')].astype(float), 
              data = allinfos1, color = 'k', dodge=False, ax=ax, alpha = 0.2)
for dots in ax.collections[old_len_collections:]:
    dots.set_offsets(dots.get_offsets())
ax.set_xlim(xlim)
ax.set_ylim(ylim)
plt.ylabel('Replay speed (m/s)')
ax.set_box_aspect(1)

plt.subplot(122)
plt.title('Reverse Replay')
ax = sns.violinplot( x = allinfos1['genotype'], y=allinfos1['slope'][(allinfos1['sign'] == 'neg')].astype(float), 
                    data = allinfos1, dodge=False,
                    palette = palette,cut = 2,
                    scale="width", inner=None)
ax.tick_params(bottom=True, left=True)
xlim = ax.get_xlim()
ylim = ax.get_ylim()
for violin in ax.collections:
    x0, y0, width, height = violin.get_paths()[0].get_extents().bounds
    violin.set_clip_path(plt.Rectangle((x0, y0), width / 2, height, transform=ax.transData))
sns.boxplot(x = allinfos1['genotype'], y=allinfos1['slope'][(allinfos1['sign'] == 'neg')].astype(float), 
            data = allinfos1, saturation=1, showfliers=False,
            width=0.3, boxprops={'zorder': 3, 'facecolor': 'none'}, ax=ax)
old_len_collections = len(ax.collections)
sns.stripplot(x = allinfos1['genotype'], y = allinfos1['slope'][(allinfos1['sign'] == 'neg')].astype(float), 
              data = allinfos1, color = 'k', dodge=False, ax=ax, alpha = 0.2)
for dots in ax.collections[old_len_collections:]:
    dots.set_offsets(dots.get_offsets())
ax.set_xlim(xlim)
ax.set_ylim(ylim)
plt.ylabel('Replay speed (m/s)')
ax.set_box_aspect(1)

###Outbound 

# plt.figure()
# plt.suptitle('Replay Slope - Outbound')
# sns.set_style('white')
# palette = ['royalblue', 'indianred'] 
# plt.subplot(121)
# plt.title('Forward Replay')
# ax = sns.violinplot( x = allinfos1['genotype'], y=allinfos1['slope'][(allinfos1['direction'] == 'rev') & (allinfos1['sign'] == 'pos')].astype(float), 
#                     data = allinfos1, dodge=False,
#                     palette = palette,cut = 2,
#                     scale="width", inner=None)
# ax.tick_params(bottom=True, left=True)
# xlim = ax.get_xlim()
# ylim = ax.get_ylim()
# for violin in ax.collections:
#     x0, y0, width, height = violin.get_paths()[0].get_extents().bounds
#     violin.set_clip_path(plt.Rectangle((x0, y0), width / 2, height, transform=ax.transData))
# sns.boxplot(x = allinfos1['genotype'], y=allinfos1['slope'][(allinfos1['direction'] == 'rev') & (allinfos1['sign'] == 'pos')].astype(float), 
#             data = allinfos1, saturation=1, showfliers=False,
#             width=0.3, boxprops={'zorder': 3, 'facecolor': 'none'}, ax=ax)
# old_len_collections = len(ax.collections)
# sns.stripplot(x = allinfos1['genotype'], y = allinfos1['slope'][(allinfos1['direction'] == 'rev') & (allinfos1['sign'] == 'pos')].astype(float), 
#               data = allinfos1, color = 'k', dodge=False, ax=ax, alpha = 0.2)
# for dots in ax.collections[old_len_collections:]:
#     dots.set_offsets(dots.get_offsets())
# ax.set_xlim(xlim)
# ax.set_ylim(ylim)
# plt.ylabel('Replay slope')
# ax.set_box_aspect(1)

# plt.subplot(122)
# plt.title('Reverse Replay')
# ax = sns.violinplot( x = allinfos1['genotype'], y=allinfos1['slope'][(allinfos1['direction'] == 'rev') & (allinfos1['sign'] == 'neg')].astype(float), 
#                     data = allinfos1, dodge=False,
#                     palette = palette,cut = 2,
#                     scale="width", inner=None)
# ax.tick_params(bottom=True, left=True)
# xlim = ax.get_xlim()
# ylim = ax.get_ylim()
# for violin in ax.collections:
#     x0, y0, width, height = violin.get_paths()[0].get_extents().bounds
#     violin.set_clip_path(plt.Rectangle((x0, y0), width / 2, height, transform=ax.transData))
# sns.boxplot(x = allinfos1['genotype'], y=allinfos1['slope'][(allinfos1['direction'] == 'rev') & (allinfos1['sign'] == 'neg')].astype(float), 
#             data = allinfos1, saturation=1, showfliers=False,
#             width=0.3, boxprops={'zorder': 3, 'facecolor': 'none'}, ax=ax)
# old_len_collections = len(ax.collections)
# sns.stripplot(x = allinfos1['genotype'], y = allinfos1['slope'][(allinfos1['direction'] == 'rev') & (allinfos1['sign'] == 'neg')].astype(float), 
#               data = allinfos1, color = 'k', dodge=False, ax=ax, alpha = 0.2)
# for dots in ax.collections[old_len_collections:]:
#     dots.set_offsets(dots.get_offsets())
# ax.set_xlim(xlim)
# ax.set_ylim(ylim)
# plt.ylabel('Replay slope')
# ax.set_box_aspect(1)

#%% Organize replay distance data

wt1 = np.array(['WT' for x in range(len(pos_dist_fwd_wt))])
ko1 = np.array(['KO' for x in range(len(pos_dist_fwd_ko))])
wt2 = np.array(['WT' for x in range(len(neg_dist_fwd_wt))])
ko2 = np.array(['KO' for x in range(len(neg_dist_fwd_ko))])
wt3 = np.array(['WT' for x in range(len(pos_dist_rev_wt))])
ko3 = np.array(['KO' for x in range(len(pos_dist_rev_ko))])
wt4 = np.array(['WT' for x in range(len(neg_dist_rev_wt))])
ko4 = np.array(['KO' for x in range(len(neg_dist_rev_ko))])
genotype = np.hstack([wt1, ko1, wt2, ko2, wt3, ko3, wt4, ko4])

p1 = np.array(['pos' for x in range(len(pos_dist_fwd_wt))])
p2 = np.array(['pos' for x in range(len(pos_dist_fwd_ko))])
n1 = np.array(['neg' for x in range(len(neg_dist_fwd_wt))])
n2 = np.array(['neg' for x in range(len(neg_dist_fwd_ko))])
p3 = np.array(['pos' for x in range(len(pos_dist_rev_wt))])
p4 = np.array(['pos' for x in range(len(pos_dist_rev_ko))])
n3 = np.array(['neg' for x in range(len(neg_dist_rev_wt))])
n4 = np.array(['neg' for x in range(len(neg_dist_rev_ko))])
slopesign = np.hstack([p1, p2, n1, n2, p3, p4, n3, n4])

f1 = np.array(['fwd' for x in range(len(pos_dist_fwd_wt))])
f2 = np.array(['fwd' for x in range(len(pos_dist_fwd_ko))])
f3 = np.array(['fwd' for x in range(len(neg_dist_fwd_wt))])
f4 = np.array(['fwd' for x in range(len(neg_dist_fwd_ko))])
r1 = np.array(['rev' for x in range(len(pos_dist_rev_wt))])
r2 = np.array(['rev' for x in range(len(pos_dist_rev_ko))])
r3 = np.array(['rev' for x in range(len(neg_dist_rev_wt))])
r4 = np.array(['rev' for x in range(len(neg_dist_rev_ko))])
direction = np.hstack([f1, f2, f3, f4, r1, r2, r3, r4])

sinfos1 = []
sinfos1.extend(pos_dist_fwd_wt)
sinfos1.extend(pos_dist_fwd_ko)
sinfos1.extend(neg_dist_fwd_wt)
sinfos1.extend(neg_dist_fwd_ko)
sinfos1.extend(pos_dist_rev_wt)
sinfos1.extend(pos_dist_rev_ko)
sinfos1.extend(neg_dist_rev_wt)
sinfos1.extend(neg_dist_rev_ko)

allinfos1 = pd.DataFrame(data = [sinfos1, slopesign, direction, genotype], index = ['dist', 'sign', 'direction', 'genotype']).T

t_dist_f, p_dist_f = mannwhitneyu(allinfos1['dist'][(allinfos1['sign'] == 'pos') & (allinfos1['genotype'] == 'WT')].values.astype(float)
                                    , allinfos1['dist'][(allinfos1['sign'] == 'pos') & (allinfos1['genotype'] == 'KO')].values.astype(float)) 

t_dist_r, p_dist_r = mannwhitneyu(allinfos1['dist'][(allinfos1['sign'] == 'neg') & (allinfos1['genotype'] == 'WT')].values.astype(float)
                                    , allinfos1['dist'][(allinfos1['sign'] == 'neg') & (allinfos1['genotype'] == 'KO')].values.astype(float)) 



#%% Plot replay distance 

plt.figure()
plt.suptitle('Replay Distance')
sns.set_style('white')
palette = ['royalblue', 'indianred'] 
plt.subplot(121)
plt.title('Forward Replay')
ax = sns.violinplot( x = allinfos1['genotype'], y=allinfos1['dist'][(allinfos1['sign'] == 'pos')].astype(float), 
                    data = allinfos1, dodge=False,
                    palette = palette,cut = 2,
                    scale="width", inner=None)
ax.tick_params(bottom=True, left=True)
xlim = ax.get_xlim()
ylim = ax.get_ylim()
for violin in ax.collections:
    x0, y0, width, height = violin.get_paths()[0].get_extents().bounds
    violin.set_clip_path(plt.Rectangle((x0, y0), width / 2, height, transform=ax.transData))
sns.boxplot(x = allinfos1['genotype'], y=allinfos1['dist'][(allinfos1['sign'] == 'pos')].astype(float), 
            data = allinfos1, saturation=1, showfliers=False,
            width=0.3, boxprops={'zorder': 3, 'facecolor': 'none'}, ax=ax)
old_len_collections = len(ax.collections)
sns.stripplot(x = allinfos1['genotype'], y = allinfos1['dist'][(allinfos1['sign'] == 'pos')].astype(float), 
              data = allinfos1, color = 'k', dodge=False, ax=ax, alpha = 0.2)
for dots in ax.collections[old_len_collections:]:
    dots.set_offsets(dots.get_offsets())
ax.set_xlim(xlim)
ax.set_ylim(ylim)
plt.ylabel('Replay distance (cm)')
ax.set_box_aspect(1)

plt.subplot(122)
plt.title('Reverse Replay')
ax = sns.violinplot( x = allinfos1['genotype'], y=allinfos1['dist'][(allinfos1['sign'] == 'neg')].astype(float), 
                    data = allinfos1, dodge=False,
                    palette = palette,cut = 2,
                    scale="width", inner=None)
ax.tick_params(bottom=True, left=True)
xlim = ax.get_xlim()
ylim = ax.get_ylim()
for violin in ax.collections:
    x0, y0, width, height = violin.get_paths()[0].get_extents().bounds
    violin.set_clip_path(plt.Rectangle((x0, y0), width / 2, height, transform=ax.transData))
sns.boxplot(x = allinfos1['genotype'], y=allinfos1['dist'][(allinfos1['sign'] == 'neg')].astype(float), 
            data = allinfos1, saturation=1, showfliers=False,
            width=0.3, boxprops={'zorder': 3, 'facecolor': 'none'}, ax=ax)
old_len_collections = len(ax.collections)
sns.stripplot(x = allinfos1['genotype'], y = allinfos1['dist'][(allinfos1['direction'] == 'fwd') & (allinfos1['sign'] == 'neg')].astype(float), 
              data = allinfos1, color = 'k', dodge=False, ax=ax, alpha = 0.2)
for dots in ax.collections[old_len_collections:]:
    dots.set_offsets(dots.get_offsets())
ax.set_xlim(xlim)
ax.set_ylim(ylim)
plt.ylabel('Replay distance (cm)')
ax.set_box_aspect(1)

###Outbound 

# plt.figure()
# plt.suptitle('Replay Distance - Outbound')
# sns.set_style('white')
# palette = ['royalblue', 'indianred'] 
# plt.subplot(121)
# plt.title('Forward Replay')
# ax = sns.violinplot( x = allinfos1['genotype'], y=allinfos1['dist'][(allinfos1['direction'] == 'rev') & (allinfos1['sign'] == 'pos')].astype(float), 
#                     data = allinfos1, dodge=False,
#                     palette = palette,cut = 2,
#                     scale="width", inner=None)
# ax.tick_params(bottom=True, left=True)
# xlim = ax.get_xlim()
# ylim = ax.get_ylim()
# for violin in ax.collections:
#     x0, y0, width, height = violin.get_paths()[0].get_extents().bounds
#     violin.set_clip_path(plt.Rectangle((x0, y0), width / 2, height, transform=ax.transData))
# sns.boxplot(x = allinfos1['genotype'], y=allinfos1['dist'][(allinfos1['direction'] == 'rev') & (allinfos1['sign'] == 'pos')].astype(float), 
#             data = allinfos1, saturation=1, showfliers=False,
#             width=0.3, boxprops={'zorder': 3, 'facecolor': 'none'}, ax=ax)
# old_len_collections = len(ax.collections)
# sns.stripplot(x = allinfos1['genotype'], y = allinfos1['dist'][(allinfos1['direction'] == 'rev') & (allinfos1['sign'] == 'pos')].astype(float), 
#               data = allinfos1, color = 'k', dodge=False, ax=ax, alpha = 0.2)
# for dots in ax.collections[old_len_collections:]:
#     dots.set_offsets(dots.get_offsets())
# ax.set_xlim(xlim)
# ax.set_ylim(ylim)
# plt.ylabel('Replay distance (bins)')
# ax.set_box_aspect(1)

# plt.subplot(122)
# plt.title('Reverse Replay')
# ax = sns.violinplot( x = allinfos1['genotype'], y=allinfos1['dist'][(allinfos1['direction'] == 'rev') & (allinfos1['sign'] == 'neg')].astype(float), 
#                     data = allinfos1, dodge=False,
#                     palette = palette,cut = 2,
#                     scale="width", inner=None)
# ax.tick_params(bottom=True, left=True)
# xlim = ax.get_xlim()
# ylim = ax.get_ylim()
# for violin in ax.collections:
#     x0, y0, width, height = violin.get_paths()[0].get_extents().bounds
#     violin.set_clip_path(plt.Rectangle((x0, y0), width / 2, height, transform=ax.transData))
# sns.boxplot(x = allinfos1['genotype'], y=allinfos1['dist'][(allinfos1['direction'] == 'rev') & (allinfos1['sign'] == 'neg')].astype(float), 
#             data = allinfos1, saturation=1, showfliers=False,
#             width=0.3, boxprops={'zorder': 3, 'facecolor': 'none'}, ax=ax)
# old_len_collections = len(ax.collections)
# sns.stripplot(x = allinfos1['genotype'], y = allinfos1['dist'][(allinfos1['direction'] == 'rev') & (allinfos1['sign'] == 'neg')].astype(float), 
#               data = allinfos1, color = 'k', dodge=False, ax=ax, alpha = 0.2)
# for dots in ax.collections[old_len_collections:]:
#     dots.set_offsets(dots.get_offsets())
# ax.set_xlim(xlim)
# ax.set_ylim(ylim)
# plt.ylabel('Replay distance (bins)')
# ax.set_box_aspect(1)

#%% Organize nCells data

wt1 = np.array(['WT' for x in range(len(pos_cell_fwd_wt))])
ko1 = np.array(['KO' for x in range(len(pos_cell_fwd_ko))])
wt2 = np.array(['WT' for x in range(len(neg_cell_fwd_wt))])
ko2 = np.array(['KO' for x in range(len(neg_cell_fwd_ko))])
wt3 = np.array(['WT' for x in range(len(pos_cell_rev_wt))])
ko3 = np.array(['KO' for x in range(len(pos_cell_rev_ko))])
wt4 = np.array(['WT' for x in range(len(neg_cell_rev_wt))])
ko4 = np.array(['KO' for x in range(len(neg_cell_rev_ko))])
genotype = np.hstack([wt1, ko1, wt2, ko2, wt3, ko3, wt4, ko4])

p1 = np.array(['pos' for x in range(len(pos_cell_fwd_wt))])
p2 = np.array(['pos' for x in range(len(pos_cell_fwd_ko))])
n1 = np.array(['neg' for x in range(len(neg_cell_fwd_wt))])
n2 = np.array(['neg' for x in range(len(neg_cell_fwd_ko))])
p3 = np.array(['pos' for x in range(len(pos_cell_rev_wt))])
p4 = np.array(['pos' for x in range(len(pos_cell_rev_ko))])
n3 = np.array(['neg' for x in range(len(neg_cell_rev_wt))])
n4 = np.array(['neg' for x in range(len(neg_cell_rev_ko))])
slopesign = np.hstack([p1, p2, n1, n2, p3, p4, n3, n4])

f1 = np.array(['fwd' for x in range(len(pos_cell_fwd_wt))])
f2 = np.array(['fwd' for x in range(len(pos_cell_fwd_ko))])
f3 = np.array(['fwd' for x in range(len(neg_cell_fwd_wt))])
f4 = np.array(['fwd' for x in range(len(neg_cell_fwd_ko))])
r1 = np.array(['rev' for x in range(len(pos_cell_rev_wt))])
r2 = np.array(['rev' for x in range(len(pos_cell_rev_ko))])
r3 = np.array(['rev' for x in range(len(neg_cell_rev_wt))])
r4 = np.array(['rev' for x in range(len(neg_cell_rev_ko))])
direction = np.hstack([f1, f2, f3, f4, r1, r2, r3, r4])

sinfos1 = []
sinfos1.extend(pos_cell_fwd_wt)
sinfos1.extend(pos_cell_fwd_ko)
sinfos1.extend(neg_cell_fwd_wt)
sinfos1.extend(neg_cell_fwd_ko)
sinfos1.extend(pos_cell_rev_wt)
sinfos1.extend(pos_cell_rev_ko)
sinfos1.extend(neg_cell_rev_wt)
sinfos1.extend(neg_cell_rev_ko)

allinfos1 = pd.DataFrame(data = [sinfos1, slopesign, direction, genotype], index = ['cell', 'sign', 'direction', 'genotype']).T

t_cell_f, p_cell_f = mannwhitneyu(allinfos1['cell'][(allinfos1['sign'] == 'pos') & (allinfos1['genotype'] == 'WT')].values.astype(float)
                                    , allinfos1['cell'][(allinfos1['sign'] == 'pos') & (allinfos1['genotype'] == 'KO')].values.astype(float)) 

t_cell_r, p_cell_r = mannwhitneyu(allinfos1['cell'][(allinfos1['sign'] == 'neg') & (allinfos1['genotype'] == 'WT')].values.astype(float)
                                    , allinfos1['cell'][(allinfos1['sign'] == 'neg') & (allinfos1['genotype'] == 'KO')].values.astype(float)) 


#%% Plot nCells

plt.figure()
plt.suptitle('Replay nCells')
sns.set_style('white')
palette = ['royalblue', 'indianred'] 
plt.subplot(121)
plt.title('Forward Replay')
ax = sns.violinplot( x = allinfos1['genotype'], y=allinfos1['cell'][(allinfos1['sign'] == 'pos')].astype(float), 
                    data = allinfos1, dodge=False,
                    palette = palette,cut = 2,
                    scale="width", inner=None)
ax.tick_params(bottom=True, left=True)
xlim = ax.get_xlim()
ylim = ax.get_ylim()
for violin in ax.collections:
    x0, y0, width, height = violin.get_paths()[0].get_extents().bounds
    violin.set_clip_path(plt.Rectangle((x0, y0), width / 2, height, transform=ax.transData))
sns.boxplot(x = allinfos1['genotype'], y=allinfos1['cell'][(allinfos1['sign'] == 'pos')].astype(float), 
            data = allinfos1, saturation=1, showfliers=False,
            width=0.3, boxprops={'zorder': 3, 'facecolor': 'none'}, ax=ax)
old_len_collections = len(ax.collections)
sns.stripplot(x = allinfos1['genotype'], y = allinfos1['cell'][(allinfos1['sign'] == 'pos')].astype(float), 
              data = allinfos1, color = 'k', dodge=False, ax=ax, alpha = 0.2)
for dots in ax.collections[old_len_collections:]:
    dots.set_offsets(dots.get_offsets())
ax.set_xlim(xlim)
ax.set_ylim(ylim)
plt.ylabel('#cells per event')
ax.set_box_aspect(1)

plt.subplot(122)
plt.title('Reverse Replay')
ax = sns.violinplot( x = allinfos1['genotype'], y=allinfos1['cell'][(allinfos1['sign'] == 'neg')].astype(float), 
                    data = allinfos1, dodge=False,
                    palette = palette,cut = 2,
                    scale="width", inner=None)
ax.tick_params(bottom=True, left=True)
xlim = ax.get_xlim()
ylim = ax.get_ylim()
for violin in ax.collections:
    x0, y0, width, height = violin.get_paths()[0].get_extents().bounds
    violin.set_clip_path(plt.Rectangle((x0, y0), width / 2, height, transform=ax.transData))
sns.boxplot(x = allinfos1['genotype'], y=allinfos1['cell'][(allinfos1['sign'] == 'neg')].astype(float), 
            data = allinfos1, saturation=1, showfliers=False,
            width=0.3, boxprops={'zorder': 3, 'facecolor': 'none'}, ax=ax)
old_len_collections = len(ax.collections)
sns.stripplot(x = allinfos1['genotype'], y = allinfos1['cell'][(allinfos1['sign'] == 'neg')].astype(float), 
              data = allinfos1, color = 'k', dodge=False, ax=ax, alpha = 0.2)
for dots in ax.collections[old_len_collections:]:
    dots.set_offsets(dots.get_offsets())
ax.set_xlim(xlim)
ax.set_ylim(ylim)
plt.ylabel('#cells per event')
ax.set_box_aspect(1)

###Outbound 

# plt.figure()
# plt.suptitle('Replay nCells - Outbound')
# sns.set_style('white')
# palette = ['royalblue', 'indianred'] 
# plt.subplot(121)
# plt.title('Forward Replay')
# ax = sns.violinplot( x = allinfos1['genotype'], y=allinfos1['cell'][(allinfos1['direction'] == 'rev') & (allinfos1['sign'] == 'pos')].astype(float), 
#                     data = allinfos1, dodge=False,
#                     palette = palette,cut = 2,
#                     scale="width", inner=None)
# ax.tick_params(bottom=True, left=True)
# xlim = ax.get_xlim()
# ylim = ax.get_ylim()
# for violin in ax.collections:
#     x0, y0, width, height = violin.get_paths()[0].get_extents().bounds
#     violin.set_clip_path(plt.Rectangle((x0, y0), width / 2, height, transform=ax.transData))
# sns.boxplot(x = allinfos1['genotype'], y=allinfos1['cell'][(allinfos1['direction'] == 'rev') & (allinfos1['sign'] == 'pos')].astype(float), 
#             data = allinfos1, saturation=1, showfliers=False,
#             width=0.3, boxprops={'zorder': 3, 'facecolor': 'none'}, ax=ax)
# old_len_collections = len(ax.collections)
# sns.stripplot(x = allinfos1['genotype'], y = allinfos1['cell'][(allinfos1['direction'] == 'rev') & (allinfos1['sign'] == 'pos')].astype(float), 
#               data = allinfos1, color = 'k', dodge=False, ax=ax, alpha = 0.2)
# for dots in ax.collections[old_len_collections:]:
#     dots.set_offsets(dots.get_offsets())
# ax.set_xlim(xlim)
# ax.set_ylim(ylim)
# plt.ylabel('#cells per event')
# ax.set_box_aspect(1)

# plt.subplot(122)
# plt.title('Reverse Replay')
# ax = sns.violinplot( x = allinfos1['genotype'], y=allinfos1['cell'][(allinfos1['direction'] == 'rev') & (allinfos1['sign'] == 'neg')].astype(float), 
#                     data = allinfos1, dodge=False,
#                     palette = palette,cut = 2,
#                     scale="width", inner=None)
# ax.tick_params(bottom=True, left=True)
# xlim = ax.get_xlim()
# ylim = ax.get_ylim()
# for violin in ax.collections:
#     x0, y0, width, height = violin.get_paths()[0].get_extents().bounds
#     violin.set_clip_path(plt.Rectangle((x0, y0), width / 2, height, transform=ax.transData))
# sns.boxplot(x = allinfos1['genotype'], y=allinfos1['cell'][(allinfos1['direction'] == 'rev') & (allinfos1['sign'] == 'neg')].astype(float), 
#             data = allinfos1, saturation=1, showfliers=False,
#             width=0.3, boxprops={'zorder': 3, 'facecolor': 'none'}, ax=ax)
# old_len_collections = len(ax.collections)
# sns.stripplot(x = allinfos1['genotype'], y = allinfos1['cell'][(allinfos1['direction'] == 'rev') & (allinfos1['sign'] == 'neg')].astype(float), 
#               data = allinfos1, color = 'k', dodge=False, ax=ax, alpha = 0.2)
# for dots in ax.collections[old_len_collections:]:
#     dots.set_offsets(dots.get_offsets())
# ax.set_xlim(xlim)
# ax.set_ylim(ylim)
# plt.ylabel('#cells per event')
# ax.set_box_aspect(1)