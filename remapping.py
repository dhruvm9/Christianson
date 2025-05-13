#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 15:17:52 2023

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
from scipy.stats import mannwhitneyu, wilcoxon, ks_2samp
from functions_DM import *
    
#%% 

warnings.filterwarnings("ignore")

data_directory = '/media/dhruv/Expansion/Processed'
# data_directory = '/media/dhruv/Expansion/Processed/CA3'
# data_directory = '/media/adrien/Expansion/Processed'
datasets = np.genfromtxt(os.path.join(data_directory,'remapping_DM.list'), delimiter = '\n', dtype = str, comments = '#')
# datasets = np.genfromtxt(os.path.join(data_directory,'remapping_2sq.list'), delimiter = '\n', dtype = str, comments = '#')
# datasets = np.genfromtxt(os.path.join(data_directory,'remapping_CA3.list'), delimiter = '\n', dtype = str, comments = '#')

env_stability_wt = []
env_stability_ko = []

halfsession1_corr_wt = []
halfsession1_corr_ko = []

halfsession2_corr_wt = []
halfsession2_corr_ko = []

SI1_wt = []
SI1_ko = []

SI2_wt = []
SI2_ko = []

SI3_wt = []
SI3_ko = []

SI4_wt = []
SI4_ko = []

allspatialinfo_env1_wt = []
allspatialinfo_env2_wt = []

allspatialinfo_env1_ko = []
allspatialinfo_env2_ko = []

sparsity1_wt = []
sparsity2_wt = []
sparsity1_ko = []
sparsity2_ko = []

pvcorr_wt = []
pvcorr_ko = []

pvec1_wt = []
pvec2_wt = []
pvec1_ko = []
pvec2_ko = []

pvsess_wt = []
pvsess_ko = []

sess_wt = []
sess_ko = []

oddevencorr_wt = []
oddevencorr_ko = []

npyr2_wt = []
npyr2_ko = []

npyr3_wt = []
npyr3_ko = []

allcells = []
pyrcells = []
fscells = []

for s in datasets:
    print(s)
    name = s.split('-')[0]
       
    path = os.path.join(data_directory, s)
    
    if name == 'B2618' or name == 'B2627' or name == 'B2628' or name == 'B3805' or name == 'B3813':
        isWT = 0
    else: isWT = 1 

    sp2 = np.load(os.path.join(path, 'spikedata_0.55.npz'), allow_pickle = True)
    time_support = nap.IntervalSet(sp2['start'], sp2['end'])
    tsd = nap.Tsd(t=sp2['t'], d=sp2['index'], time_support = time_support)
    spikes = tsd.to_tsgroup()
    spikes.set_info(group = sp2['group'], location = sp2['location'], celltype = sp2['celltype'], tr2pk = sp2['tr2pk'])
    
    allcells.append(len(spikes))
    
    data = ntm.load_session(path, 'neurosuite')
    epochs = data.epochs
    position = data.position
    
#%% Rotate position 

    rot_pos = []
        
    xypos = np.array(position[['x', 'z']])
    
    if name == 'B2625' or name == 'B3800':
        rad = 0.6
    elif name == 'B2618' :
        rad = 0.95
    elif s == 'B2627-240528' or s == 'B2627-240530' or name == 'B3804' or name == 'B3807' or name == 'B3813':
        rad = 0.05
    elif name == 'B3811':
        rad = 0.1
    else: rad = 0    
        
    
    for i in range(len(xypos)):
        newx, newy = rotate_via_numpy(xypos[i], rad)
        rot_pos.append((newx, newy))
        
    rot_pos = nap.TsdFrame(t = position.index.values, d = rot_pos, columns = ['x', 'z'])
    
    w1 = nap.IntervalSet(start = epochs['wake'][0]['start'], end = epochs['wake'][0]['end'])
    w2 = nap.IntervalSet(start = epochs['wake'][1]['start'], end = epochs['wake'][1]['end'])
 

#%% Plot tracking 

    # if isWT == 1:   
    # plt.figure()
    # plt.suptitle(s)
    # plt.subplot(121)
    # plt.plot(rot_pos['x'].restrict(w1), rot_pos['z'].restrict(w1))
    # plt.subplot(122)
    # plt.plot(rot_pos['x'].restrict(w2), rot_pos['z'].restrict(w2))
    
    
# sys.exit()
    
#%% Get cells with wake rate more than 0.5Hz
        
    spikes_by_celltype = spikes.getby_category('celltype')
    
    if 'pyr' in spikes._metadata['celltype'].values:
        pyr = spikes_by_celltype['pyr']
        pyrcells.append(len(pyr))
        
    if 'fs' in spikes._metadata['celltype'].values:
        fs = spikes_by_celltype['fs']
        fscells.append(len(fs))
        
    keep = []
    
    for i in pyr.index:
        if pyr.restrict(nap.IntervalSet(epochs['wake'][0]))._metadata['rate'][i] > 0.5:
            keep.append(i)

    pyr2 = pyr[keep]
    
#%% Compute speed during wake 

    if len(pyr2) > 2:
        
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
        
        ep1 = moving_ep.intersect(epochs['wake'].loc[[0]])
        ep2 = moving_ep.intersect(epochs['wake'].loc[[1]])
              
           
#%% Select out place fields based on epoch 1 
    

### SI criteria from shuffle 

        # placefields1, binsxy1 = nap.compute_2d_tuning_curves(group = pyr2, 
        #                                                     features = rot_pos[['x', 'z']], 
        #                                                     ep = ep1, 
        #                                                     nb_bins=24)  
               
        # px1 = occupancy_prob(rot_pos, ep1, nb_bins=24, norm = True)
        # SI_1 = nap.compute_2d_mutual_info(placefields1, rot_pos[['x', 'z']], ep = ep1)
        
        # keep = []
        
        # for i in pyr2.index:
        #     if SI_1['SI'][i] > 1.25 :
        #         keep.append(i)
                
        # pyr3 = pyr2[keep]
        
### Odd-even epoch correlation
        
        oddep = ep1[1::2]
        evenep = ep1[::2]
        
        pf1, binsxy = nap.compute_2d_tuning_curves(group = pyr2, features = rot_pos[['x', 'z']], ep = oddep, nb_bins=24)  
        pxx1 = occupancy_prob(rot_pos, oddep, nb_bins=24, norm = True)
        SI1 = nap.compute_2d_mutual_info(pf1, rot_pos[['x', 'z']], ep = oddep)
        
        pf2, binsxy = nap.compute_2d_tuning_curves(group = pyr2, features = rot_pos[['x', 'z']], ep = evenep, nb_bins=24)  
        pxx2 = occupancy_prob(rot_pos, evenep, nb_bins=24, norm = True)  
        SI2 = nap.compute_2d_mutual_info(pf2, rot_pos[['x', 'z']], ep = evenep)
        
        keep = []
        oddeven = []
        
        for i in pyr2.keys(): 
            
            pf1[i][np.isnan(pf1[i])] = 0
            pf1[i] = scipy.ndimage.gaussian_filter(pf1[i], 1.5, mode = 'nearest')
            masked_array = np.ma.masked_where(pxx1 == 0, pf1[i]) #should work fine without it 
            pf1[i] = masked_array
            
            pf2[i][np.isnan(pf2[i])] = 0
            pf2[i] = scipy.ndimage.gaussian_filter(pf2[i], 1.5, mode = 'nearest')
            masked_array = np.ma.masked_where(pxx2 == 0, pf2[i]) #should work fine without it 
            pf2[i] = masked_array
        
        for k in pyr2:
        
            good = np.logical_and(np.isfinite(pf1[k].flatten()), np.isfinite(pf2[k].flatten()))
            corr, p = scipy.stats.pearsonr(pf1[k].flatten()[good], pf2[k].flatten()[good]) 
            oddeven.append(corr)
            
            
            if corr > 0.53: ###CA1
            # if corr > 0.54: ###CA3
                keep.append(k)
            
                
        pyr3 = pyr2[keep]
        # pyr3 = pyr2 
        
#%% 

        # if s == 'B2625-240321':
        # if len(pyr3) > 0:
        #     if isWT == 0:
            
        # for i,n in enumerate(pyr3):
        #     plt.figure()
        #     good = np.logical_and(np.isfinite(pf1[n].flatten()), np.isfinite(pf2[n].flatten()))
        #     corr, _ = scipy.stats.pearsonr(pf1[n].flatten()[good], pf2[n].flatten()[good]) 
        #     plt.suptitle(s + '_cell' + str(n) + '_OddevenR = '  + str(round(corr, 2)))
        #     plt.subplot(121)
        #     plt.title(round(SI1['SI'][n],2))
        #     plt.imshow(pf1[n], extent=(binsxy[1][0],binsxy[1][-1],binsxy[0][0],binsxy[0][-1]), cmap = 'jet')        
        #     plt.colorbar()
        #     plt.subplot(122)
        #     plt.title(round(SI2['SI'][n],2))
        #     plt.imshow(pf2[n], extent=(binsxy[1][0],binsxy[1][-1],binsxy[0][0],binsxy[0][-1]), cmap = 'jet')        
        #     plt.colorbar()
                    
        
#%% 

        if len(pyr3) > 0:     
        #     print('yes')
   
            placefields1, binsxy1 = nap.compute_2d_tuning_curves(group = pyr3, 
                                                                features = rot_pos[['x', 'z']], 
                                                                ep = ep1, 
                                                                nb_bins=24)  
                   
            px1 = occupancy_prob(rot_pos, ep1, nb_bins=24, norm = True)
            spatialinfo_env1 = nap.compute_2d_mutual_info(placefields1, rot_pos[['x', 'z']], ep = ep1)
            
            placefields2, binsxy2 = nap.compute_2d_tuning_curves(group = pyr3, 
                                                                features = rot_pos[['x', 'z']], 
                                                                ep = ep2, 
                                                                nb_bins=24)
                    
            px2 = occupancy_prob(rot_pos, ep2, nb_bins=24, norm = True)
            
            SI_2 = nap.compute_2d_mutual_info(placefields2, rot_pos[['x', 'z']], ep = ep2)
            spatialinfo_env2 = nap.compute_2d_mutual_info(placefields2, rot_pos[['x', 'z']], ep = ep2)       
                
                       
            
            sp1 = [] 
            sp2 = []
            for k in pyr3:
                tmp = sparsity(placefields1[k], px1)
                tmp2 = sparsity(placefields2[k], px2)
                sp1.append(tmp)
                sp2.append(tmp2)
            
            
            for i in pyr3.keys(): 
                
                placefields1[i][np.isnan(placefields1[i])] = 0
                placefields1[i] = scipy.ndimage.gaussian_filter(placefields1[i], 1.5, mode = 'nearest')
                masked_array = np.ma.masked_where(px1 == 0, placefields1[i]) #should work fine without it 
                placefields1[i] = masked_array
                
                placefields2[i][np.isnan(placefields2[i])] = 0
                placefields2[i] = scipy.ndimage.gaussian_filter(placefields2[i], 1.5, mode = 'nearest')
                masked_array = np.ma.masked_where(px2 == 0, placefields2[i]) #should work fine without it 
                placefields2[i] = masked_array
                
            if isWT == 1:
                allspatialinfo_env1_wt.extend(spatialinfo_env1['SI'].tolist())
                allspatialinfo_env2_wt.extend(spatialinfo_env2['SI'].tolist())
                sparsity1_wt.extend(sp1)
                sparsity2_wt.extend(sp2)
                oddevencorr_wt.extend(oddeven)
                # pvec1_wt.append(pvec1)
                # pvec2_wt.append(pvec2)
                npyr3_wt.append(len(pyr3))
            else: 
                allspatialinfo_env1_ko.extend(spatialinfo_env1['SI'].tolist())
                allspatialinfo_env2_ko.extend(spatialinfo_env2['SI'].tolist())
                sparsity1_ko.extend(sp1)
                sparsity2_ko.extend(sp2)
                oddevencorr_ko.extend(oddeven)
                # pvec1_ko.append(pvec1)
                # pvec2_ko.append(pvec2)
                pvsess_ko.append(name)
                npyr3_ko.append(len(pyr3))
            
            if len(pyr3) >= 5:
                normpf1 = {}
                for i in placefields1.keys():
                    normpf1[i] = placefields1[i] / np.max(placefields1[i])
                    
                normpf2 = {}
                for i in placefields2.keys():
                    normpf2[i] = placefields2[i] / np.max(placefields2[i])
                    
                pvcorr = compute_PVcorrs(normpf1, normpf2, pyr3.index)
                
                if isWT == 1:
                    pvcorr_wt.append(pvcorr)
                    pvsess_wt.append(name)
                else: 
                    pvcorr_ko.append(pvcorr)
                    pvsess_ko.append(name)
                   
                 
            # plt.figure()
            # plt.title(s + '_1')
            # plt.imshow(compute_population_vectors(normpf1, pyr3.index))
            # plt.colorbar()
            
            # plt.figure()
            # plt.title(s + '_2')
            # plt.imshow(compute_population_vectors(normpf2, pyr3.index))
            # plt.colorbar()
            
                
                
            
            # pvcorr = compute_PVcorrs(placefields1, placefields2, pyr3.index)
            
           
            # pvec1 = compute_population_vectors(placefields1, pyr3.index)
            # pvec2 = compute_population_vectors(placefields2, pyr3.index)
                        
            # print(pvcorr)
            # print(len(pyr2), len(pyr3))
            
            
        
    # sys.exit()
    



#%% Plot remapping 
    
            # ref = pyr3.keys()
            # nrows = int(np.sqrt(len(ref)))
            # ncols = int(len(ref)/nrows)+1
        
            # plt.figure()
            # plt.suptitle(s + ' Wake1')
            # for i,n in enumerate(pyr3):
            #     plt.subplot(nrows, ncols, i+1)
            #     # plt.title(spikes._metadata['celltype'][i])
            #     # plt.imshow(placefields1[n], extent=(binsxy1[0][0],binsxy1[0][-1],binsxy1[1][0],binsxy1[1][-1]), cmap = 'jet')        
            #     plt.imshow(placefields1[n].T, extent=(binsxy1[0][0],binsxy1[0][-1],binsxy1[1][0],binsxy1[1][-1]), origin = 'lower', cmap = 'viridis')        
            #     # plt.imshow(placefields1[n].T, cmap = 'viridis', aspect = 'auto', origin = 'lower')   
            #     plt.colorbar()
        
            # plt.figure()
            # plt.suptitle(s + ' Wake2')
            # for i,n in enumerate(pyr3):
            #     plt.subplot(nrows, ncols, i+1)
            #     # plt.title(spikes._metadata['celltype'][i])
            #     # plt.imshow(placefields2[n], extent=(binsxy2[0][0],binsxy2[0][-1],binsxy2[1][0],binsxy2[1][-1]), cmap = 'jet')        
            #     plt.imshow(placefields2[n].T, extent=(binsxy2[0][0],binsxy2[0][-1],binsxy2[1][0],binsxy2[1][-1]), origin = 'lower', cmap = 'viridis')        
            #     # plt.imshow(placefields2[n].T, cmap = 'viridis', aspect = 'auto', origin = 'lower')   
                
            #     plt.colorbar()
    
    
# ###EXAMPLES 
    # for i,n in enumerate(pyr2):
    #     plt.figure()
    #     good = np.logical_and(np.isfinite(placefields1[n].flatten()), np.isfinite(placefields2[n].flatten()))
    #     corr, _ = scipy.stats.pearsonr(placefields1[n].flatten()[good], placefields2[n].flatten()[good]) 
    #     plt.suptitle('R = '  + str(round(corr, 2)))
    #     plt.subplot(121)
    #     plt.imshow(placefields1[n], extent=(binsxy1[1][0],binsxy1[1][-1],binsxy1[0][0],binsxy1[0][-1]), cmap = 'jet')        
    #     plt.colorbar()
    #     plt.subplot(122)
    #     plt.imshow(placefields2[n], extent=(binsxy2[1][0],binsxy2[1][-1],binsxy2[0][0],binsxy2[0][-1]), cmap = 'jet')        
    #     plt.colorbar()

    
#%% Split both wake epochs into halves 

            center1 = rot_pos.restrict(nap.IntervalSet(epochs['wake'][0])).time_support.get_intervals_center()
                      
            center2 = rot_pos.restrict(nap.IntervalSet(epochs['wake'][1])).time_support.get_intervals_center()
            
            halves1 = nap.IntervalSet(start = [rot_pos.restrict(nap.IntervalSet(epochs['wake'].loc[[0]])).time_support.start[0], center1.t[0]],
                                      end = [center1.t[0], rot_pos.restrict(nap.IntervalSet(epochs['wake'].loc[[0]])).time_support.end[0]])
        
            halves2 = nap.IntervalSet(start = [rot_pos.restrict(nap.IntervalSet(epochs['wake'].loc[[1]])).time_support.start[0], center2.t[0]],
                                      end = [center2.t[0], rot_pos.restrict(nap.IntervalSet(epochs['wake'].loc[[1]])).time_support.end[0]])
        
            
            ep_wake1 = halves1.intersect(moving_ep)
            ep_wake2 = halves2.intersect(moving_ep)
                
            half1_wake1 = ep_wake1[0:len(ep_wake1)//2]
            half2_wake1 = ep_wake1[(len(ep_wake1)//2)+1:]
            
            half1_wake2 = ep_wake2[0:len(ep_wake2)//2]
            half2_wake2 = ep_wake2[(len(ep_wake2)//2)+1:]
                
            pf1, binsxy = nap.compute_2d_tuning_curves(group = pyr3, features = rot_pos[['x', 'z']], ep = half1_wake1, nb_bins=24)  
            px1 = occupancy_prob(rot_pos, half1_wake1, nb_bins=24)
            spatialinfo1 = nap.compute_2d_mutual_info(pf1, rot_pos[['x', 'z']], ep = half1_wake1)
            
            norm_px1 = occupancy_prob(rot_pos, half1_wake1, nb_bins=24, norm=True)
            norm_px2 = occupancy_prob(rot_pos, half1_wake2, nb_bins=24, norm=True)
            norm_px3 = occupancy_prob(rot_pos, half2_wake1, nb_bins=24, norm=True)
            norm_px4 = occupancy_prob(rot_pos, half2_wake2, nb_bins=24, norm=True)
             
            
            pf2, binsxy = nap.compute_2d_tuning_curves(group = pyr3, features = rot_pos[['x', 'z']], ep = half2_wake1, nb_bins=24)  
            px2 = occupancy_prob(rot_pos, half2_wake1, nb_bins=24)
            spatialinfo2 = nap.compute_2d_mutual_info(pf2, rot_pos[['x', 'z']], ep = half2_wake1)
        
            pf3, binsxy = nap.compute_2d_tuning_curves(group = pyr3, features = rot_pos[['x', 'z']], ep = half1_wake2, nb_bins=24)  
            px3 = occupancy_prob(rot_pos, half1_wake2, nb_bins=24)
            spatialinfo3 = nap.compute_2d_mutual_info(pf3, rot_pos[['x', 'z']], ep = half1_wake2)
                
            pf4, binsxy = nap.compute_2d_tuning_curves(group = pyr3, features = rot_pos[['x', 'z']], ep = half2_wake2, nb_bins=24)  
            px4 = occupancy_prob(rot_pos, half2_wake2, nb_bins=24)
            spatialinfo4 = nap.compute_2d_mutual_info(pf4, rot_pos[['x', 'z']], ep = half2_wake2)
            
            for i in pyr3.keys(): 
                pf1[i][np.isnan(pf1[i])] = 0
                pf1[i] = scipy.ndimage.gaussian_filter(pf1[i], 1.5, mode = 'nearest')
                masked_array = np.ma.masked_where(px1 == 0, pf1[i]) #should work fine without it 
                pf1[i] = masked_array
                
                pf2[i][np.isnan(pf2[i])] = 0
                pf2[i] = scipy.ndimage.gaussian_filter(pf2[i], 1.5, mode = 'nearest')
                masked_array = np.ma.masked_where(px2 == 0, pf2[i]) #should work fine without it 
                pf2[i] = masked_array
                
                pf3[i][np.isnan(pf3[i])] = 0
                pf3[i] = scipy.ndimage.gaussian_filter(pf3[i], 1.5, mode = 'nearest')
                masked_array = np.ma.masked_where(px3 == 0, pf3[i]) #should work fine without it 
                pf3[i] = masked_array
                
                pf4[i][np.isnan(pf4[i])] = 0
                pf4[i] = scipy.ndimage.gaussian_filter(pf4[i], 1.5, mode = 'nearest')
                masked_array = np.ma.masked_where(px4 == 0, pf4[i]) #should work fine without it 
                pf4[i] = masked_array
                        
            
            if isWT == 1:
                SI1_wt.extend(spatialinfo1.values)
                SI2_wt.extend(spatialinfo2.values)
                SI3_wt.extend(spatialinfo3.values)
                SI4_wt.extend(spatialinfo4.values)
            else: 
                SI1_ko.extend(spatialinfo1.values)
                SI2_ko.extend(spatialinfo2.values)
                SI3_ko.extend(spatialinfo3.values)
                SI4_ko.extend(spatialinfo4.values)
        
        
   
        
#%% Quantify spatial maps between 2 environments 

            for k in pyr3:
                
            #     ###Between 2 environments
                good = np.logical_and(np.isfinite(placefields1[k].flatten()), np.isfinite(placefields2[k].flatten()))
                corr, p = scipy.stats.pearsonr(placefields1[k].flatten()[good], placefields2[k].flatten()[good]) 
                
                if isWT == 1:
                    env_stability_wt.append(corr)
                    sess_wt.append(s)
                else: 
                    env_stability_ko.append(corr)
                    sess_ko.append(s)
                
            #     ###Between 2 halves of first wake 
                good2 = np.logical_and(np.isfinite(pf1[k].flatten()), np.isfinite(pf2[k].flatten()))
                corr2, p2 = scipy.stats.pearsonr(pf1[k].flatten()[good2], pf2[k].flatten()[good2]) 
                
                if isWT == 1:
                    halfsession1_corr_wt.append(corr2)
                else:
                    halfsession1_corr_ko.append(corr2)
                
            #     ###Between 2 halves of second wake 
                good3 = np.logical_and(np.isfinite(pf3[k].flatten()), np.isfinite(pf4[k].flatten()))
                corr3, p3 = scipy.stats.pearsonr(pf3[k].flatten()[good3], pf4[k].flatten()[good3]) 
                
                if isWT == 1:
                    halfsession2_corr_wt.append(corr3)
                else: 
                    halfsession2_corr_ko.append(corr3)
        
### PLOT EXAMPLES ARENA 1 
    # if isWT == 0:
    # for i,n in enumerate(pyr3):
    #     plt.figure()
    #     good = np.logical_and(np.isfinite(pf1[n].flatten()), np.isfinite(pf2[n].flatten()))
    #     corr, _ = scipy.stats.pearsonr(pf1[n].flatten()[good], pf2[n].flatten()[good]) 
    #     plt.suptitle(s + '_cell' + str(n) + '_HalfsessionR = '  + str(round(corr, 2)))
    #     plt.subplot(121)
    #     plt.title(round(spatialinfo1['SI'][n],2))
    #     plt.imshow(pf1[n], extent=(binsxy[1][0],binsxy[1][-1],binsxy[0][0],binsxy[0][-1]), cmap = 'jet')        
    #     plt.colorbar()
    #     plt.subplot(122)
    #     plt.title(round(spatialinfo2['SI'][n],2))
    #     plt.imshow(pf2[n], extent=(binsxy[1][0],binsxy[1][-1],binsxy[0][0],binsxy[0][-1]), cmap = 'jet')        
    #     plt.colorbar()
        
        # sys.exit()
        
### PLOT EXAMPLES ARENA 2
    
    # for i,n in enumerate(pyr3):
    #     plt.figure()
    #     good = np.logical_and(np.isfinite(pf3[n].flatten()), np.isfinite(pf4[n].flatten()))
    #     corr, _ = scipy.stats.pearsonr(pf3[n].flatten()[good], pf4[n].flatten()[good]) 
    #     plt.suptitle('R = '  + str(round(corr, 2)))
    #     plt.subplot(121)
    #     plt.title(round(spatialinfo3['SI'][n],2))
    #     plt.imshow(pf3[n], extent=(binsxy[1][0],binsxy[1][-1],binsxy[0][0],binsxy[0][-1]), cmap = 'jet')        
    #     plt.colorbar()
    #     plt.subplot(122)
    #     plt.title(round(spatialinfo4['SI'][n],2))
    #     plt.imshow(pf4[n], extent=(binsxy[1][0],binsxy[1][-1],binsxy[0][0],binsxy[0][-1]), cmap = 'jet')        
    #     plt.colorbar()
        
        
        
#%% Organize and plot environment stability data 
    
###Across 2 envs

wt = np.array(['WT' for x in range(len(env_stability_wt))])
ko = np.array(['KO' for x in range(len(env_stability_ko))])

genotype = np.hstack([wt, ko])

sinfos = []
sinfos.extend(env_stability_wt)
sinfos.extend(env_stability_ko)

sinfos2 = []
sinfos2.extend(sess_wt)
sinfos2.extend(sess_ko)


allinfos = pd.DataFrame(data = [sinfos, genotype, sinfos2], index = ['corr', 'type', 'sess']).T

color_labels = allinfos['sess'].unique()
rgb_values = sns.color_palette("viridis", len(color_labels))
color_map = dict(zip(color_labels, rgb_values))

###Half-session corr: 1st env

wt = np.array(['WT' for x in range(len(halfsession1_corr_wt))])
ko = np.array(['KO' for x in range(len(halfsession1_corr_ko))])

genotype2 = np.hstack([wt, ko])

sinfos2 = []
sinfos2.extend(halfsession1_corr_wt)
sinfos2.extend(halfsession1_corr_ko)

allinfos2 = pd.DataFrame(data = [sinfos2, genotype2], index = ['corr', 'type']).T

# ###Half-session corr: 2nd env 

wt = np.array(['WT' for x in range(len(halfsession2_corr_wt))])
ko = np.array(['KO' for x in range(len(halfsession2_corr_ko))])

genotype3 = np.hstack([wt, ko])

sinfos3 = []
sinfos3.extend(halfsession2_corr_wt)
sinfos3.extend(halfsession2_corr_ko)

allinfos3 = pd.DataFrame(data = [sinfos3, genotype3], index = ['corr', 'type']).T

#%% 

# plt.figure()
# plt.suptitle('Remapping')

# # plt.subplot(131)
# plt.title('A v/s B')
# sns.set_style('white')
# palette = ['royalblue', 'indianred']
# ax = sns.violinplot( x = allinfos['type'], y=allinfos['corr'].astype(float) , data = allinfos, dodge=False,
#                     palette = palette,cut = 2,
#                     scale="width", inner=None)
# ax.tick_params(bottom=True, left=True)
# xlim = ax.get_xlim()
# ylim = ax.get_ylim()
# for violin in ax.collections:
#     x0, y0, width, height = violin.get_paths()[0].get_extents().bounds
#     violin.set_clip_path(plt.Rectangle((x0, y0), width / 2, height, transform=ax.transData))
# sns.boxplot(x = allinfos['type'], y=allinfos['corr'].astype(float) , data = allinfos, saturation=1, showfliers=False,
#             width=0.3, boxprops={'zorder': 3, 'facecolor': 'none'}, ax=ax)
# old_len_collections = len(ax.collections)
# sns.stripplot(x = allinfos['type'], y = allinfos['corr'].astype(float), data = allinfos, 
#               hue = allinfos['sess'], palette=sns.color_palette('inferno_r', len(allinfos['sess'].unique())), 
#                                                                 dodge=False, ax=ax, legend=False)
# # sns.swarmplot(x = durdf['genotype'], y = durdf['dur'].astype(float), data = durdf, color = 'k', dodge=False, ax=ax)
# for dots in ax.collections[old_len_collections:]:
#     dots.set_offsets(dots.get_offsets())
# ax.set_xlim(xlim)
# ax.set_ylim(ylim)
# plt.ylabel('Correlation (R)')
# plt.axhline(0, linestyle = '--', color = 'silver')
# ax.set_box_aspect(1)

#%% 
bins = np.linspace(-0.75, 0.8, 100) ##CA1 
# bins = np.linspace(-0.35, 1, 100) ##CA3

xcenters = np.mean(np.vstack([bins[:-1], bins[1:]]), axis = 0)
wtcounts, _ = np.histogram(env_stability_wt, bins)
kocounts, _ = np.histogram(env_stability_ko, bins)

wtprop = wtcounts/sum(wtcounts)
koprop = kocounts/sum(kocounts)

plt.figure()
plt.title('Remapping Correlation')
plt.plot(xcenters, np.cumsum(wtprop), color = 'royalblue', label  = 'WT')
plt.plot(xcenters, np.cumsum(koprop), color = 'indianred', label  = 'KO')
plt.axvline(0, linestyle = '--', color = 'silver')
plt.xlabel('Correlation (R)')
plt.ylabel('% cells')
plt.legend(loc = 'upper left')
plt.gca().set_box_aspect(1)

#%% 

# plt.subplot(132)
# plt.title('A1 v/s A2')
# sns.set_style('white')
# palette = ['royalblue', 'indianred']
# ax = sns.violinplot( x = allinfos2['type'], y=allinfos2['corr'].astype(float) , data = allinfos2, dodge=False,
#                     palette = palette,cut = 2,
#                     scale="width", inner=None)
# ax.tick_params(bottom=True, left=True)
# xlim = ax.get_xlim()
# ylim = ax.get_ylim()
# for violin in ax.collections:
#     x0, y0, width, height = violin.get_paths()[0].get_extents().bounds
#     violin.set_clip_path(plt.Rectangle((x0, y0), width / 2, height, transform=ax.transData))
# sns.boxplot(x = allinfos2['type'], y=allinfos2['corr'].astype(float) , data = allinfos2, saturation=1, showfliers=False,
#             width=0.3, boxprops={'zorder': 3, 'facecolor': 'none'}, ax=ax)
# old_len_collections = len(ax.collections)
# sns.stripplot(x = allinfos2['type'], y = allinfos2['corr'].astype(float), data = allinfos2, 
#               hue = allinfos['sess'], palette=sns.color_palette('inferno_r', len(allinfos['sess'].unique())), 
#                                                                   dodge=False, ax=ax, legend=False)
# # sns.swarmplot(x = durdf['genotype'], y = durdf['dur'].astype(float), data = durdf, color = 'k', dodge=False, ax=ax)
# for dots in ax.collections[old_len_collections:]:
#     dots.set_offsets(dots.get_offsets())
# ax.set_xlim(xlim)
# ax.set_ylim(ylim)
# plt.ylabel('Correlation (R)')
# ax.set_box_aspect(1)

# plt.subplot(133)
# plt.title('B1 v/s B2')
# sns.set_style('white')
# palette = ['royalblue', 'indianred']
# ax = sns.violinplot( x = allinfos3['type'], y=allinfos3['corr'].astype(float) , data = allinfos3, dodge=False,
#                     palette = palette,cut = 2,
#                     scale="width", inner=None)
# ax.tick_params(bottom=True, left=True)
# xlim = ax.get_xlim()
# ylim = ax.get_ylim()
# for violin in ax.collections:
#     x0, y0, width, height = violin.get_paths()[0].get_extents().bounds
#     violin.set_clip_path(plt.Rectangle((x0, y0), width / 2, height, transform=ax.transData))
# sns.boxplot(x = allinfos3['type'], y=allinfos3['corr'].astype(float) , data = allinfos3, saturation=1, showfliers=False,
#             width=0.3, boxprops={'zorder': 3, 'facecolor': 'none'}, ax=ax)
# old_len_collections = len(ax.collections)
# sns.stripplot(x = allinfos3['type'], y = allinfos3['corr'].astype(float), data = allinfos3, 
#               hue = allinfos['sess'], palette=sns.color_palette('inferno_r', len(allinfos['sess'].unique())), dodge=False, ax=ax)
# # sns.swarmplot(x = durdf['genotype'], y = durdf['dur'].astype(float), data = durdf, color = 'k', dodge=False, ax=ax)
# for dots in ax.collections[old_len_collections:]:
#     dots.set_offsets(dots.get_offsets())
# ax.set_xlim(xlim)
# ax.set_ylim(ylim)
# plt.ylabel('Correlation (R)')
# ax.set_box_aspect(1)


#%% Stats for remapping 

t, p = mannwhitneyu(env_stability_wt, env_stability_ko)    

z_wt, p_wt = wilcoxon(np.array(env_stability_wt)-0)
z_ko, p_ko = wilcoxon(np.array(env_stability_ko)-0)

t2, p2 = mannwhitneyu(halfsession1_corr_wt, halfsession1_corr_ko)
t3, p3 = mannwhitneyu(halfsession2_corr_wt, halfsession2_corr_ko)
        
#%% Plot Example cells 

# examples = [3, 6]

# for n in examples:
#     plt.figure()
#     # peakfreq = max(placefields1[n].max(), placefields2[n].max()) 
#     pf1 = placefields1[n] / placefields1[n].max()
#     pf2 = placefields2[n] / placefields2[n].max()
    
    
#     plt.subplot(1,2,1)
#     plt.title('FR=' + str(round(placefields1[n].max(),2)) + '_SI=' + str(round(spatialinfo_env1.loc[n],2)))
#     plt.imshow(pf1.T, cmap = 'viridis', aspect = 'auto', origin = 'lower', vmin = 0, vmax = 1)   
#     plt.tight_layout()
#     plt.gca().set_box_aspect(1)
#     plt.subplot(1,2,2)
#     plt.title('FR=' + str(round(placefields2[n].max(),2)) + '_SI=' + str(round(spatialinfo_env2.loc[n],2)))
#     plt.imshow(pf2.T, cmap = 'viridis', aspect = 'auto', origin = 'lower', vmin = 0, vmax = 1)   
#     # plt.colorbar()
#     plt.gca().set_box_aspect(1)
#     plt.tight_layout()
    
    
#     plt.figure()
#     plt.subplot(121)
#     plt.plot(rot_pos['x'].restrict(ep1), rot_pos['z'].restrict(ep1), color = 'grey')
#     spk_pos1 = pyr2[n].value_from(rot_pos.restrict(ep1))
#     plt.plot(spk_pos1['x'], spk_pos1['z'], 'o', color = 'r', markersize = 5, alpha = 0.5)
#     plt.gca().set_box_aspect(1)
#     plt.subplot(122)
#     plt.plot(rot_pos['x'].restrict(ep2), rot_pos['z'].restrict(ep2), color = 'grey')
#     spk_pos2 = pyr2[n].value_from(rot_pos.restrict(ep2))
#     plt.plot(spk_pos2['x'], spk_pos2['z'], 'o', color = 'r', markersize = 5, alpha = 0.5)
#     plt.gca().set_box_aspect(1)    
    

# plt.figure()
# for i,n in enumerate(spikes):
#     plt.subplot(4,5,i+1)
#     plt.plot(rot_pos['x'].restrict(ep2), rot_pos['z'].restrict(ep2), color = 'grey')    
#     spk_pos1 = spikes[i].value_from(rot_pos.restrict(ep2))    
#     plt.plot(spk_pos1['x'], spk_pos1['z'], 'o', color = 'r', markersize = 0.32, alpha = 0.5)
#     plt.gca().set_box_aspect(1)
   

#%% Organize spatial information data 

wt1 = np.array(['WT' for x in range(len(allspatialinfo_env1_wt))])
ko1 = np.array(['KO' for x in range(len(allspatialinfo_env1_ko))])

genotype = np.hstack([wt1, ko1])

sinfos1 = []
sinfos1.extend(allspatialinfo_env1_wt)
sinfos1.extend(allspatialinfo_env1_ko)

allinfos1 = pd.DataFrame(data = [sinfos1, genotype], index = ['SI', 'genotype']).T

wt2 = np.array(['WT' for x in range(len(allspatialinfo_env2_wt))])
ko2 = np.array(['KO' for x in range(len(allspatialinfo_env2_ko))])

genotype = np.hstack([wt2, ko2])

sinfos2 = []
sinfos2.extend(allspatialinfo_env2_wt)
sinfos2.extend(allspatialinfo_env2_ko)

allinfos2 = pd.DataFrame(data = [sinfos2, genotype], index = ['SI', 'genotype']).T


#%% Plotting SI

# plt.figure()
# plt.suptitle('Spatial Information')
# # plt.subplot(121)
# plt.title('Arena A')
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
# # sns.swarmplot(x = wakedf['type'], y = wakedf['rate'].astype(float), data = wakedf, color = 'k', dodge=False, ax=ax)
# for dots in ax.collections[old_len_collections:]:
#     dots.set_offsets(dots.get_offsets())
# ax.set_xlim(xlim)
# ax.set_ylim(ylim)
# plt.ylabel('Spatial Information (bits per spike)')
# ax.set_box_aspect(1)

# plt.subplot(122)
# plt.title('Arena B')
# sns.set_style('white')
# palette = ['royalblue', 'indianred'] 
# ax = sns.violinplot( x = allinfos2['genotype'], y=allinfos2['SI'].astype(float) , data = allinfos2, dodge=False,
#                     palette = palette,cut = 2,
#                     scale="width", inner=None)
# ax.tick_params(bottom=True, left=True)
# xlim = ax.get_xlim()
# ylim = ax.get_ylim()
# for violin in ax.collections:
#     x0, y0, width, height = violin.get_paths()[0].get_extents().bounds
#     violin.set_clip_path(plt.Rectangle((x0, y0), width / 2, height, transform=ax.transData))
# sns.boxplot(x = allinfos2['genotype'], y=allinfos2['SI'].astype(float) , data = allinfos2, saturation=1, showfliers=False,
#             width=0.3, boxprops={'zorder': 3, 'facecolor': 'none'}, ax=ax)
# old_len_collections = len(ax.collections)
# sns.stripplot(x = allinfos2['genotype'], y = allinfos2['SI'].astype(float), data = allinfos2, color = 'k', dodge=False, ax=ax, alpha = 0.2)
# # sns.swarmplot(x = wakedf['type'], y = wakedf['rate'].astype(float), data = wakedf, color = 'k', dodge=False, ax=ax)
# for dots in ax.collections[old_len_collections:]:
#     dots.set_offsets(dots.get_offsets())
# ax.set_xlim(xlim)
# ax.set_ylim(ylim)
# plt.ylabel('Spatial Information (bits per spike)')
# ax.set_box_aspect(1)


#%% Stats

t_env1, p_env1 = mannwhitneyu(allspatialinfo_env1_wt, allspatialinfo_env1_ko)
t_env2, p_env2 = mannwhitneyu(allspatialinfo_env2_wt, allspatialinfo_env2_ko)    

#%% Organize sparsity data 

wt1 = np.array(['WT' for x in range(len(sparsity1_wt))])
ko1 = np.array(['KO' for x in range(len(sparsity1_ko))])

genotype = np.hstack([wt1, ko1])

sinfos1 = []
sinfos1.extend(sparsity1_wt)
sinfos1.extend(sparsity1_ko)

allinfos3 = pd.DataFrame(data = [sinfos1, genotype], index = ['Sparsity', 'genotype']).T

wt2 = np.array(['WT' for x in range(len(sparsity2_wt))])
ko2 = np.array(['KO' for x in range(len(sparsity2_ko))])

genotype = np.hstack([wt2, ko2])

sinfos2 = []
sinfos2.extend(sparsity2_wt)
sinfos2.extend(sparsity2_ko)

allinfos4 = pd.DataFrame(data = [sinfos2, genotype], index = ['Sparsity', 'genotype']).T


#%% Plotting Sparsity

# plt.figure()
# plt.suptitle('Sparsity')
# # plt.subplot(121)
# plt.title('Square arena')
# sns.set_style('white')
# palette = ['royalblue', 'indianred'] 
# ax = sns.violinplot( x = allinfos3['genotype'], y=allinfos3['Sparsity'].astype(float) , data = allinfos3, dodge=False,
#                     palette = palette,cut = 2,
#                     scale="width", inner=None)
# ax.tick_params(bottom=True, left=True)
# xlim = ax.get_xlim()
# ylim = ax.get_ylim()
# for violin in ax.collections:
#     x0, y0, width, height = violin.get_paths()[0].get_extents().bounds
#     violin.set_clip_path(plt.Rectangle((x0, y0), width / 2, height, transform=ax.transData))
# sns.boxplot(x = allinfos3['genotype'], y=allinfos3['Sparsity'].astype(float) , data = allinfos3, saturation=1, showfliers=False,
#             width=0.3, boxprops={'zorder': 3, 'facecolor': 'none'}, ax=ax)
# old_len_collections = len(ax.collections)
# sns.stripplot(x = allinfos3['genotype'], y = allinfos3['Sparsity'].astype(float), data = allinfos3, color = 'k', dodge=False, ax=ax, alpha = 0.2)
# # sns.swarmplot(x = wakedf['type'], y = wakedf['rate'].astype(float), data = wakedf, color = 'k', dodge=False, ax=ax)
# for dots in ax.collections[old_len_collections:]:
#     dots.set_offsets(dots.get_offsets())
# ax.set_xlim(xlim)
# ax.set_ylim(ylim)
# plt.ylabel('Place field sparsity')
# ax.set_box_aspect(1)

# plt.subplot(122)
# plt.title('Circular arena')
# sns.set_style('white')
# palette = ['royalblue', 'indianred'] 
# ax = sns.violinplot( x = allinfos4['genotype'], y=allinfos4['Sparsity'].astype(float) , data = allinfos4, dodge=False,
#                     palette = palette,cut = 2,
#                     scale="width", inner=None)
# ax.tick_params(bottom=True, left=True)
# xlim = ax.get_xlim()
# ylim = ax.get_ylim()
# for violin in ax.collections:
#     x0, y0, width, height = violin.get_paths()[0].get_extents().bounds
#     violin.set_clip_path(plt.Rectangle((x0, y0), width / 2, height, transform=ax.transData))
# sns.boxplot(x = allinfos4['genotype'], y=allinfos4['Sparsity'].astype(float) , data = allinfos4, saturation=1, showfliers=False,
#             width=0.3, boxprops={'zorder': 3, 'facecolor': 'none'}, ax=ax)
# old_len_collections = len(ax.collections)
# sns.stripplot(x = allinfos4['genotype'], y = allinfos4['Sparsity'].astype(float), data = allinfos4, color = 'k', dodge=False, ax=ax, alpha = 0.2)
# # sns.swarmplot(x = wakedf['type'], y = wakedf['rate'].astype(float), data = wakedf, color = 'k', dodge=False, ax=ax)
# for dots in ax.collections[old_len_collections:]:
#     dots.set_offsets(dots.get_offsets())
# ax.set_xlim(xlim)
# ax.set_ylim(ylim)
# plt.ylabel('Place field sparsity')
# ax.set_box_aspect(1)

#%% Stats for sparsity

t_env1, p_e1 = mannwhitneyu(sparsity1_wt, sparsity1_ko)
t_env2, p_e2 = mannwhitneyu(sparsity2_wt, sparsity2_ko)    

#%% Population Vector (with norm)

## PVs per animal 
# occ_wt = {}
# for i, num in enumerate(pvsess_wt):
#     if num in occ_wt:
#         occ_wt[num].append(i)
#     else:
#         occ_wt[num] = [i]
        
# occ_ko = {}
# for i, num in enumerate(pvsess_ko):
#     if num in occ_ko:
#         occ_ko[num].append(i)
#     else:
#         occ_ko[num] = [i]


# ### Get PV of animals together 

#     ###WT
# wt_pv1 = {}
# for i in occ_wt.keys():
#     wt_pv1[i] = [pvec1_wt[k] for k in occ_wt[i]]
    
# wt_pv2 = {}
# for i in occ_wt.keys():
#     wt_pv2[i] = [pvec2_wt[k] for k in occ_wt[i]]
 
#     ###KO
# ko_pv1 = {}
# for i in occ_ko.keys():
#     ko_pv1[i] = [pvec1_ko[k] for k in occ_ko[i]]
    
# ko_pv2 = {}
# for i in occ_ko.keys():
#     ko_pv2[i] = [pvec2_ko[k] for k in occ_ko[i]]
 
    
# ### Compute normalized PV     
   
#     ### WT 
# for i in wt_pv1.keys():
#     for j in range(len(wt_pv1[i])):
#         wt_pv1[i][j] = wt_pv1[i][j] / np.max(wt_pv1[i])
        
# for i in wt_pv2.keys():
#     for j in range(len(wt_pv2[i])):
#         wt_pv2[i][j] = wt_pv2[i][j] / np.max(wt_pv2[i])
        
# norm_pv1_wt = []         
# for i in wt_pv1.keys():
#     for j in range(len(wt_pv1[i])):
#         norm_pv1_wt.append(wt_pv1[i][j])
# n1_wt = {index: value for index, value in enumerate(norm_pv1_wt)}
        
# norm_pv2_wt = []         
# for i in wt_pv2.keys():
#     for j in range(len(wt_pv2[i])):
#         norm_pv2_wt.append(wt_pv2[i][j])
# n2_wt = {index: value for index, value in enumerate(norm_pv2_wt)}

#     ### KO 
# for i in ko_pv1.keys():
#     for j in range(len(ko_pv1[i])):
#         ko_pv1[i][j] = ko_pv1[i][j] / np.max(ko_pv1[i])
        
# for i in ko_pv2.keys():
#     for j in range(len(ko_pv2[i])):
#         ko_pv2[i][j] = ko_pv2[i][j] / np.max(ko_pv2[i])
        
# norm_pv1_ko = []         
# for i in ko_pv1.keys():
#     for j in range(len(ko_pv1[i])):
#         norm_pv1_ko.append(ko_pv1[i][j])
# n1_ko = {index: value for index, value in enumerate(norm_pv1_ko)}
        
# norm_pv2_ko = []         
# for i in ko_pv2.keys():
#     for j in range(len(ko_pv2[i])):
#         norm_pv2_ko.append(ko_pv2[i][j])
# n2_ko = {index: value for index, value in enumerate(norm_pv2_ko)}

# ### Correlate PVs

# pvcorr_wt = [] 
# for i in n1_wt.keys():
#     pvcorr_wt.append(population_vector_correlation(n1_wt[i],n2_wt[i])[0])
    
# pvcorr_ko = [] 
# for i in n1_ko.keys():
#     pvcorr_ko.append(population_vector_correlation(n1_ko[i],n2_ko[i])[0])

#%% Organize PV corr data 

wt1 = np.array(['WT' for x in range(len(pvcorr_wt))])
ko1 = np.array(['KO' for x in range(len(pvcorr_ko))])

genotype = np.hstack([wt1, ko1])

sinfos1 = []
sinfos1.extend(pvcorr_wt)
sinfos1.extend(pvcorr_ko)

allinfos3 = pd.DataFrame(data = [sinfos1, genotype], index = ['PVCorr', 'genotype']).T

#%% Plotting PV corr

# plt.figure()
# plt.title('Population vector correlation')
# sns.set_style('white')
# palette = ['royalblue', 'indianred'] 
# ax = sns.violinplot( x = allinfos3['genotype'], y=allinfos3['PVCorr'].astype(float) , data = allinfos3, dodge=False,
#                     palette = palette,cut = 2,
#                     scale="width", inner=None)
# ax.tick_params(bottom=True, left=True)
# xlim = ax.get_xlim()
# ylim = ax.get_ylim()
# for violin in ax.collections:
#     x0, y0, width, height = violin.get_paths()[0].get_extents().bounds
#     violin.set_clip_path(plt.Rectangle((x0, y0), width / 2, height, transform=ax.transData))
# sns.boxplot(x = allinfos3['genotype'], y=allinfos3['PVCorr'].astype(float) , data = allinfos3, saturation=1, showfliers=False,
#             width=0.3, boxprops={'zorder': 3, 'facecolor': 'none'}, ax=ax)
# old_len_collections = len(ax.collections)
# sns.stripplot(x = allinfos3['genotype'], y = allinfos3['PVCorr'].astype(float), data = allinfos3, color = 'k', dodge=False, ax=ax, alpha = 0.2)
# # sns.swarmplot(x = wakedf['type'], y = wakedf['rate'].astype(float), data = wakedf, color = 'k', dodge=False, ax=ax)
# for dots in ax.collections[old_len_collections:]:
#     dots.set_offsets(dots.get_offsets())
# ax.set_xlim(xlim)
# ax.set_ylim(ylim)
# plt.ylabel('Correlation (R)')
# plt.axhline(0, linestyle = '--', color = 'silver')
# ax.set_box_aspect(1)

#%% 

bins = np.linspace(-0.05, 0.2, 100) ###CA1
# bins = np.linspace(-0.2, 0.3, 100) ###CA3


xcenters = np.mean(np.vstack([bins[:-1], bins[1:]]), axis = 0)
wtcounts, _ = np.histogram(pvcorr_wt, bins)
kocounts, _ = np.histogram(pvcorr_ko, bins)

wtprop = wtcounts/sum(wtcounts)
koprop = kocounts/sum(kocounts)

plt.figure()
plt.title('Population Vector Correlation')
plt.plot(xcenters, np.cumsum(wtprop), color = 'royalblue', label  = 'WT')
plt.plot(xcenters, np.cumsum(koprop), color = 'indianred', label  = 'KO')
plt.axvline(0, linestyle = '--', color = 'silver')
plt.xlabel('Correlation (R)')
plt.ylabel('% sessions')
plt.legend(loc = 'upper left')
plt.gca().set_box_aspect(1)

#%% Stats for PV corr

z_pvcorr_wt, p_pvcorr_wt = wilcoxon(np.array(pvcorr_wt)-0)
z_pvcorr_ko, p_pvcorr_ko = wilcoxon(np.array(pvcorr_ko)-0)

#%% 

bins = np.linspace(-0.4, 1, 30) ###CA1 
# bins = np.linspace(-0.2, 1, 30) ###CA3

xcenters = np.mean(np.vstack([bins[:-1], bins[1:]]), axis = 0)
wtcounts, _ = np.histogram(oddevencorr_wt, bins)
kocounts, _ = np.histogram(oddevencorr_ko, bins)

wtprop = wtcounts/sum(wtcounts)
koprop = kocounts/sum(kocounts)

plt.figure()
# plt.plot(xcenters, 1-np.cumsum(wtprop), color = 'royalblue', label  = 'WT')
# plt.plot(xcenters, 1-np.cumsum(koprop), color = 'indianred', label  = 'KO')
plt.plot(xcenters, np.cumsum(wtprop), color = 'royalblue', label  = 'WT')
plt.plot(xcenters, np.cumsum(koprop), color = 'indianred', label  = 'KO')


plt.axvline(0.53, linestyle = '--', color = 'silver') ### CA1
# plt.axvline(0.54, linestyle = '--', color = 'silver') ### CA3

plt.xlabel('odd-even correlation (R)')
plt.ylabel('% cells')
plt.legend(loc = 'upper left')
plt.gca().set_box_aspect(1)


# plt.figure()
# plt.stairs(wtprop, bins, label = 'WT', color = 'royalblue' , linewidth = 2)
# plt.stairs(koprop, bins, label = 'KO', color = 'indianred' , linewidth = 2)
# plt.xlabel('odd-even correlation (R)')
# plt.ylabel('% events')
# plt.legend(loc = 'upper right')

#%% 

# plt.figure()
# plt.title('WT')
# plt.scatter(halfsession1_corr_wt, oddevencorr_wt, color = 'k', zorder = 3) 
# plt.gca().axline((min(min(halfsession1_corr_wt),min(oddevencorr_wt)),min(min(halfsession1_corr_wt),min(oddevencorr_wt)) ), slope=1, color = 'silver', linestyle = '--')
# plt.xlabel('Half Session R')
# plt.ylabel('Odd-Even R')
# plt.axis('square')

# plt.figure()
# plt.title('KO')
# plt.scatter(halfsession1_corr_ko, oddevencorr_ko, color = 'k', zorder = 3) 
# plt.gca().axline((min(min(halfsession1_corr_ko),min(oddevencorr_ko)),min(min(halfsession1_corr_ko),min(oddevencorr_ko)) ), slope=1, color = 'silver', linestyle = '--')
# plt.xlabel('Half Session R')
# plt.ylabel('Odd-Even R')
# plt.axis('square')

                     