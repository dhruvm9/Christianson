#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 16:48:18 2024

@author: adrien
"""

import numpy as np 
import pandas as pd 
import nwbmatic as ntm
import scipy.io
import pynapple as nap 
import os, sys
import matplotlib.pyplot as plt 

#%% Load functions 

def rotate_via_numpy(xy, radians):
    """xy is a tuple or array """
    x, y = xy
    c, s = np.cos(radians), np.sin(radians)
    j = np.array([[c, s], [-s, c]])
    m = np.dot(j, [x, y])

    return float(m.T[0]), float(m.T[1])

def occupancy_prob(position, ep, nb_bins=24, norm = False):
    pos= position[['x','z']]
    position_tsd = pos.restrict(ep)
    xpos = position_tsd[:,0]
    ypos = position_tsd[:,1]
    xbins = np.linspace(xpos.min(), xpos.max()+1e-6, nb_bins+1)
    ybins = np.linspace(ypos.min(), ypos.max()+1e-6, nb_bins+1) 
    occupancy, _, _ = np.histogram2d(ypos, xpos, [ybins,xbins])
    
    if norm is True:
        occupancy = occupancy/sum(occupancy)
        
    masked_array = np.ma.masked_where(occupancy == 0, occupancy) 
    masked_array = np.flipud(masked_array)
    return masked_array

#%% Load data

data_directory = '/media/DataAdrienBig/PeyracheLabData/Dhruv/Christianson/Screening/B2629-240617'

data = ntm.load_session(data_directory, 'neurosuite')
spikes = data.spikes
epochs = data.epochs
position = data.position

#%% Rotate position 

rot_pos = []
    
xypos = np.array(position[['x', 'z']])

##ENTER ROTATION VALUE HERE
rad = -0.1

for i in range(len(xypos)):
    newx, newy = rotate_via_numpy(xypos[i], rad)
    rot_pos.append((newx, newy))
    
rot_pos = nap.TsdFrame(t = position.index.values, d = rot_pos, columns = ['x', 'z'])

#%% Compute running speed to find moving epochs
        
speedbinsize = np.diff(rot_pos.index.values)[0]

time_bins = np.arange(rot_pos.index[0], rot_pos.index[-1] + speedbinsize, speedbinsize)
index = np.digitize(rot_pos.index.values, time_bins)
tmp = rot_pos.as_dataframe().groupby(index).mean()
tmp.index = time_bins[np.unique(index)-1]+(speedbinsize)/2
distance = np.sqrt(np.power(np.diff(tmp['x']), 2) + np.power(np.diff(tmp['z']), 2)) * 100 #in cm
speed = nap.Tsd(t = tmp.index.values[0:-1]+ speedbinsize/2, d = distance/speedbinsize) # in cm/s
 
moving_ep = nap.IntervalSet(speed.threshold(2).time_support) #Epochs in which speed is > 2 cm/s
ep = moving_ep.intersect(epochs['wake'])

#%% Plot tracking 

plt.figure()
plt.plot(rot_pos['x'].restrict(epochs['wake']), rot_pos['z'].restrict(epochs['wake']))

#%% Split both wake epochs into halves 

center = rot_pos.restrict(nap.IntervalSet(epochs['wake'])).time_support.get_intervals_center()

halves = nap.IntervalSet(start = [rot_pos.restrict(epochs['wake']).time_support.start[0], center.t[0]],
                          end = [center.t[0], rot_pos.restrict(epochs['wake']).time_support.end[0]])

ep_wake = halves.intersect(moving_ep)

half1 = ep_wake[0:len(ep_wake)//2]
half2 = ep_wake[(len(ep_wake)//2)+1:]

#%% Compute place fields over each half

pf1, binsxy = nap.compute_2d_tuning_curves(group = spikes, features = rot_pos[['x', 'z']], ep = half1, nb_bins=24)  
px1 = occupancy_prob(rot_pos, half1, nb_bins=24)
spatialinfo1 = nap.compute_2d_mutual_info(pf1, rot_pos[['x', 'z']], ep = half1)

pf2, binsxy = nap.compute_2d_tuning_curves(group = spikes, features = rot_pos[['x', 'z']], ep = half2, nb_bins=24)  
px2 = occupancy_prob(rot_pos, half2, nb_bins=24)
spatialinfo2 = nap.compute_2d_mutual_info(pf2, rot_pos[['x', 'z']], ep = half2)

for i in spikes.keys(): 
    pf1[i][np.isnan(pf1[i])] = 0
    pf1[i] = scipy.ndimage.gaussian_filter(pf1[i], 1.5, mode = 'nearest')
    masked_array = np.ma.masked_where(px1 == 0, pf1[i]) #should work fine without it 
    pf1[i] = masked_array
    
    pf2[i][np.isnan(pf2[i])] = 0
    pf2[i] = scipy.ndimage.gaussian_filter(pf2[i], 1.5, mode = 'nearest')
    masked_array = np.ma.masked_where(px2 == 0, pf2[i]) #should work fine without it 
    pf2[i] = masked_array
     
    
#%% Plotting 

for i,n in enumerate(spikes):
    plt.figure()
    good = np.logical_and(np.isfinite(pf1[n].flatten()), np.isfinite(pf2[n].flatten()))
    corr, _ = scipy.stats.pearsonr(pf1[n].flatten()[good], pf2[n].flatten()[good]) 
    plt.suptitle('R = '  + str(round(corr, 2)))
    plt.subplot(121)
    plt.title('SI = ' + str(round(spatialinfo1['SI'][n],2)))
    plt.imshow(pf1[n], extent=(binsxy[1][0],binsxy[1][-1],binsxy[0][0],binsxy[0][-1]), cmap = 'jet')        
    plt.colorbar()
    plt.subplot(122)
    plt.title('SI = ' + str(round(spatialinfo2['SI'][n],2)))
    plt.imshow(pf2[n], extent=(binsxy[1][0],binsxy[1][-1],binsxy[0][0],binsxy[0][-1]), cmap = 'jet')        
    plt.colorbar()
