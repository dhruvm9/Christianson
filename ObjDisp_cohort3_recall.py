#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 09:40:59 2024

@author: dhruv
"""

#%% Import all necessary libraries

import pynapple as nap

import numpy as np
import pandas as pd
import scipy.io
import os, sys
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import warnings 
from pylab import *
from scipy.stats import mannwhitneyu, wilcoxon, kruskal
import scikit_posthocs as sp
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd


#%% 

warnings.filterwarnings("ignore")

##Add path to data folder
# data_directory = '/media/DataDhruv/Recordings/Christianson/test_cohort3'
data_directory = '/media/DataDhruv/Recordings/Christianson/testF_cohort4'

##File with list of all sessions
datasets = np.genfromtxt(os.path.normpath(os.path.join(data_directory,'dataset.list')), delimiter = '\n', dtype = str, comments = '#')
displaced = np.genfromtxt(os.path.normpath(os.path.join(data_directory,'displaced.list')), delimiter = '\n', dtype = str, comments = '#')

## Variables to store final results 
allsex_phase1 = []
allsex_phase2 = []

allmice_phase1 = []
allmice_phase2 = []

all_lefts_phase1 = []
all_lefts_phase2 = []

all_rights_phase1 = []
all_rights_phase2 = []

genotype_phase1 = []
genotype_phase2 = []

allDI_phase1 = []
allDI_phase2 = []

##Loop through each session from the list

for r, s in enumerate(datasets):
    print(s)
    mousename = s.split('_')[0] #The first character is the mouse name
    sex = s.split('_')[1] #Second character is sex (ignore)
    taskphase = s.split('_')[-1] #Last character is encoding or recall phase (Modify these portions as needed)
    dispobj = displaced[r]
    
    im = plt.imread(data_directory + '/' + s + '.png')
    
    #Read the tracking data
    #reads an HDF5 file and stores its contents into a Pandas DataFrame named tracking_data.
    #r is reading only mode
    tracking_data =  pd.read_hdf(data_directory + '/' + s + '.h5', mode = 'r')

    #Columns containing x and y coordinates (Modify as needed)
    position_cols = [0,1,3,4,6,7,9,10,12,13]
    
    #Select only those columns with tracked data
    #It selects all rows (because of :) but only the columns specified by position_cols from the tracking_data DataFrame.
    trackedpos = tracking_data.iloc[:, position_cols]
    
    #A separate table for likelihood values - likilihood of what? of ears being in range or any body part?
    #using NumPy to vertically stack (combine) two columns from a Pandas DataFrame called tracking_data into a single NumPy array. 
    likelihoods = np.vstack([tracking_data.iloc[:,2], tracking_data.iloc[:,5], tracking_data.iloc[:,8]])
    
    # If recall phase, analyze only the first 3 min (180s) of task. Each video has 30 frames per second.
    # if taskphase == '2':
    trackedpos = trackedpos[0:30*180]
    likelihoods = likelihoods[:,0:30*180]
    
    #Cutoff value of likelihood for reliable tracking
    pcutoff = 0.6

    #Keep only those frames where likelihood for mouse body parts is above the cutoff
    #for loop generates a sequence of indices from 0 to number_of_columns - 1
    tokeep = []
    for i in range(shape(likelihoods)[1]):
        if (likelihoods[0][i] > pcutoff) and (likelihoods[1][i] > pcutoff)and (likelihoods[2][i]):
            tokeep.append(i)
    
    #X- and Y-column indices for bodyparts (ear + nose)
    x_ear = [0,2,4]
    y_ear = [1,3,5]

    #X- and Y- coordinates for ears
    all_x_ear = trackedpos.iloc[:, x_ear]
    all_y_ear = trackedpos.iloc[:, y_ear]
    
    #NEW CODE x_nose index is 4, y_nose index is 5:
    x_nose = 4
    y_nose = 5
    all_x_nose = trackedpos.iloc[:, x_nose]
    all_y_nose = trackedpos.iloc[:, y_nose]
    
    #X- and Y- coordinates for left object
    all_x_objL = trackedpos.iloc[:,6]
    all_y_objL = trackedpos.iloc[:,7]
    
    #X- and Y- coordinates for right object
    all_x_objR = trackedpos.iloc[:,8]
    all_y_objR = trackedpos.iloc[:,9]
    
    #Compute mean position of left object by averaging its position across time
    x_objL = all_x_objL.mean()
    y_objL = all_y_objL.mean()
    
    #Compute mean position of right object by averaging its position across time
    x_objR = all_x_objR.mean()
    y_objR = all_y_objR.mean()
    
    #Compute centroid of ears (proxy for head position) -- Add the nose to this!
    #x_cent and y_cent represent the average x and y coordinates, respectively, 
    #calculated by summing all x and y coordinates for each point across measurements and dividing by the total number of measurements.
    #we compare distance of nose to object to centroid of ears to object, and see which one is shorter
    x_cent = all_x_ear.sum(axis=1)/all_x_ear.iloc[0,:].shape[0]
    y_cent = all_y_ear.sum(axis=1)/all_y_ear.iloc[0,:].shape[0]

    #New variable having centroid positions
    mouseposition = np.zeros((len(x_cent),2))
    mouseposition[:,0] = x_cent 
    mouseposition[:,1] = y_cent
    
    #X and Y-coordinates of centroid
    x = mouseposition[:,0]
    y = mouseposition[:,1]
    
    #30 frames per second
    fs = 30
    
    #Using the above sampling create, assign a timestamp to each frame
    timestamps = x_cent.index.values/fs
    
    #Creating a position variable with centroids that exceed likelihood cutooff
    position = np.vstack([x, y]).T
    position = nap.TsdFrame(t = timestamps[tokeep], d = position[tokeep], columns = ['x', 'y'], time_units = 's')
    
    nosepos  = np.vstack([all_x_nose[tokeep], all_y_nose[tokeep]]).T
    nosepos = nap.TsdFrame(t = timestamps[tokeep], d = nosepos, columns = ['x', 'y'], time_units = 's')
    
    #Create a variable for the coordinates of the left and right object
    objL_coords = np.hstack([x_objL, y_objL])
    objR_coords = np.hstack([x_objR, y_objR])
    
#%% Speed calculation 

    # speedbinsize = np.diff(position.index.values)[0]
    # time_bins = np.arange(position.index[0], position.index[-1] + speedbinsize, speedbinsize)
    # index = np.digitize(position.index.values, time_bins)
    # tmp = position.as_dataframe().groupby(index).mean()
    # tmp.index = time_bins[np.unique(index)-1]+(speedbinsize)/2
    # distance = np.sqrt(np.power(np.diff(tmp['x']), 2) + np.power(np.diff(tmp['y']), 2)) 
    # speed = pd.Series(index = tmp.index.values[0:-1]+ speedbinsize/2, data = distance/speedbinsize) 
    # speed2 = speed.rolling(window = 25, win_type='gaussian', center=True, min_periods=1).mean(std=10)
    # speed2 = nap.Tsd(speed2)
         
#%% Building rectangle around objects and selecting times when animal is in rectangle

### COHORT3 MALE

    if sex == 'M':

        if dispobj == 'right':    
    
            rectL_inner = patches.Rectangle((x_objL - 26, y_objL - 28), 77, 120, linewidth=1, edgecolor='g', facecolor='none')
            rectL = patches.Rectangle((x_objL - 47.5, y_objL - 49.5), 120, 163, linewidth=1, edgecolor='b', facecolor='none')
            
            rectR_inner = patches.Rectangle((x_objR - 31, y_objR - 35), 75, 125, linewidth=1, edgecolor='g', facecolor='none')
            rectR = patches.Rectangle((x_objR - 52.5, y_objR - 56.5), 118, 168, linewidth=1, edgecolor='b', facecolor='none')
        
        else: 
            
            rectL_inner = patches.Rectangle((x_objL - 26, y_objL - 35), 77, 120, linewidth=1, edgecolor='g', facecolor='none')
            rectL = patches.Rectangle((x_objL - 47.5, y_objL - 56.5), 120, 163, linewidth=1, edgecolor='b', facecolor='none')
            
            rectR_inner = patches.Rectangle((x_objR - 31, y_objR - 25), 75, 125, linewidth=1, edgecolor='g', facecolor='none')
            rectR = patches.Rectangle((x_objR - 52.5, y_objR - 46.5), 118, 168, linewidth=1, edgecolor='b', facecolor='none')
        

### COHORT4 FEMALE

    else: 

        if dispobj == 'right' and (mousename not in ['1103', '1114', '1115', '1122', '2710', '2711']):  
            
            rectL_inner = patches.Rectangle((x_objL - 29, y_objL - 23), 77, 120, linewidth=1, edgecolor='g', facecolor='none')
            rectL = patches.Rectangle((x_objL - 50.5, y_objL - 44.5), 120, 163, linewidth=1, edgecolor='b', facecolor='none')
         
            rectR_inner = patches.Rectangle((x_objR - 30, y_objR - 35), 75, 125, linewidth=1, edgecolor='g', facecolor='none')
            rectR = patches.Rectangle((x_objR - 51.5, y_objR - 56.5), 118, 168, linewidth=1, edgecolor='b', facecolor='none')
            
        elif (dispobj == 'right') and (mousename in ['1103', '1114', '1115', '1122', '2710', '2711']):
                    
            rectL_inner = patches.Rectangle((x_objL - 29, y_objL - 23), 77, 120, linewidth=1, edgecolor='g', facecolor='none')
            rectL = patches.Rectangle((x_objL - 50.5, y_objL - 44.5), 120, 163, linewidth=1, edgecolor='b', facecolor='none')
         
            rectR_inner = patches.Rectangle((x_objR - 35, y_objR - 36), 75, 125, linewidth=1, edgecolor='g', facecolor='none')
            rectR = patches.Rectangle((x_objR - 56.5, y_objR - 57.5), 118, 168, linewidth=1, edgecolor='b', facecolor='none')
               
        else: 
            
            rectL_inner = patches.Rectangle((x_objL - 26, y_objL - 32), 77, 120, linewidth=1, edgecolor='g', facecolor='none')
            rectL = patches.Rectangle((x_objL - 47.5, y_objL - 53.5), 120, 163, linewidth=1, edgecolor='b', facecolor='none')
            
            rectR_inner = patches.Rectangle((x_objR - 36, y_objR - 25), 75, 125, linewidth=1, edgecolor='g', facecolor='none')
            rectR = patches.Rectangle((x_objR - 57.5, y_objR - 46.5), 118, 168, linewidth=1, edgecolor='b', facecolor='none')

      
    rectL_coords = rectL.get_corners()
    rectR_coords = rectR.get_corners()
    rectL_inner_coords = rectL_inner.get_corners()
    rectR_inner_coords = rectR_inner.get_corners()
    
    bbL = matplotlib.path.Path(rectL_coords)
    bbR = matplotlib.path.Path(rectR_coords)
    bbL_inner = matplotlib.path.Path(rectL_inner_coords)
    bbR_inner = matplotlib.path.Path(rectR_inner_coords)
    
       
    posvals = np.column_stack((position['x'].values, position['y'].values))
        
    inL = bbL.contains_points(posvals)
    inR = bbR.contains_points(posvals)
        
    inrectL_idx = np.where(inL == True)
    inrectR_idx = np.where(inR == True)
    
    if shape(inrectL_idx)[1] > 0:
        inrectL_times = nap.Ts(position.index.values[inrectL_idx])
        lefttimes = inrectL_times.find_support(2/30)
    else: lefttimes = nap.IntervalSet(start = 0, end = 0)
    
    if shape(inrectR_idx)[1] > 0:
        inrectR_times = nap.Ts(position.index.values[inrectR_idx])
        righttimes = inrectR_times.find_support(2/30)
    else: righttimes = nap.IntervalSet(start = 0, end = 0)
        
   
    #%% Plot 
    
    # if dispobj == 'left':
    
        # plt.figure()
        # plt.title(s)
        # plt.imshow(im, origin = 'lower')
        # plt.gca().add_patch(rectL)
        # plt.gca().add_patch(rectL_inner)
        # plt.gca().add_patch(rectR)
        # plt.gca().add_patch(rectR_inner)
        
    
    
    #%% 
    
    # plt.figure()
    # plt.title(s)
    # plt.imshow(im, origin = 'lower')
    # ax = sns.scatterplot(data = position.as_dataframe(), x = x[tokeep], y = y[tokeep])
    # ax.add_patch(rectL)
    # ax.add_patch(rectL_inner)
    # ax.add_patch(rectR)
    # ax.add_patch(rectR_inner)
    # test = nap.IntervalSet(start=32, end=33)
    # plt.plot(position['x'].restrict(test), position['y'].restrict(test), 'o', zorder = 2, label = 'left ROI', color = 'k')
    # plt.plot(nosepos['x'].restrict(test), nosepos['y'].restrict(test), 'o', zorder = 2, label = 'left ROI', color = 'r')
    
    
    
    # plt.plot(position['x'].restrict(lefttimes), position['y'].restrict(lefttimes), 'o', zorder = 2, label = 'left ROI', color = 'k')
    # plt.plot(position['x'].restrict(righttimes), position['y'].restrict(righttimes), 'o', zorder = 2, label = 'right ROI', color = 'r')
    # plt.legend(loc = 'upper right')
    
#%% Check if position is also in inner rectangle 
    
    pos_inL = position.restrict(lefttimes)
    pos_inR = position.restrict(righttimes)
    
    nos_inL = nosepos.restrict(lefttimes)
    nos_inR = nosepos.restrict(righttimes)

    
       
    posvalsL = np.column_stack((pos_inL['x'].values, pos_inL['y'].values))
    posvalsR = np.column_stack((pos_inR['x'].values, pos_inR['y'].values))
        
    inL_in = bbL_inner.contains_points(posvalsL)
    inR_in = bbR_inner.contains_points(posvalsR)
        
    inrectL_idx = np.where(inL_in == True)
    inrectR_idx = np.where(inR_in == True)
    
    inrectL_times = nap.Ts(pos_inL.index.values[inrectL_idx])
    inrectR_times = nap.Ts(pos_inR.index.values[inrectR_idx])
    
    if len(inrectL_times) > 0:
        lefttimes_inner = inrectL_times.find_support(2/30)
    else: lefttimes_inner = nap.IntervalSet(start = 0, end = 0)
        
    if len(inrectR_times) > 0:
        righttimes_inner = inrectR_times.find_support(2/30)
    else: righttimes_inner = nap.IntervalSet(start = 0, end = 0)
    
    
    
    # lefttimes_inner = inrectL_times.find_support(1)
    # righttimes_inner = inrectR_times.find_support(1)
    
#%% IntervalSet of position in between the 2 rectangles

    betweenL = lefttimes.set_diff(lefttimes_inner)
    betweenR = righttimes.set_diff(righttimes_inner)
    
    # plt.figure()
    # plt.title(s)
    # plt.imshow(im, origin = 'lower')
    # ax = sns.scatterplot(data = position.as_dataframe(), x = x[tokeep], y = y[tokeep])
    # ax.add_patch(rectL)
    # ax.add_patch(rectL_inner)
    # # plt.plot(pos_inL['x'].restrict(betweenL), pos_inL['y'].restrict(betweenL), 'o', zorder = 2, label = 'left ROI', color = 'k')
    # plt.legend(loc = 'upper right')
    
#%% Compute distance of nose and body parts for epochs inside rectangle 

    pos_inb_L = pos_inL.restrict(betweenL)
    pos_inb_R = pos_inR.restrict(betweenR)
    
    nos_inb_L = nos_inL.restrict(betweenL)
    nos_inb_R = nos_inR.restrict(betweenR)
    
    d_objL = np.sqrt((pos_inb_L['x'].values - objL_coords[0])**2 + (pos_inb_L['y'].values - objL_coords[1])**2)
    dist_objL = nap.Tsd(t = pos_inb_L.index.values, d = d_objL, time_units = 's')
    # dist_objL = dist_objL.restrict(lefttimes) 
    
    d_objR = np.sqrt((pos_inb_R['x'].values - objR_coords[0])**2 + (pos_inb_R['y'].values - objR_coords[1])**2)
    dist_objR = nap.Tsd(t = pos_inb_R.index.values, d = d_objR, time_units = 's')
    # dist_objR = dist_objR.restrict(righttimes) 
           
    # #NEW CODE: compute distance between nose and object
    d_nose_objL = np.sqrt((nos_inb_L['x'].values - objL_coords[0])**2 + (nos_inb_L['y'].values - objL_coords[1])**2)
    dist_nose_objL = nap.Tsd(t = nos_inb_L.index.values, d = d_nose_objL, time_units = 's')
    # dist_nose_objL = dist_nose_objL.restrict(betweenL) 
    
    d_nose_objR = np.sqrt((nos_inb_R['x'].values - objR_coords[0])**2 + (nos_inb_R['y'].values - objR_coords[1])**2)
    dist_nose_objR = nap.Tsd(t = nos_inb_R.index.values, d = d_nose_objR, time_units = 's')
    # dist_nose_objR = dist_nose_objR.restrict(betweenR) 

        
#%% Second interval: All points where nose is facing the marker

    closer_nose_objL = np.where(dist_nose_objL.values < dist_objL.values)[0]
    closer_nose_objR = np.where(dist_nose_objR.values < dist_objR.values)[0]

    faceL = nap.Ts(dist_nose_objL.index.values[closer_nose_objL])
    faceR = nap.Ts(dist_nose_objR.index.values[closer_nose_objR])
    
    if len(faceL) > 0:
        leftfacing = faceL.find_support(2/30)
    else: leftfacing = nap.IntervalSet(start = 0, end = 0)
        
    if len(faceR) > 0:
        rightfacing = faceR.find_support(2/30)
    else: rightfacing = nap.IntervalSet(start = 0, end = 0)
    

    # leftfacing = faceL.find_support(1)
    # rightfacing = faceR.find_support(1)

       
#%% Merge into 1 intervalSet, compute duration
    
    all_lefttimes = lefttimes_inner.union(leftfacing)
    # all_lefttimes = all_lefttimes.merge_close_intervals(1)
    
    all_righttimes = righttimes_inner.union(rightfacing)
    # all_righttimes = all_righttimes.merge_close_intervals(1)
    
#%% Filter by velocity 

    # bins = np.arange(0,260,5)
    # plt.figure()
    # plt.title(s)
    # plt.hist(speed2.restrict(all_lefttimes),bins, alpha = 0.5)
    # plt.hist(speed2.restrict(all_righttimes),bins, alpha = 0.5)
    
    # immo = speed2.restrict(all_lefttimes.union(all_righttimes)).threshold(23, 'below')
    
    
    # all_lefts = all_lefttimes.set_diff(immo.time_support)
    # all_rights = all_righttimes.set_diff(immo.time_support)
    
    # all_lefts = all_lefts.drop_short_intervals(1/30)
    # all_lefts = all_lefts.drop_long_intervals(3)
    
    # all_rights = all_rights.drop_short_intervals(1/30)
    # all_rights = all_rights.drop_long_intervals(3)
    
      
#%%  
    
    # objL_dur = all_lefts.tot_length()
    # objR_dur = all_rights.tot_length()
    
    objL_dur = all_lefttimes.tot_length()
    objR_dur = all_righttimes.tot_length()
    
    if dispobj == 'right':
        DI = (objR_dur - objL_dur) / (objR_dur + objL_dur)
               
    else: 
        DI = (objL_dur - objR_dur) / (objL_dur + objR_dur)    
       
    
    
#%% 

    plt.figure()
    plt.title(s)
    plt.imshow(im, origin = 'lower')
    ax = sns.scatterplot(data = position.as_dataframe(), x = x[tokeep], y = y[tokeep])
    ax.add_patch(rectL)
    ax.add_patch(rectL_inner)
    ax.add_patch(rectR)
    ax.add_patch(rectR_inner)
    # plt.plot(position['x'].restrict(all_lefts), position['y'].restrict(all_lefts), 'o', zorder = 2, label = 'left ROI', color = 'k')
    # plt.plot(position['x'].restrict(all_rights), position['y'].restrict(all_rights), 'o', zorder = 2, label = 'right ROI', color = 'r')
    plt.plot(position['x'].restrict(all_lefttimes), position['y'].restrict(all_lefttimes), 'o', zorder = 2, label = 'left ROI', color = 'k')
    plt.plot(position['x'].restrict(all_righttimes), position['y'].restrict(all_righttimes), 'o', zorder = 2, label = 'right ROI', color = 'r')
    plt.legend(loc = 'upper right')


#%% 
    
    # #Compute time spent in each radius zone
    # objL_dur = (ep_objL['end'] - ep_objL['start']).sum()
    # objR_dur = (ep_objR['end'] - ep_objR['start']).sum()
    
    #Quantify object displacement
    # if '8' in s.split('_')[2]:
    #     DI = (objR_dur - objL_dur) / (objR_dur + objL_dur)
    #     # DI = objR_dur / (objR_dur + objL_dur)
    #     print('right moved!')
    # else:
    #     DI = (objL_dur - objR_dur) / (objR_dur + objL_dur)
    #     # DI = objL_dur / (objR_dur + objL_dur)
    #     print('left moved!')
    
    # #Sort results by task phase and sex (not relevant for now)
    if taskphase == '1':
        all_lefts_phase1.append(objL_dur)
        all_rights_phase1.append(objR_dur)
        allDI_phase1.append(DI)
        allmice_phase1.append(mousename)
        
        
        #Sort results by genotype (not relevant for now)
        if s[0] == '2':
            genotype_phase1.append('WT')
            # print('WT')
        else: 
            genotype_phase1.append('KO')
            # print('KO')
        
    else:
        all_lefts_phase2.append(objL_dur)
        all_rights_phase2.append(objR_dur)
        allDI_phase2.append(DI)
        allmice_phase2.append(mousename)
                
        if s[0] == '2':
            genotype_phase2.append('WT')
            # print('WT')
        else: 
            genotype_phase2.append('KO')
            # print('KO')
    
    
#%% Compile the results from all sessions into a dataframe

# info1 = pd.DataFrame((zip(allmice_phase1, allsex_phase1, genotype_phase1, all_lefts_phase1, all_rights_phase1, allDI_phase1)), columns =['Name', 'sex', 'genotype', 'left', 'right', 'DI'])

# m_wt = info1[(info1['sex'] == 'M') & (info1['genotype'] == 'WT')]['DI'].values
# m_ko = info1[(info1['sex'] == 'M') & (info1['genotype'] == 'KO')]['DI'].values
# f_ko = info1[(info1['sex'] == 'F') & (info1['genotype'] == 'KO')]['DI'].values

# wt_m = np.array(['WT_male' for x in range(len(m_wt))])
# ko_m = np.array(['KO_male' for x in range(len(m_ko))])
# ko_f = np.array(['KO_female' for x in range(len(f_ko))])

# gtype = np.hstack([wt_m, ko_m, ko_f])
# DIs = np.hstack([m_wt, m_ko, f_ko])

#Look at the output of this variable to understand what the table looks like


# infos_phase1 = pd.DataFrame(data = [DIs, gtype], index = ['DI', 'genotype']).T


### PHASE 2 (Not relevant: but same thing for both task phases)

info2 = pd.DataFrame((zip(allmice_phase2, genotype_phase2, all_lefts_phase2, all_rights_phase2, allDI_phase2)), columns =['Name', 'genotype', 'left', 'right', 'DI'])
info2 = info2.dropna()

m_wt = info2[(info2['genotype'] == 'WT')]['DI'].values
m_ko = info2[(info2['genotype'] == 'KO')]['DI'].values

wt_m = np.array(['WT' for x in range(len(m_wt))])
ko_m = np.array(['KO' for x in range(len(m_ko))])


gtype = np.hstack([wt_m, ko_m])
DIs = np.hstack([m_wt, m_ko])

infos_phase2 = pd.DataFrame(data = [DIs, gtype], index = ['DI', 'genotype']).T


#%% IGNORE

# plt.figure()
# plt.suptitle('Object Displacement')
# plt.subplot(121)
# plt.title('Encoding Phase')
# sns.set_style('white')
# palette = ['royalblue', 'indianred', 'darkslategray']
# ax = sns.violinplot( x = infos_phase1['genotype'], y=infos_phase1['DI'].astype(float) , data = infos_phase1, dodge=False,
#                     palette = palette,cut = 2,
#                     scale="width", inner=None)
# ax.tick_params(bottom=True, left=True)
# xlim = ax.get_xlim()
# ylim = ax.get_ylim()
# for violin in ax.collections:
#     x0, y0, width, height = violin.get_paths()[0].get_extents().bounds
#     violin.set_clip_path(plt.Rectangle((x0, y0), width / 2, height, transform=ax.transData))
# sns.boxplot(x = infos_phase1['genotype'], y=infos_phase1['DI'].astype(float) , data = infos_phase1, saturation=1, showfliers=False,
#             width=0.3, boxprops={'zorder': 3, 'facecolor': 'none'}, ax=ax)
# old_len_collections = len(ax.collections)
# sns.stripplot(x = infos_phase1['genotype'], y=infos_phase1['DI'].astype(float) , data = infos_phase1, color = 'k', dodge=False, ax=ax)

# for dots in ax.collections[old_len_collections:]:
#     dots.set_offsets(dots.get_offsets())
# ax.set_xlim(xlim)
# ax.set_ylim(ylim)
# plt.ylabel('Discrimination Index')
# plt.axhline(0, linestyle = '--', color ='silver')
# ax.set_box_aspect(1)

# plt.subplot(122)
# plt.title('Recall Phase')
# sns.set_style('white')
# palette = ['royalblue', 'indianred', 'darkslategray']
# ax = sns.violinplot( x = infos_phase2['genotype'], y=infos_phase2['DI'].astype(float) , data = infos_phase2, dodge=False,
#                     palette = palette,cut = 2,
#                     scale="width", inner=None)
# ax.tick_params(bottom=True, left=True)
# xlim = ax.get_xlim()
# ylim = ax.get_ylim()
# for violin in ax.collections:
#     x0, y0, width, height = violin.get_paths()[0].get_extents().bounds
#     violin.set_clip_path(plt.Rectangle((x0, y0), width / 2, height, transform=ax.transData))
# sns.boxplot(x = infos_phase2['genotype'], y=infos_phase2['DI'].astype(float) , data = infos_phase2, saturation=1, showfliers=False,
#             width=0.3, boxprops={'zorder': 3, 'facecolor': 'none'}, ax=ax)
# old_len_collections = len(ax.collections)
# sns.stripplot(x = infos_phase2['genotype'], y=infos_phase2['DI'].astype(float) , data = infos_phase2, color = 'k', dodge=False, ax=ax)

# for dots in ax.collections[old_len_collections:]:
#     dots.set_offsets(dots.get_offsets())
# ax.set_xlim(xlim)
# ax.set_ylim(ylim)
# plt.ylabel('Discrimination Index')
# plt.axhline(0, linestyle = '--', color ='silver')
# ax.set_box_aspect(1)


#%% Plot all results (needs modification, will modify once everything is ready)

# label = ['WT male']
# x = np.arange(len(label))  # the label locations
# width = 0.35  # the width of the bars

# fig, ax = plt.subplots()
# rects1 = ax.bar(x - width/2, infos_phase1[infos_phase1['genotype'] == 'WT_male']['DI'].mean(), width, label='Encoding Phase')
# rects2 = ax.bar(x + width/2, infos_phase2[infos_phase2['genotype'] == 'WT_male']['DI'].mean(), width, label='Recall Phase')
# pval = np.vstack([(infos_phase1[infos_phase1['genotype'] == 'WT_male']['DI'].values), (infos_phase2[infos_phase2['genotype'] == 'WT_male']['DI'].values)])
# x2 = [x-width/2, x+width/2]
# plt.plot(x2, np.vstack(pval), 'o-', color = 'k')
# Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_ylabel('Discrimination index')
# ax.set_title('WT male')
# ax.set_xticks(x)
# ax.legend(loc = 'upper right')
# ax.set_box_aspect(1)
# fig.tight_layout()

# fig, ax = plt.subplots()
# rects1 = ax.bar(x - width/2, infos_phase1[infos_phase1['genotype'] == 'KO_male']['DI'].mean(), width, label='Encoding Phase')
# rects2 = ax.bar(x + width/2, infos_phase2[infos_phase2['genotype'] == 'KO_male']['DI'].mean(), width, label='Recall Phase')
# pval = np.vstack([(infos_phase1[infos_phase1['genotype'] == 'KO_male']['DI'].values), (infos_phase2[infos_phase2['genotype'] == 'KO_male']['DI'].values)])
# x2 = [x-width/2, x+width/2]
# plt.plot(x2, np.vstack(pval), 'o-', color = 'k')
# # Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_ylabel('Discrimination index')
# ax.set_title('KO male')
# ax.set_xticks(x)
# ax.legend(loc = 'upper right')
# ax.set_box_aspect(1)
# fig.tight_layout()

# fig, ax = plt.subplots()
# rects1 = ax.bar(x - width/2, infos_phase1[infos_phase1['genotype'] == 'KO_female']['DI'].mean(), width, label='Encoding Phase')
# rects2 = ax.bar(x + width/2, infos_phase2[infos_phase2['genotype'] == 'KO_female']['DI'].mean(), width, label='Recall Phase')
# pval = np.vstack([(infos_phase1[infos_phase1['genotype'] == 'KO_female']['DI'].values), (infos_phase2[infos_phase2['genotype'] == 'KO_female']['DI'].values)])
# x2 = [x-width/2, x+width/2]
# plt.plot(x2, np.vstack(pval), 'o-', color = 'k')
# # Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_ylabel('Discrimination index')
# ax.set_title('KO female')
# ax.set_xticks(x)
# ax.legend(loc = 'upper right')
# ax.set_box_aspect(1)
# fig.tight_layout()

#%% 

plt.figure()
plt.boxplot(infos_phase2['DI'][infos_phase2['genotype'] == 'WT'], positions = [0], showfliers= False, patch_artist=True,boxprops=dict(facecolor='royalblue', color='royalblue'),
            capprops=dict(color='royalblue'),
            whiskerprops=dict(color='royalblue'),
            medianprops=dict(color='white', linewidth = 2))
plt.boxplot(infos_phase2['DI'][infos_phase2['genotype'] == 'KO'], positions = [0.3], showfliers= False, patch_artist=True,boxprops=dict(facecolor='indianred', color='indianred'),
            capprops=dict(color='indianred'),
            whiskerprops=dict(color='indianred'),
            medianprops=dict(color='white', linewidth = 2))

x1 = np.random.normal(0, 0.01, size=len(infos_phase2['DI'][infos_phase2['genotype'] == 'WT']))
# x2 = np.random.normal(0, 0.01, size=len(info2['mean'][info2['Genotype'] == 'WT'][info2['Sex'] == 'F']))
x3 = np.random.normal(0.3, 0.01, size=len(infos_phase2['DI'][infos_phase2['genotype'] == 'KO']))
# x4 = np.random.normal(0.3, 0.01, size=len(info2['mean'][info2['Genotype'] == 'KO'][info2['Sex'] == 'F']))
                      
plt.plot(x1, infos_phase2['DI'][infos_phase2['genotype'] == 'WT'], '.', color = 'k', fillstyle = 'none', markersize = 8, zorder =3)
# plt.plot(x2, info2['mean'][info2['Genotype'] == 'WT'][info2['Sex'] == 'F'], '.', color = 'k', fillstyle = 'full', markersize = 8, zorder =3, label = 'female')
plt.plot(x3, infos_phase2['DI'][infos_phase2['genotype'] == 'KO'], '.', color = 'k', fillstyle = 'none', markersize = 8, zorder =3)
# plt.plot(x4, info2['mean'][info2['Genotype'] == 'KO'][info2['Sex'] == 'F'], '.', color = 'k', fillstyle = 'full', markersize = 8, zorder =3)

plt.ylabel('Discrimination Index')
plt.axhline(0, linestyle = '--',  color = 'silver')
# plt.legend(loc = 'upper left')
plt.xticks([0, 0.3],['WT', 'KO'])
plt.ylim([-1.1, 1.1])
plt.gca().set_box_aspect(1)

z_wt, p_wt = wilcoxon(np.array(infos_phase2['DI'][infos_phase2['genotype'] == 'WT'])-0)
z_ko, p_ko = wilcoxon(np.array(infos_phase2['DI'][infos_phase2['genotype'] == 'KO'])-0)

#%% 

# plt.scatter(man_m, auto_m, color = 'orange', zorder = 3, label = 'Male') 
# plt.scatter(man_f, auto_f, color = 'purple', zorder = 3, label = 'Female') 
# plt.gca().axline(
#     (min(min(man_m),min(auto_m),min(man_f),min(auto_f)),(min(min(man_m),min(auto_m),min(man_f), min(auto_f)))), 
#      slope = 1, color = 'silver', linestyle = '--')
# plt.xlabel('DI (manual)')
# plt.ylabel('DI (automated)')
# plt.axis('square')
# plt.legend(loc = 'upper left')

#%% 

# DI_wt_f = [0.4857138743449835, 0.6306066533777056, 0.3333335165200056, -0.19093813622513545, 0.17018264807318878, 0.3571413263075806,
#            0.2895998647291097, 0.7493185367286013, 0.8306189221459894, 0.46357615628311605, 0.3018866485940092, 0.40318917194118803,
#            -1.0, 0.03966617171169174]

# DI_ko_f = [0.023493434497851955, -0.18165141460196862, 0.756878698208257, 0.5736841510856754, 0.2000001143084898, -0.9656312009245304,
#            0.23684224980850346, 0.49798864220203176, -0.19615915831560482, 0.6039603916380796, 0.397101382745879, 0.17492703828439232,
#            0.45736440698986236, -0.027522858392479893]

# DI_wt_m = [-0.8810808916250508,	-0.06097567007275186, 0.36338027442064613, 0.2000002177450981, 1.0, 0.07142777764487694,	1.0,
#            1.0, 	0.6301076625186209, 0.3635428293325658, 	0.2148998028998942, 	0.10580737707116199, 1.0]

# DI_ko_m = [-0.31147510369285697, -0.6020234245148113, 0.18162609743784447, -0.36521708072714026, 	0.23926367937687734,
#            -0.45348840651829825, 0.5051544367736622, -0.34265704719693574, -1.0, 0.06117054815517352, 0.5345581512372484,
# 	     -0.008498614869189744,	0.6376375488385441,	0.7516231802999758]

# times = np.hstack([DI_wt_f, DI_ko_f, DI_wt_m, DI_ko_m])

# g1 = np.array(['WT' for x in range(len(DI_wt_f))])
# g2 = np.array(['KO' for x in range(len(DI_ko_f))])
# g3 = np.array(['WT' for x in range(len(DI_wt_m))])
# g4 = np.array(['KO' for x in range(len(DI_ko_m))])
# genotype = np.hstack([g1, g2, g3, g4])

# mf1 = np.array(['F' for x in range(len(DI_wt_f))])
# mf2 = np.array(['F' for x in range(len(DI_ko_f))])
# mf3 = np.array(['M' for x in range(len(DI_wt_m))])
# mf4 = np.array(['M' for x in range(len(DI_ko_m))])
# sex = np.hstack([mf1, mf2, mf3, mf4])

# df = pd.DataFrame(data = [times, genotype, sex], index = ['DI', 'genotype', 'sex']).T
# df['DI']= pd.to_numeric(df['DI'])

# model = ols('times ~ C(genotype) * C(sex)', data=df).fit()
# anova_table = sm.stats.anova_lm(model, typ=2)

# tukey_genotype = pairwise_tukeyhsd(endog=df['DI'], groups=df['genotype'], alpha=0.05)
# tukey_sex = pairwise_tukeyhsd(endog=df['DI'], groups=df['sex'], alpha=0.05)
