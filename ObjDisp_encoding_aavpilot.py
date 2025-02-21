#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 10:42:59 2025

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
from scipy.stats import mannwhitneyu, wilcoxon, f_oneway, kruskal
import scikit_posthocs as sp
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

#%% 

warnings.filterwarnings("ignore")

##Add path to data folder
data_directory = '/media/DataDhruv/Recordings/Christianson/aavpilot/encoding'

##File with list of all sessions
datasets = np.genfromtxt(os.path.normpath(os.path.join(data_directory,'dataset.list')), delimiter = '\n', dtype = str, comments = '#')

## Variables to store final results 
allmice = []

all_lefts_wt = []
all_lefts_ko = []

all_rights_wt = []
all_rights_ko = []


##Loop through each session from the list

for s in datasets:
    print(s)
    mousename = s.split('_')[0] #The first character is the mouse name
    sex = s.split('_')[1] #Second character is sex (ignore)
    taskphase = s.split('_')[-1] #Last character is encoding or recall phase (Modify these portions as needed)
    
    
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
    # trackedpos = trackedpos[0:30*180]
    # likelihoods = likelihoods[:,0:30*180]
    
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
    
    #60 frames per second
    fs = 60
    
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

    # if mousename == '642':
    
    if sex == 'M':
        
        rectL_inner = patches.Rectangle((x_objL - 60, y_objL - 45), 130, 230, linewidth=1, edgecolor='g', facecolor='none')
        rectL = patches.Rectangle((x_objL - 101.15, y_objL - 86.15), 212.3, 312.3, linewidth=1, edgecolor='b', facecolor='none')
        
        rectR_inner = patches.Rectangle((x_objR - 85, y_objR - 45), 130, 230, linewidth=1, edgecolor='g', facecolor='none')
        rectR = patches.Rectangle((x_objR - 126.15, y_objR - 86.15), 212.3, 312.3, linewidth=1, edgecolor='b', facecolor='none')


      
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
    
    allmice.append(mousename)

    if s[0] == '2':
                
        all_lefts_wt.append(objL_dur)
        all_rights_wt.append(objR_dur)
        
    else: 
        
        all_lefts_ko.append(objL_dur)
        all_rights_ko.append(objR_dur)
        
   
    
    
#%% Compile the results from all sessions into a dataframe

wt = np.array(['WT' for x in range(len(all_lefts_wt))])
ko = np.array(['KO' for x in range(len(all_lefts_ko))])

genotype = np.hstack([wt, ko])

leftdurs = []
leftdurs.extend(all_lefts_wt)
leftdurs.extend(all_lefts_ko)

rightdurs = []
rightdurs.extend(all_rights_wt)
rightdurs.extend(all_rights_ko)

durdf = pd.DataFrame(data = [leftdurs, rightdurs, genotype], index = ['left','right', 'genotype']).T


#%% Plot all results 

label = ['WT male']
x = np.arange(len(label))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, durdf[durdf['genotype'] == 'WT']['left'].mean(), width, label='Left', color = 'cadetblue')
rects2 = ax.bar(x + width/2, durdf[durdf['genotype'] == 'WT']['right'].mean(), width, label='Right', color = 'violet')
pval = np.vstack([(durdf[durdf['genotype'] == 'WT']['left'].values, durdf[durdf['genotype'] == 'WT']['right'].values)])
x2 = [x-width/2, x+width/2]
plt.plot(x2, np.vstack(pval), 'o-', fillstyle = 'none',  color = 'k')
plt.errorbar(x2[0], np.mean(pval[0]), yerr = scipy.stats.sem(pval[0]) , fmt = 'o', color="k", linewidth = 2, capsize = 6)
plt.errorbar(x2[1], np.mean(pval[1]), yerr = scipy.stats.sem(pval[1]) , fmt = 'o', color="k", linewidth = 2, capsize = 6)
plt.ylim([0, 130])

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Interaction time (s)')
ax.set_title('WT male')
ax.set_xticks(x)
ax.legend(loc = 'upper right')
ax.set_box_aspect(1)
fig.tight_layout()

z, p = wilcoxon(all_lefts_wt, all_rights_wt)

#%% 

label = ['KO male']
x = np.arange(len(label))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, durdf[durdf['genotype'] == 'KO']['left'].mean(), width, label='Left', color = 'cadetblue')
rects2 = ax.bar(x + width/2, durdf[durdf['genotype'] == 'KO']['right'].mean(), width, label='Right', color = 'violet')
pval = np.vstack([(durdf[durdf['genotype'] == 'KO']['left'].values, durdf[durdf['genotype'] == 'KO']['right'].values)])
x2 = [x-width/2, x+width/2]
plt.plot(x2, np.vstack(pval), 'o-', fillstyle = 'none',  color = 'k')
plt.errorbar(x2[0], np.mean(pval[0]), yerr = scipy.stats.sem(pval[0]) , fmt = 'o', color="k", linewidth = 2, capsize = 6)
plt.errorbar(x2[1], np.mean(pval[1]), yerr = scipy.stats.sem(pval[1]) , fmt = 'o', color="k", linewidth = 2, capsize = 6)
plt.ylim([0, 130])

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Interaction time (s)')
ax.set_title('KO male')
ax.set_xticks(x)
ax.legend(loc = 'upper right')
ax.set_box_aspect(1)
fig.tight_layout()

z, p = wilcoxon(all_lefts_ko, all_rights_ko)

#%% 

# left_wt_f = [90.533419331, 39.800027994, 67.600047996, 72.166719663, 46.500034001, 16.800021999, 54.966726669, 44.300071998, 
#               71.06672267, 49.533408328, 12.233349333, 63.833407335, 76.366750669]

# right_wt_f = [73.900075991, 43.533369333, 40.100025999, 71.000048002, 67.200050001, 12.766687668, 100.033412338, 70.400086005,
#               60.200061002, 43.166729665, 15.566680666, 38.700039999, 57.833405334]

# left_ko_f = [117.066774669, 103.36673167, 83.700071001, 17.300021001, 61.466711668, 132.633471345, 98.966743666, 47.300040998,
#               25.100017999, 57.16673067, 69.266727666, 94.533402334, 81.966748671, 76.500088999]

# right_ko_f = [105.033398334, 117.100074003,84.200070003,13.800020999,40.900027001, 74.200062,72.966723664,119.966756657,
#               20.533353334, 40.43338033, 108.166762668, 78.166717665, 94.733425338, 96.600090992]

left_wt_m = [32.66670967, 63.783387337,	19.100034]
right_wt_m = [31.700043001, 38.733373331, 21.800035]

left_ko_m = [125.283, 36.783396342,	41.900079998]
right_ko_m = [73.116752665, 	24.083377333, 44.916727665]


# kruskal(left_wt_f, right_wt_f, left_ko_f, right_ko_f, left_wt_m, right_wt_m, left_ko_m, right_ko_m)
    
# data = [left_wt_f, right_wt_f, left_ko_f, right_ko_f, left_wt_m, right_wt_m, left_ko_m, right_ko_m]

# sp.posthoc_dunn(data, p_adjust = 'holm')

# tot_wt_m = np.add(left_wt_m, right_wt_m)
# tot_wt_f = np.add(left_wt_f, right_wt_f)
# tot_ko_m = np.add(left_ko_m, right_ko_m)
# tot_ko_f = np.add(left_ko_f, right_ko_f)


# kruskal(tot_wt_m, tot_wt_f, tot_ko_m, tot_ko_f)
# data = [tot_wt_m, tot_wt_f, tot_ko_m, tot_ko_f]

# sp.posthoc_dunn(data, p_adjust = 'holm')

#%% 

times = np.hstack([left_wt_m, right_wt_m, left_ko_m, right_ko_m])

# s1 = np.array(['L' for x in range(len(left_wt_f))])
# s2 = np.array(['R' for x in range(len(right_wt_f))])
# s3 = np.array(['L' for x in range(len(left_ko_f))])
# s4 = np.array(['R' for x in range(len(right_ko_f))])
s5 = np.array(['L' for x in range(len(left_wt_m))])
s6 = np.array(['R' for x in range(len(right_wt_m))])
s7 = np.array(['L' for x in range(len(left_ko_m))])
s8 = np.array(['R' for x in range(len(right_ko_m))])
side = np.hstack([s5, s6, s7, s8])

# g1 = np.array(['WT' for x in range(len(left_wt_f))])
# g2 = np.array(['WT' for x in range(len(right_wt_f))])
# g3 = np.array(['KO' for x in range(len(left_ko_f))])
# g4 = np.array(['KO' for x in range(len(right_ko_f))])
g5 = np.array(['WT' for x in range(len(left_wt_m))])
g6 = np.array(['WT' for x in range(len(right_wt_m))])
g7 = np.array(['KO' for x in range(len(left_ko_m))])
g8 = np.array(['KO' for x in range(len(right_ko_m))])
genotype = np.hstack([g5, g6, g7, g8])

# mf1 = np.array(['F' for x in range(len(left_wt_f))])
# mf2 = np.array(['F' for x in range(len(right_wt_f))])
# mf3 = np.array(['F' for x in range(len(left_ko_f))])
# mf4 = np.array(['F' for x in range(len(right_ko_f))])
# mf5 = np.array(['M' for x in range(len(left_wt_m))])
# mf6 = np.array(['M' for x in range(len(right_wt_m))])
# mf7 = np.array(['M' for x in range(len(left_ko_m))])
# mf8 = np.array(['M' for x in range(len(right_ko_m))])
# sex = np.hstack([mf5, mf6, mf7, mf8])

df = pd.DataFrame(data = [times, side, genotype], index = ['times','side', 'genotype']).T
df['times']= pd.to_numeric(df['times'])

model = ols('times ~ C(side) * C(genotype)', data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)

# tukey_genotype = pairwise_tukeyhsd(endog=df['times'], groups=df['genotype'], alpha=0.05)
# tukey_sex = pairwise_tukeyhsd(endog=df['times'], groups=df['sex'], alpha=0.05)

# groups_side = [df['times'][df['side'] == level] for level in df['side'].unique()]
# groups_genotype = [df['times'][df['genotype'] == level] for level in df['genotype'].unique()]
# groups_sex = [df['times'][df['sex'] == level] for level in df['sex'].unique()]

# kruskal_result_side = kruskal(*groups_side)
# kruskal_result_genotype = kruskal(*groups_genotype)
# kruskal_result_sex = kruskal(*groups_sex)

# dunn_side = sp.posthoc_dunn(df, val_col='times', group_col='side', p_adjust='holm')
# dunn_genotype = sp.posthoc_dunn(df, val_col='times', group_col='genotype', p_adjust='holm')
# dunn_sex = sp.posthoc_dunn(df, val_col='times', group_col='sex', p_adjust='holm')

#%% 