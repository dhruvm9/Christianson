#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 12:39:59 2024

@author: dhruv
"""

#%% Import all necessary libraries

import pynapple as nap

import numpy as np
import pandas as pd
import scipy.io
import os, sys
import seaborn as sns
import matplotlib.pyplot as plt
from pylab import *
from scipy.stats import mannwhitneyu, wilcoxon

#%% 

##Add path to data folder
data_directory = '/media/DataDhruv/Recordings/Christianson/ObjDisp-240802'

##File with list of all sessions
datasets = np.genfromtxt(os.path.join(data_directory,'dataset.list'), delimiter = '\n', dtype = str, comments = '#')

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

for s in datasets:
    print(s)
    mousename = s.split('_')[0] #The first character is the mouse name
    sex = s.split('_')[1] #Second character is sex (ignore)
    taskphase = s.split('_')[-1] #Last character is encoding or recall phase (Modify these portions as needed)
    
    #Read the tracking data
    tracking_data =  pd.read_hdf(data_directory + '/' + s + '.h5', mode = 'r')

    #Columns containing x and y coordinates (Modify as needed)
    position_cols = [0,1,3,4,6,7,9,10]
    
    #Select only those columns with tracked data
    trackedpos = tracking_data.iloc[:, position_cols]
    
    #A separate table for likelihood values
    likelihoods = np.vstack([tracking_data.iloc[:,2], tracking_data.iloc[:,5]])
    
    # If recall phase, analyze only the first 3 min (180s) of task. Each video has 30 frames per second.
    if taskphase == '2':
        trackedpos = trackedpos[0:30*180]
        likelihoods = likelihoods[:,0:30*180]
    
    #Cutoff value of likelihood for reliable tracking
    pcutoff = 0.6

    #Keep only those frames where likelihood for mouse body parts is above the cutoff
    tokeep = []
    for i in range(shape(likelihoods)[1]):
        if (likelihoods[0][i] > pcutoff) and (likelihoods[1][i] > pcutoff):
            tokeep.append(i)
    
    #X- and Y-column indices for ears
    x_ear = [0,2]
    y_ear = [1,3]

    #X- and Y- coordinates for ears
    all_x_ear = trackedpos.iloc[:, x_ear]
    all_y_ear = trackedpos.iloc[:, y_ear]
    
    #X- and Y- coordinates for left object
    all_x_objL = trackedpos.iloc[:,4]
    all_y_objL = trackedpos.iloc[:,5]
    
    #X- and Y- coordinates for right object
    all_x_objR = trackedpos.iloc[:,6]
    all_y_objR = trackedpos.iloc[:,7]
    
    #Compute mean position of left object by averaging its position across time
    x_objL = all_x_objL.mean()
    y_objL = all_y_objL.mean()
    
    #Compute mean position of right object by averaging its position across time
    x_objR = all_x_objR.mean()
    y_objR = all_y_objR.mean()
    
    #Compute centroid of ears (proxy for head position) -- Add the nose to this!
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
    
    #Create a variable for the coordinates of the left and right object
    objL_coords = np.hstack([x_objL, y_objL])
    objR_coords = np.hstack([x_objR, y_objR])

#%% Plot the position of the tracked points

    # plt.figure()
    # plt.plot(position['x'], position['y'],'.')
    # plt.plot(objL_coords[0], objL_coords[1], 'o', color = 'k')
    # plt.plot(objR_coords[0], objR_coords[1], 'o', color = 'r')

#%% Plot the circle around objects 

    #Radius of circle
    roi = 100
    
    # circle1 = plt.Circle((objL_coords[0], objL_coords[1]), roi, color='k', fill = False)
    # circle2 = plt.Circle((objR_coords[0], objR_coords[1]), roi, color='r', fill = False)
    # ax = sns.scatterplot(data = position.as_dataframe(), x = x[tokeep], y = y[tokeep])
    # ax.add_patch(circle1)
    # ax.add_patch(circle2)

#%% Compute distance of objects from mouse head position (add nose modification)

    d_objL = np.sqrt((x[tokeep] - objL_coords[0])**2 + (y[tokeep] - objL_coords[1])**2)
    dist_objL = nap.Tsd(t = timestamps[tokeep], d = d_objL, time_units = 's')
    
    d_objR = np.sqrt((x[tokeep] - objR_coords[0])**2 + (y[tokeep] - objR_coords[1])**2)
    dist_objR = nap.Tsd(t = timestamps[tokeep], d = d_objR, time_units = 's')

#%% Check whether the position of the animal is within the radius

    within_objL = dist_objL.threshold(roi, 'below')
    ep_objL = within_objL.time_support
    
    within_objR = dist_objR.threshold(roi, 'below')
    ep_objR = within_objR.time_support


#%% Plot the tracked position, colour coded by radius zones

    plt.figure()
    plt.title(s)
    plt.plot(x[tokeep], y[tokeep], 'o')
    plt.plot(position['x'].restrict(ep_objL), position['y'].restrict(ep_objL), 'o', zorder = 2, label = 'left ROI', color = 'k')
    plt.plot(position['x'].restrict(ep_objR), position['y'].restrict(ep_objR), 'o', zorder = 2, label = 'right ROI', color = 'r')
    plt.legend(loc = 'upper right')

#%% 
    
    #Compute time spent in each radius zone
    objL_dur = (ep_objL['end'] - ep_objL['start']).sum()
    objR_dur = (ep_objR['end'] - ep_objR['start']).sum()
    
    #Quantify object displacement
    if '8' in s.split('_')[2]:
        DI = (objR_dur - objL_dur) / (objR_dur + objL_dur)
        # DI = objR_dur / (objR_dur + objL_dur)
        print('right moved!')
    else:
        DI = (objL_dur - objR_dur) / (objR_dur + objL_dur)
        # DI = objL_dur / (objR_dur + objL_dur)
        print('left moved!')
    
    #Sort results by task phase and sex (not relevant for now)
    if taskphase == '1':
        all_lefts_phase1.append(objL_dur)
        all_rights_phase1.append(objR_dur)
        allDI_phase1.append(DI)
        allmice_phase1.append(mousename)
        allsex_phase1.append(sex)
        
        #Sort results by genotype (not relevant for now)
        if s[0] == '2':
            genotype_phase1.append('WT')
        else: genotype_phase1.append('KO')
        
    else:
        all_lefts_phase2.append(objL_dur)
        all_rights_phase2.append(objR_dur)
        allDI_phase2.append(DI)
        allmice_phase2.append(mousename)
        allsex_phase2.append(sex)
        
        if s[0] == '2':
            genotype_phase2.append('WT')
        else: genotype_phase2.append('KO')
    
    
#%% Compile the results from all sessions into a dataframe

info1 = pd.DataFrame((zip(allmice_phase1, allsex_phase1, genotype_phase1, all_lefts_phase1, all_rights_phase1, allDI_phase1)), columns =['Name', 'sex', 'genotype', 'left', 'right', 'DI'])

m_wt = info1[(info1['sex'] == 'M') & (info1['genotype'] == 'WT')]['DI'].values
m_ko = info1[(info1['sex'] == 'M') & (info1['genotype'] == 'KO')]['DI'].values
f_ko = info1[(info1['sex'] == 'F') & (info1['genotype'] == 'KO')]['DI'].values

wt_m = np.array(['WT_male' for x in range(len(m_wt))])
ko_m = np.array(['KO_male' for x in range(len(m_ko))])
ko_f = np.array(['KO_female' for x in range(len(f_ko))])

gtype = np.hstack([wt_m, ko_m, ko_f])
DIs = np.hstack([m_wt, m_ko, f_ko])

#Look at the output of this variable to understand what the table looks like


infos_phase1 = pd.DataFrame(data = [DIs, gtype], index = ['DI', 'genotype']).T


### PHASE 2 (Not relevant: but same thing for both task phases)

info2 = pd.DataFrame((zip(allmice_phase2, allsex_phase2, genotype_phase2, all_lefts_phase2, all_rights_phase2, allDI_phase2)), columns =['Name', 'sex', 'genotype', 'left', 'right', 'DI'])

m_wt = info2[(info2['sex'] == 'M') & (info2['genotype'] == 'WT')]['DI'].values
m_ko = info2[(info2['sex'] == 'M') & (info2['genotype'] == 'KO')]['DI'].values
f_ko = info2[(info2['sex'] == 'F') & (info2['genotype'] == 'KO')]['DI'].values

wt_m = np.array(['WT_male' for x in range(len(m_wt))])
ko_m = np.array(['KO_male' for x in range(len(m_ko))])
ko_f = np.array(['KO_female' for x in range(len(f_ko))])

gtype = np.hstack([wt_m, ko_m, ko_f])
DIs = np.hstack([m_wt, m_ko, f_ko])

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

label = ['WT male']
x = np.arange(len(label))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, infos_phase1[infos_phase1['genotype'] == 'WT_male']['DI'].mean(), width, label='Encoding Phase')
rects2 = ax.bar(x + width/2, infos_phase2[infos_phase2['genotype'] == 'WT_male']['DI'].mean(), width, label='Recall Phase')
pval = np.vstack([(infos_phase1[infos_phase1['genotype'] == 'WT_male']['DI'].values), (infos_phase2[infos_phase2['genotype'] == 'WT_male']['DI'].values)])
x2 = [x-width/2, x+width/2]
plt.plot(x2, np.vstack(pval), 'o-', color = 'k')
# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Discrimination index')
ax.set_title('WT male')
ax.set_xticks(x)
ax.legend(loc = 'upper right')
ax.set_box_aspect(1)
fig.tight_layout()

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, infos_phase1[infos_phase1['genotype'] == 'KO_male']['DI'].mean(), width, label='Encoding Phase')
rects2 = ax.bar(x + width/2, infos_phase2[infos_phase2['genotype'] == 'KO_male']['DI'].mean(), width, label='Recall Phase')
pval = np.vstack([(infos_phase1[infos_phase1['genotype'] == 'KO_male']['DI'].values), (infos_phase2[infos_phase2['genotype'] == 'KO_male']['DI'].values)])
x2 = [x-width/2, x+width/2]
plt.plot(x2, np.vstack(pval), 'o-', color = 'k')
# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Discrimination index')
ax.set_title('KO male')
ax.set_xticks(x)
ax.legend(loc = 'upper right')
ax.set_box_aspect(1)
fig.tight_layout()

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, infos_phase1[infos_phase1['genotype'] == 'KO_female']['DI'].mean(), width, label='Encoding Phase')
rects2 = ax.bar(x + width/2, infos_phase2[infos_phase2['genotype'] == 'KO_female']['DI'].mean(), width, label='Recall Phase')
pval = np.vstack([(infos_phase1[infos_phase1['genotype'] == 'KO_female']['DI'].values), (infos_phase2[infos_phase2['genotype'] == 'KO_female']['DI'].values)])
x2 = [x-width/2, x+width/2]
plt.plot(x2, np.vstack(pval), 'o-', color = 'k')
# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Discrimination index')
ax.set_title('KO female')
ax.set_xticks(x)
ax.legend(loc = 'upper right')
ax.set_box_aspect(1)
fig.tight_layout()

