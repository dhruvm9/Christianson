#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 09:56:44 2024

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
from pylab import *
from scipy.stats import mannwhitneyu, wilcoxon

#%% 

##Add path to data folder
data_directory = '/media/DataDhruv/Recordings/Christianson/H3_cohort3'

##File with list of all sessions
datasets = np.genfromtxt(os.path.normpath(os.path.join(data_directory,'dataset.list')), delimiter = '\n', dtype = str, comments = '#')

allmice = []
genotype = []
alldist = []

for s in datasets:
    print(s)
    mousename = s.split('_')[0] #The first character is the mouse name
    sex = s.split('_')[1] #Second character is sex 
    
    im = plt.imread(data_directory + '/' + mousename + '.png')
    
    #Read the tracking data
    #reads an HDF5 file and stores its contents into a Pandas DataFrame named tracking_data.
    #r is reading only mode
    tracking_data =  pd.read_hdf(data_directory + '/' + s + '.h5', mode = 'r')
    
    #Columns containing x and y coordinates (Modify as needed)
    position_cols = [0,1,3,4,6,7]
    
    #Select only those columns with tracked data
    #It selects all rows (because of :) but only the columns specified by position_cols from the tracking_data DataFrame.
    trackedpos = tracking_data.iloc[:, position_cols]
    
    #A separate table for likelihood values - likilihood of what? of ears being in range or any body part?
    #using NumPy to vertically stack (combine) two columns from a Pandas DataFrame called tracking_data into a single NumPy array. 
    likelihoods = np.vstack([tracking_data.iloc[:,2], tracking_data.iloc[:,5], tracking_data.iloc[:,8]])
    
    #Cutoff value of likelihood for reliable tracking
    pcutoff = 0.6
    
    #Keep only those frames where likelihood for mouse body parts is above the cutoff
    #for loop generates a sequence of indices from 0 to number_of_columns - 1
    tokeep = []
    for i in range(shape(likelihoods)[1]):
        if (likelihoods[0][i] > pcutoff) and (likelihoods[1][i] > pcutoff)and (likelihoods[2][i]):
            tokeep.append(i)
            
    #X- and Y-column indices for bodyparts (ear + nose)
    x_coords = [0,2,4]
    y_coords = [1,3,5]
    
    #X- and Y- coordinates for ears
    all_x_coords = trackedpos.iloc[:, x_coords]
    all_y_coords = trackedpos.iloc[:, y_coords]
    
    #Compute centroid of ears (proxy for head position)
    #x_cent and y_cent represent the average x and y coordinates, respectively, 
    #calculated by summing all x and y coordinates for each point across measurements and dividing by the total number of measurements.
    #we compare distance of nose to object to centroid of ears to object, and see which one is shorter
    x_cent = all_x_coords.sum(axis=1)/all_x_coords.iloc[0,:].shape[0]
    y_cent = all_y_coords.sum(axis=1)/all_y_coords.iloc[0,:].shape[0]
    
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
    
#%% Plot tracking 

    # plt.figure()
    # plt.title(mousename)
    # plt.imshow(im, origin = 'lower')
    # ax = sns.scatterplot(data = position.as_dataframe(), x = position.as_dataframe()['x'], y = position.as_dataframe()['y'])
    
    
#%% Compute distance between positions of consecutive frames 

    distance = np.sqrt(np.power(np.diff(position['x']), 2) + np.power(np.diff(position['y']), 2)) 
    tot_dist = (sum(distance) * 60) / 470 ##approx conversion to cm
    
#%% Sort results by genotype
        
    if s[0] == '2':
        allmice.append(mousename)
        genotype.append('WT')
        alldist.append(tot_dist)
        
    else: 
        allmice.append(mousename)
        genotype.append('KO')
        alldist.append(tot_dist)
        
#%% Compile the results from all sessions into a dataframe

info = pd.DataFrame((zip(allmice, genotype, alldist)), columns =['name', 'genotype', 'dist'])

#%% 

# plt.figure()
# plt.boxplot(info['dist'][info['genotype'] == 'WT'], positions = [0], showfliers= False, patch_artist=True,boxprops=dict(facecolor='royalblue', color='royalblue'),
#             capprops=dict(color='royalblue'),
#             whiskerprops=dict(color='royalblue'),
#             medianprops=dict(color='white', linewidth = 2))
# plt.boxplot(info['dist'][info['genotype'] == 'KO'], positions = [0.3], showfliers= False, patch_artist=True,boxprops=dict(facecolor='indianred', color='indianred'),
#             capprops=dict(color='indianred'),
#             whiskerprops=dict(color='indianred'),
#             medianprops=dict(color='white', linewidth = 2))

# x1 = np.random.normal(0, 0.01, size=len(info['dist'][info['genotype'] == 'WT']))
# x2 = np.random.normal(0.3, 0.01, size=len(info['dist'][info['genotype'] == 'KO']))
                      
# plt.plot(x1, info['dist'][info['genotype'] == 'WT'], '.', color = 'k', fillstyle = 'none', markersize = 8, zorder =3)
# plt.plot(x2, info['dist'][info['genotype'] == 'KO'], '.', color = 'k', fillstyle = 'none', markersize = 8, zorder =3)

# plt.ylabel('Distance travelled (cm)')
# plt.xticks([0, 0.3],['WT', 'KO'])
# plt.gca().set_box_aspect(1)

#%%

label = ['Genotype']
x = np.arange(len(label))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, info[info['genotype'] == 'WT']['dist'].mean(), width, label = 'WT', color = 'royalblue')
rects2 = ax.bar(x + width/2, info[info['genotype'] == 'KO']['dist'].mean(), width, label = 'KO', color = 'indianred')
pval = np.vstack([(info[info['genotype'] == 'WT']['dist'].values), (info[info['genotype'] == 'KO']['dist'].values)])
x2 = [x-width/2, x+width/2]
plt.plot(x2, np.vstack(pval), 'o', fillstyle = 'none', color = 'k', zorder = 5)
plt.errorbar(x2[0], np.mean(pval[0]), yerr = scipy.stats.sem(pval[0]) , fmt = 'o', color="k", linewidth = 2, capsize = 6)
plt.errorbar(x2[1], np.mean(pval[1]), yerr = scipy.stats.sem(pval[1]) , fmt = 'o', color="k", linewidth = 2, capsize = 6)
# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Distance travelled (cm)')
ax.set_xticks(x)
ax.set_xticklabels(label)
ax.legend(loc = 'upper right')
ax.set_box_aspect(1)
fig.tight_layout()



#%% 

t, p = mannwhitneyu(np.array(info['dist'][info['genotype'] == 'WT']), np.array(info['dist'][info['genotype'] == 'KO']))

    
    
    
    