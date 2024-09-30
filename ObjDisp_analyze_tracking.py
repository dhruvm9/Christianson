#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 11:30:39 2024

@author: dhruv
"""

import pynapple as nap

import numpy as np
import pandas as pd
import scipy.io
import os, sys
import seaborn as sns
import matplotlib.pyplot as plt
from pylab import *

#%% 

data_directory = '/media/DataDhruv/Recordings/Christianson/ObjDisp-240802'
datasets = np.genfromtxt(os.path.join(data_directory,'dataset_encoding.list'), delimiter = '\n', dtype = str, comments = '#')

allsex = []
allmice = []
all_lefts = []
all_rights = []

for s in datasets:
    print(s)
    mousename = s[0:4]
    sex = s[4]

    allmice.append(mousename)
    allsex.append(sex)

    tracking_data =  pd.read_hdf(data_directory + '/' + s + '.h5', mode = 'r')

    position_cols = [0,1,3,4,6,7,9,10]
    trackedpos = tracking_data.iloc[:, position_cols]

    likelihoods = np.vstack([tracking_data.iloc[:,2], tracking_data.iloc[:,5]])

    pcutoff = 0.6

    tokeep = []
    for i in range(shape(likelihoods)[1]):
        if (likelihoods[0][i] > pcutoff) and (likelihoods[1][i] > pcutoff):
            tokeep.append(i)

    x_ear = [0,2]
    y_ear = [1,3]

    all_x_ear = trackedpos.iloc[:, x_ear]
    all_y_ear = trackedpos.iloc[:, y_ear]
    
    all_x_objL = trackedpos.iloc[:,4]
    all_y_objL = trackedpos.iloc[:,5]
    
    all_x_objR = trackedpos.iloc[:,6]
    all_y_objR = trackedpos.iloc[:,7]
    
    x_objL = all_x_objL.mean()
    y_objL = all_y_objL.mean()
    
    x_objR = all_x_objR.mean()
    y_objR = all_y_objR.mean()
    
    x_cent = all_x_ear.sum(axis=1)/all_x_ear.iloc[0,:].shape[0]
    y_cent = all_y_ear.sum(axis=1)/all_y_ear.iloc[0,:].shape[0]

    mouseposition = np.zeros((len(x_cent),2))
    mouseposition[:,0] = x_cent 
    mouseposition[:,1] = y_cent

    x = mouseposition[:,0]
    y = mouseposition[:,1]
    
    fs = 30
    timestamps = x_cent.index.values/fs
    
    position = np.vstack([x, y]).T
    position = nap.TsdFrame(t = timestamps[tokeep], d = position[tokeep], columns = ['x', 'y'], time_units = 's')
    
    objL_coords = np.hstack([x_objL, y_objL])
    objR_coords = np.hstack([x_objR, y_objR])

#%% 

    # plt.figure()
    # plt.plot(position['x'], position['y'],'.')
    # plt.plot(objL_coords[0], objL_coords[1], 'o', color = 'k')
    # plt.plot(objR_coords[0], objR_coords[1], 'o', color = 'r')

#%% 

    roi = 100
    
    # circle1 = plt.Circle((objL_coords[0], objL_coords[1]), roi, color='k', fill = False)
    # circle2 = plt.Circle((objR_coords[0], objR_coords[1]), roi, color='r', fill = False)
    # ax = sns.scatterplot(data = position.as_dataframe(), x = x[tokeep], y = y[tokeep])
    # ax.add_patch(circle1)
    # ax.add_patch(circle2)

#%% 

    d_objL = np.sqrt((x[tokeep] - objL_coords[0])**2 + (y[tokeep] - objL_coords[1])**2)
    dist_objL = nap.Tsd(t = timestamps[tokeep], d = d_objL, time_units = 's')
    
    d_objR = np.sqrt((x[tokeep] - objR_coords[0])**2 + (y[tokeep] - objR_coords[1])**2)
    dist_objR = nap.Tsd(t = timestamps[tokeep], d = d_objR, time_units = 's')

#%%

    within_objL = dist_objL.threshold(roi, 'below')
    ep_objL = within_objL.time_support
    
    within_objR = dist_objR.threshold(roi, 'below')
    ep_objR = within_objR.time_support


#%% 

    plt.figure()
    sns.scatterplot(data = position.as_dataframe(), x = x[tokeep], y = y[tokeep])
    plt.scatter(position['x'].restrict(ep_objL), position['y'].restrict(ep_objL), zorder = 2, label = 'left ROI', color = 'k')
    plt.scatter(position['x'].restrict(ep_objR), position['y'].restrict(ep_objR), zorder = 2, label = 'right ROI', color = 'r')
    plt.legend(loc = 'upper right')

#%% 

    objL_dur = (ep_objL['end'] - ep_objL['start']).sum()
    objR_dur = (ep_objR['end'] - ep_objR['start']).sum()
    
    all_lefts.append(objL_dur)
    all_rights.append(objR_dur)
    
#%% 