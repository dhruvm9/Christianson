#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 16:21:31 2023

@author: dhruv
"""

import numpy as np 
import pandas as pd 
import nwbmatic as ntm
import pynapple as nap 
import scipy.io
import os, sys
import matplotlib.pyplot as plt 
import seaborn as sns 
from scipy.stats import mannwhitneyu

#%% 

data_directory = '/media/adrien/Expansion/Processed/NoExplo'
datasets = np.genfromtxt(os.path.join(data_directory,'dataset_sleep.list'), delimiter = '\n', dtype = str, comments = '#')

allwakedurs_wt = []
allnremdurs_wt = []
allremdurs_wt = []

allwakedurs_ko = []
allnremdurs_ko = []
allremdurs_ko = []

sess_wakedur_wt = []
sess_nremdur_wt = []
sess_remdur_wt = []

sess_wakedur_ko = []
sess_nremdur_ko = []
sess_remdur_ko = []

for s in datasets:
    print(s)
    name = s.split('-')[0]
    path = os.path.join(data_directory, s)
    
    if name == 'B3900' or name == 'B3901':
        isWT = 0
    else: isWT = 1 
       
    listdir    = os.listdir(path)
    file = [f for f in listdir if 'SleepState.states' in f]
    states = scipy.io.loadmat(os.path.join(path,file[0])) 
    
    sleepstate = states['SleepState']
    wake_ep = nap.IntervalSet(start = sleepstate[0][0][0][0][0][0][:,0], end = sleepstate[0][0][0][0][0][0][:,1])
    nrem_ep = nap.IntervalSet(start = sleepstate[0][0][0][0][0][1][:,0], end = sleepstate[0][0][0][0][0][1][:,1])
    rem_ep = nap.IntervalSet(start = sleepstate[0][0][0][0][0][2][:,0], end = sleepstate[0][0][0][0][0][2][:,1])
    
    recdur = len(sleepstate[0][0][1][0][0][1].flatten())
    
#%% 

    wakedur = wake_ep['end'] - wake_ep['start']
    nremdur = nrem_ep['end'] - nrem_ep['start']
    remdur = rem_ep['end'] - rem_ep['start']
    
    if isWT == 1:
        sess_wakedur_wt.append(sum(wakedur)/recdur)
        sess_nremdur_wt.append(sum(nremdur)/recdur)
        sess_remdur_wt.append(sum(remdur)/recdur)
        
        allwakedurs_wt.extend(wakedur/recdur)
        allnremdurs_wt.extend(nremdur/recdur)
        allremdurs_wt.extend(remdur/recdur)
        
    else: 
        sess_wakedur_ko.append(sum(wakedur)/recdur)
        sess_nremdur_ko.append(sum(nremdur)/recdur)
        sess_remdur_ko.append(sum(remdur)/recdur)
        
        allwakedurs_ko.extend(wakedur/recdur)
        allnremdurs_ko.extend(nremdur/recdur)
        allremdurs_ko.extend(remdur/recdur)

#%% Organize data to plot 

wt1 = np.array(['WT' for x in range(len(sess_wakedur_wt))])
wt2 = np.array(['WT' for x in range(len(sess_nremdur_wt))])
wt3 = np.array(['WT' for x in range(len(sess_remdur_wt))])

ko1 = np.array(['KO' for x in range(len(sess_wakedur_ko))])
ko2 = np.array(['KO' for x in range(len(sess_nremdur_ko))])
ko3 = np.array(['KO' for x in range(len(sess_remdur_ko))])

genotype = np.hstack([wt1, ko1, wt2, ko2, wt3, ko3])

wk = np.array(['Wake' for x in range(len(sess_wakedur_wt))])
wk2 = np.array(['Wake' for x in range(len(sess_wakedur_ko))])

nr = np.array(['NREM' for x in range(len(sess_nremdur_wt))])
nr2 = np.array(['NREM' for x in range(len(sess_nremdur_ko))])

rm = np.array(['REM' for x in range(len(sess_remdur_wt))])
rm2 = np.array(['REM' for x in range(len(sess_remdur_ko))])

state = np.hstack([wk, wk2, nr, nr2, rm, rm2])

wakeprop = []
wakeprop.extend(sess_wakedur_wt)
wakeprop.extend(sess_wakedur_ko)

nremprop = []
nremprop.extend(sess_nremdur_wt)
nremprop.extend(sess_nremdur_ko)

remprop = []
remprop.extend(sess_remdur_wt)
remprop.extend(sess_remdur_ko)

dur = np.hstack([wakeprop, nremprop, remprop])

infos = pd.DataFrame(data = [dur, state, genotype], index = ['dur', 'state', 'genotype']).T

#%% 

plt.figure()
plt.subplot(131)
plt.title('Wake')
sns.set_style('white')
palette = ['royalblue', 'indianred']
ax = sns.violinplot( x = infos[infos['state'] == 'Wake']['genotype'], y=infos[infos['state'] == 'Wake']['dur'].astype(float) , data = infos, dodge=False,
                    palette = palette,cut = 2,
                    scale="width", inner=None)
ax.tick_params(bottom=True, left=True)
xlim = ax.get_xlim()
ylim = ax.get_ylim()
for violin in ax.collections:
    x0, y0, width, height = violin.get_paths()[0].get_extents().bounds
    violin.set_clip_path(plt.Rectangle((x0, y0), width / 2, height, transform=ax.transData))
sns.boxplot(x = infos[infos['state'] == 'Wake']['genotype'], y=infos[infos['state'] == 'Wake']['dur'] , data = infos, saturation=1, showfliers=False,
            width=0.3, boxprops={'zorder': 3, 'facecolor': 'none'}, ax=ax)
old_len_collections = len(ax.collections)
sns.swarmplot(x = infos[infos['state'] == 'Wake']['genotype'], y=infos[infos['state'] == 'Wake']['dur'], data = infos, color = 'k', dodge=False, ax=ax)
# sns.stripplot(x = wakedf['type'], y = wakedf['rate'].astype(float), data = wakedf, color = 'k', dodge=False, ax=ax)
for dots in ax.collections[old_len_collections:]:
    dots.set_offsets(dots.get_offsets())
ax.set_xlim(xlim)
ax.set_ylim(ylim)
plt.ylabel('Proportion')
ax.set_box_aspect(1)

plt.subplot(132)
plt.title('NREM')
sns.set_style('white')
palette = ['royalblue', 'indianred']
ax = sns.violinplot( x = infos[infos['state'] == 'NREM']['genotype'], y=infos[infos['state'] == 'NREM']['dur'].astype(float) , data = infos, dodge=False,
                    palette = palette,cut = 2,
                    scale="width", inner=None)
ax.tick_params(bottom=True, left=True)
xlim = ax.get_xlim()
ylim = ax.get_ylim()
for violin in ax.collections:
    x0, y0, width, height = violin.get_paths()[0].get_extents().bounds
    violin.set_clip_path(plt.Rectangle((x0, y0), width / 2, height, transform=ax.transData))
sns.boxplot(x = infos[infos['state'] == 'NREM']['genotype'], y=infos[infos['state'] == 'NREM']['dur'] , data = infos, saturation=1, showfliers=False,
            width=0.3, boxprops={'zorder': 3, 'facecolor': 'none'}, ax=ax)
old_len_collections = len(ax.collections)
sns.swarmplot(x = infos[infos['state'] == 'NREM']['genotype'], y=infos[infos['state'] == 'NREM']['dur'], data = infos, color = 'k', dodge=False, ax=ax)
# sns.stripplot(x = wakedf['type'], y = wakedf['rate'].astype(float), data = wakedf, color = 'k', dodge=False, ax=ax)
for dots in ax.collections[old_len_collections:]:
    dots.set_offsets(dots.get_offsets())
ax.set_xlim(xlim)
ax.set_ylim(ylim)
plt.ylabel('Proportion')
ax.set_box_aspect(1)

plt.subplot(133)
plt.title('REM')
sns.set_style('white')
palette = ['royalblue', 'indianred']
ax = sns.violinplot( x = infos[infos['state'] == 'REM']['genotype'], y=infos[infos['state'] == 'REM']['dur'].astype(float) , data = infos, dodge=False,
                    palette = palette,cut = 2,
                    scale="width", inner=None)
ax.tick_params(bottom=True, left=True)
xlim = ax.get_xlim()
ylim = ax.get_ylim()
for violin in ax.collections:
    x0, y0, width, height = violin.get_paths()[0].get_extents().bounds
    violin.set_clip_path(plt.Rectangle((x0, y0), width / 2, height, transform=ax.transData))
sns.boxplot(x = infos[infos['state'] == 'REM']['genotype'], y=infos[infos['state'] == 'REM']['dur'] , data = infos, saturation=1, showfliers=False,
            width=0.3, boxprops={'zorder': 3, 'facecolor': 'none'}, ax=ax)
old_len_collections = len(ax.collections)
sns.swarmplot(x = infos[infos['state'] == 'REM']['genotype'], y=infos[infos['state'] == 'REM']['dur'], data = infos, color = 'k', dodge=False, ax=ax)
# sns.stripplot(x = wakedf['type'], y = wakedf['rate'].astype(float), data = wakedf, color = 'k', dodge=False, ax=ax)
for dots in ax.collections[old_len_collections:]:
    dots.set_offsets(dots.get_offsets())
ax.set_xlim(xlim)
ax.set_ylim(ylim)
plt.ylabel('Proportion')
ax.set_box_aspect(1)

#%% Stats 

t_w, p_w = mannwhitneyu(infos[infos['state'] == 'Wake'][infos['genotype'] == 'KO']['dur'].values.astype(float), 
                    infos[infos['state'] == 'Wake'][infos['genotype'] == 'WT']['dur'].values.astype(float))

t_n, p_n = mannwhitneyu(infos[infos['state'] == 'NREM'][infos['genotype'] == 'KO']['dur'].values.astype(float), 
                    infos[infos['state'] == 'NREM'][infos['genotype'] == 'WT']['dur'].values.astype(float))

t_r, p_r = mannwhitneyu(infos[infos['state'] == 'REM'][infos['genotype'] == 'KO']['dur'].values.astype(float), 
                    infos[infos['state'] == 'REM'][infos['genotype'] == 'WT']['dur'].values.astype(float))


#%% 

wt1 = np.array(['WT' for x in range(len(allwakedurs_wt))])
wt2 = np.array(['WT' for x in range(len(allnremdurs_wt))])
wt3 = np.array(['WT' for x in range(len(allremdurs_wt))])

ko1 = np.array(['KO' for x in range(len(allwakedurs_ko))])
ko2 = np.array(['KO' for x in range(len(allnremdurs_ko))])
ko3 = np.array(['KO' for x in range(len(allremdurs_ko))])

genotype = np.hstack([wt1, ko1, wt2, ko2, wt3, ko3])

wk = np.array(['Wake' for x in range(len(allwakedurs_wt))])
wk2 = np.array(['Wake' for x in range(len(allwakedurs_ko))])

nr = np.array(['NREM' for x in range(len(allnremdurs_wt))])
nr2 = np.array(['NREM' for x in range(len(allnremdurs_ko))])

rm = np.array(['REM' for x in range(len(allremdurs_wt))])
rm2 = np.array(['REM' for x in range(len(allremdurs_ko))])

state = np.hstack([wk, wk2, nr, nr2, rm, rm2])

wakevents = []
wakevents.extend(allwakedurs_wt)
wakevents.extend(allwakedurs_ko)

nremevents = []
nremevents.extend(allnremdurs_wt)
nremevents.extend(allnremdurs_ko)

remevents = []
remevents.extend(allremdurs_wt)
remevents.extend(allremdurs_ko)

evts = np.hstack([wakevents, nremevents, remevents])

infos2 = pd.DataFrame(data = [evts, state, genotype], index = ['evt', 'state', 'genotype']).T

#%% 

plt.figure()
plt.subplot(131)
plt.title('Wake')
sns.set_style('white')
palette = ['royalblue', 'indianred']
ax = sns.violinplot( x = infos2[infos2['state'] == 'Wake']['genotype'], y=infos2[infos2['state'] == 'Wake']['evt'].astype(float) , data = infos2, dodge=False,
                    palette = palette,cut = 2,
                    scale="width", inner=None)
ax.tick_params(bottom=True, left=True)
xlim = ax.get_xlim()
ylim = ax.get_ylim()
for violin in ax.collections:
    x0, y0, width, height = violin.get_paths()[0].get_extents().bounds
    violin.set_clip_path(plt.Rectangle((x0, y0), width / 2, height, transform=ax.transData))
sns.boxplot(x = infos2[infos2['state'] == 'Wake']['genotype'], y=infos2[infos2['state'] == 'Wake']['evt'] , data = infos2, saturation=1, showfliers=False,
            width=0.3, boxprops={'zorder': 3, 'facecolor': 'none'}, ax=ax)
old_len_collections = len(ax.collections)
sns.stripplot(x = infos2[infos2['state'] == 'Wake']['genotype'], y=infos2[infos2['state'] == 'Wake']['evt'], data = infos2, color = 'k', dodge=False, ax=ax)

for dots in ax.collections[old_len_collections:]:
    dots.set_offsets(dots.get_offsets())
ax.set_xlim(xlim)
ax.set_ylim(ylim)
plt.ylabel('Proportion')
ax.set_box_aspect(1)

plt.subplot(132)
plt.title('NREM')
sns.set_style('white')
palette = ['royalblue', 'indianred']
ax = sns.violinplot( x = infos2[infos2['state'] == 'NREM']['genotype'], y=infos2[infos2['state'] == 'NREM']['evt'].astype(float) , data = infos2, dodge=False,
                    palette = palette,cut = 2,
                    scale="width", inner=None)
ax.tick_params(bottom=True, left=True)
xlim = ax.get_xlim()
ylim = ax.get_ylim()
for violin in ax.collections:
    x0, y0, width, height = violin.get_paths()[0].get_extents().bounds
    violin.set_clip_path(plt.Rectangle((x0, y0), width / 2, height, transform=ax.transData))
sns.boxplot(x = infos2[infos2['state'] == 'NREM']['genotype'], y=infos2[infos2['state'] == 'NREM']['evt'] , data = infos2, saturation=1, showfliers=False,
            width=0.3, boxprops={'zorder': 3, 'facecolor': 'none'}, ax=ax)
old_len_collections = len(ax.collections)
sns.stripplot(x = infos2[infos2['state'] == 'NREM']['genotype'], y=infos2[infos2['state'] == 'NREM']['evt'], data = infos2, color = 'k', dodge=False, ax=ax)

for dots in ax.collections[old_len_collections:]:
    dots.set_offsets(dots.get_offsets())
ax.set_xlim(xlim)
ax.set_ylim(ylim)
plt.ylabel('Proportion')
ax.set_box_aspect(1)

plt.subplot(133)
plt.title('REM')
sns.set_style('white')
palette = ['royalblue', 'indianred']
ax = sns.violinplot( x = infos2[infos2['state'] == 'REM']['genotype'], y=infos2[infos2['state'] == 'REM']['evt'].astype(float) , data = infos2, dodge=False,
                    palette = palette,cut = 2,
                    scale="width", inner=None)
ax.tick_params(bottom=True, left=True)
xlim = ax.get_xlim()
ylim = ax.get_ylim()
for violin in ax.collections:
    x0, y0, width, height = violin.get_paths()[0].get_extents().bounds
    violin.set_clip_path(plt.Rectangle((x0, y0), width / 2, height, transform=ax.transData))
sns.boxplot(x = infos2[infos2['state'] == 'REM']['genotype'], y=infos2[infos2['state'] == 'REM']['evt'] , data = infos2, saturation=1, showfliers=False,
            width=0.3, boxprops={'zorder': 3, 'facecolor': 'none'}, ax=ax)
old_len_collections = len(ax.collections)
sns.stripplot(x = infos2[infos2['state'] == 'REM']['genotype'], y=infos2[infos2['state'] == 'REM']['evt'], data = infos2, color = 'k', dodge=False, ax=ax)

for dots in ax.collections[old_len_collections:]:
    dots.set_offsets(dots.get_offsets())
ax.set_xlim(xlim)
ax.set_ylim(ylim)
plt.ylabel('Proportion')
ax.set_box_aspect(1)

#%% Stats 

t2_w, p2_w = mannwhitneyu(infos2[infos2['state'] == 'Wake'][infos2['genotype'] == 'KO']['evt'].values.astype(float), 
                    infos2[infos2['state'] == 'Wake'][infos2['genotype'] == 'WT']['evt'].values.astype(float))

t2_n, p2_n = mannwhitneyu(infos2[infos2['state'] == 'NREM'][infos2['genotype'] == 'KO']['evt'].values.astype(float), 
                    infos2[infos2['state'] == 'NREM'][infos2['genotype'] == 'WT']['evt'].values.astype(float))

t2_r, p2_r = mannwhitneyu(infos2[infos2['state'] == 'REM'][infos2['genotype'] == 'KO']['evt'].values.astype(float), 
                    infos2[infos2['state'] == 'REM'][infos2['genotype'] == 'WT']['evt'].values.astype(float))
