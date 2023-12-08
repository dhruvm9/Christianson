#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 11:40:22 2023

@author: dhruv
"""

import numpy as np 
import pandas as pd 
import pynapple as nap 
import scipy.io
import os, sys
import matplotlib.pyplot as plt 
import seaborn as sns 
from scipy.stats import mannwhitneyu

#%% 

data_directory = '/media/dhruv/Expansion/Processed'
datasets = np.genfromtxt(os.path.join(data_directory,'dataset_DM.list'), delimiter = '\n', dtype = str, comments = '#')

SFI_wt = []
SFI_ko = []

p_ww_wt = []
p_wn_wt = []
p_wr_wt = []
p_nw_wt = []
p_nn_wt = []
p_nr_wt = []
p_rw_wt = []
p_rn_wt = []
p_rr_wt = []

p_ww_ko = []
p_wn_ko = []
p_wr_ko = []
p_nw_ko = []
p_nn_ko = []
p_nr_ko = []
p_rw_ko = []
p_rn_ko = []
p_rr_ko = []

for s in datasets:
    print(s)
    name = s.split('-')[0]
    path = os.path.join(data_directory, s)
    
    if name == 'B2613' or name == 'B2618':
        isWT = 0
    else: isWT = 1 
       
    listdir    = os.listdir(path)
    file = [f for f in listdir if 'SleepState.states' in f]
    states = scipy.io.loadmat(os.path.join(path,file[0])) 
    
    sleepstate = states['SleepState']
    
    # 1 = WAKE, 3 = NREM, 5 = REM
    scoredsleep = nap.Tsd(t = sleepstate[0][0][1][0][0][1][0:-1].flatten(), d = sleepstate[0][0][1][0][0][2][0:-1].flatten())
          

#%%  Count transitions

    c_ww = 0
    c_wn = 0
    c_wr = 0
    c_nw = 0
    c_nn = 0
    c_nr = 0
    c_rw = 0
    c_rn = 0
    c_rr = 0
    
    wakedur = 0
    nremdur = 0
    remdur = 0
    
    for i in range(len(scoredsleep)): 
        if scoredsleep.values[i] == 1:
            wakedur +=1
        elif scoredsleep.values[i] == 3:
            nremdur +=1
        elif scoredsleep.values[i] == 5:
            remdur +=1
        
    sleepdur = nremdur + remdur 
    recdur = wakedur + nremdur + remdur
    
    
    for i in range(len(scoredsleep)-1): 
        if scoredsleep.values[i] == 1 and scoredsleep.values[i+1] == 1:
            c_ww += 1
        elif scoredsleep.values[i] == 1 and scoredsleep.values[i+1] == 3:
            c_wn += 1
        elif scoredsleep.values[i] == 1 and scoredsleep.values[i+1] == 5:
            c_wr += 1
        elif scoredsleep.values[i] == 3 and scoredsleep.values[i+1] == 1:
            c_nw += 1
        elif scoredsleep.values[i] == 3 and scoredsleep.values[i+1] == 3:
            c_nn += 1
        elif scoredsleep.values[i] == 3 and scoredsleep.values[i+1] == 5:
            c_nr += 1
        elif scoredsleep.values[i] == 5 and scoredsleep.values[i+1] == 1:
            c_rw += 1
        elif scoredsleep.values[i] == 5 and scoredsleep.values[i+1] == 3:
            c_rn += 1
        elif scoredsleep.values[i] == 5 and scoredsleep.values[i+1] == 5:
            c_rr += 1
     
        
#%% Transition probabilities and SFI
    
    SFI = (c_nw + c_rw) / (sleepdur/3600) #Units of events per hour
    
    p_ww = c_ww / recdur
    p_wn = c_wn / recdur
    p_wr = c_wr / recdur
    p_nw = c_nw / recdur
    p_nn = c_nn / recdur
    p_nr = c_nr / recdur
    p_rw = c_rw / recdur
    p_rn = c_rn / recdur
    p_rr = c_rr / recdur

#%% Sort by genotype 

    if isWT == 1:
        SFI_wt.append(SFI)
        p_ww_wt.append(p_ww)
        p_wn_wt.append(p_wn)
        p_wr_wt.append(p_wr)
        p_nw_wt.append(p_nw)
        p_nn_wt.append(p_nn)
        p_nr_wt.append(p_nr)
        p_rw_wt.append(p_rw)
        p_rn_wt.append(p_rn)
        p_rr_wt.append(p_rr)
        
    else: 
        
        SFI_ko.append(SFI)
        p_ww_ko.append(p_ww)
        p_wn_ko.append(p_wn)
        p_wr_ko.append(p_wr)
        p_nw_ko.append(p_nw)
        p_nn_ko.append(p_nn)
        p_nr_ko.append(p_nr)
        p_rw_ko.append(p_rw)
        p_rn_ko.append(p_rn)
        p_rr_ko.append(p_rr)
    
#%% Organize SFI data

wt = np.array(['WT' for x in range(len(SFI_wt))])
ko = np.array(['KO' for x in range(len(SFI_ko))])

genotype = np.hstack([wt, ko])
allSFI = np.hstack([SFI_wt, SFI_ko])

infos = pd.DataFrame(data = [allSFI, genotype], index = ['SFI', 'genotype']).T

###SFI stats
t, p = mannwhitneyu(SFI_wt, SFI_ko)

#%% Plot SFI data

plt.figure()
plt.title('Sleep Fragmentation')
sns.set_style('white')
palette = ['royalblue', 'indianred']
ax = sns.violinplot( x = infos['genotype'], y=infos['SFI'].astype(float) , data = infos, dodge=False,
                    palette = palette,cut = 2,
                    scale="width", inner=None)
ax.tick_params(bottom=True, left=True)
xlim = ax.get_xlim()
ylim = ax.get_ylim()
for violin in ax.collections:
    x0, y0, width, height = violin.get_paths()[0].get_extents().bounds
    violin.set_clip_path(plt.Rectangle((x0, y0), width / 2, height, transform=ax.transData))
sns.boxplot(x = infos['genotype'], y=infos['SFI'] , data = infos, saturation=1, showfliers=False,
            width=0.3, boxprops={'zorder': 3, 'facecolor': 'none'}, ax=ax)
old_len_collections = len(ax.collections)
sns.swarmplot(x = infos['genotype'], y=infos['SFI'], data = infos, color = 'k', dodge=False, ax=ax)
# sns.stripplot(x = wakedf['type'], y = wakedf['rate'].astype(float), data = wakedf, color = 'k', dodge=False, ax=ax)
for dots in ax.collections[old_len_collections:]:
    dots.set_offsets(dots.get_offsets())
ax.set_xlim(xlim)
ax.set_ylim(ylim)
plt.ylabel('SFI (events/h)')
ax.set_box_aspect(1)

#%% Transition probability stats

t_ww, ww = mannwhitneyu(p_ww_wt, p_ww_ko)
t_wn, wn = mannwhitneyu(p_wn_wt, p_wn_ko)
t_wr, wr = mannwhitneyu(p_wr_wt, p_wr_ko)
t_nw, nw = mannwhitneyu(p_nw_wt, p_nw_ko)
t_nn, nn = mannwhitneyu(p_nn_wt, p_nn_ko)
t_nr, nr = mannwhitneyu(p_nr_wt, p_nr_ko)
t_rw, rw = mannwhitneyu(p_rw_wt, p_rw_ko)
t_rn, rn = mannwhitneyu(p_rn_wt, p_rn_ko)
t_rr, rr = mannwhitneyu(p_rr_wt, p_rr_ko)
    