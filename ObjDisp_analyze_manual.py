#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 19:17:38 2024

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
from scipy.stats import mannwhitneyu, wilcoxon

#%% 

data_directory = '/media/DataDhruv/Recordings/Christianson/ObjDisp-240802'

# info1 = pd.read_excel(os.path.join(data_directory,'info1.xlsx'))
info2 = pd.read_excel(os.path.join(data_directory,'info2.xlsx'))

#%% 

# m_wt1 = info1[(info1['Sex'] == 'M') & (info1['Genotype'] == 'WT')]['DI1'].values
# m_ko1 = info1[(info1['Sex'] == 'M') & (info1['Genotype'] == 'KO')]['DI1'].values
# f_wt1 = info1[(info1['Sex'] == 'F') & (info1['Genotype'] == 'WT')]['DI1'].values
# f_ko1 = info1[(info1['Sex'] == 'F') & (info1['Genotype'] == 'KO')]['DI1'].values

# m_wt2 = info1[(info1['Sex'] == 'M') & (info1['Genotype'] == 'WT')]['DI2'].values
# m_ko2 = info1[(info1['Sex'] == 'M') & (info1['Genotype'] == 'KO')]['DI2'].values
# f_wt2 = info1[(info1['Sex'] == 'F') & (info1['Genotype'] == 'WT')]['DI2'].values
# f_ko2 = info1[(info1['Sex'] == 'F') & (info1['Genotype'] == 'KO')]['DI2'].values

# wt_m = np.array(['WT_male' for x in range(len(m_wt1))])
# ko_m = np.array(['KO_male' for x in range(len(m_ko1))])
# wt_f = np.array(['WT_female' for x in range(len(f_wt1))])
# ko_f = np.array(['KO_female' for x in range(len(f_ko1))])

# gtype = np.hstack([wt_m, ko_m, wt_f, ko_f])
# DI1s = np.hstack([m_wt1, m_ko1, f_wt1, f_ko1])
# DI2s = np.hstack([m_wt2, m_ko2, f_wt2, f_ko2])

# infos_phase1 = pd.DataFrame(data = [DI1s, DI2s, gtype], index = ['DI1', 'DI2', 'genotype']).T

#%% 

m_wt1 = info2[(info2['Sex'] == 'M') & (info2['Genotype'] == 'WT')]['DI1'].values
m_ko1 = info2[(info2['Sex'] == 'M') & (info2['Genotype'] == 'KO')]['DI1'].values
f_wt1 = info2[(info2['Sex'] == 'F') & (info2['Genotype'] == 'WT')]['DI1'].values
f_ko1 = info2[(info2['Sex'] == 'F') & (info2['Genotype'] == 'KO')]['DI1'].values

m_wt2 = info2[(info2['Sex'] == 'M') & (info2['Genotype'] == 'WT')]['DI2'].values
m_ko2 = info2[(info2['Sex'] == 'M') & (info2['Genotype'] == 'KO')]['DI2'].values
f_wt2 = info2[(info2['Sex'] == 'F') & (info2['Genotype'] == 'WT')]['DI2'].values
f_ko2 = info2[(info2['Sex'] == 'F') & (info2['Genotype'] == 'KO')]['DI2'].values

wt_m = np.array(['WT_male' for x in range(len(m_wt1))])
ko_m = np.array(['KO_male' for x in range(len(m_ko1))])
wt_f = np.array(['WT_female' for x in range(len(f_wt1))])
ko_f = np.array(['KO_female' for x in range(len(f_ko1))])

gtype = np.hstack([wt_m, ko_m, wt_f, ko_f])
DI1s = np.hstack([m_wt1, m_ko1, f_wt1, f_ko1])
DI2s = np.hstack([m_wt2, m_ko2, f_wt2, f_ko2])

infos_phase2 = pd.DataFrame(data = [DI1s, DI2s, gtype], index = ['DI1', 'DI2', 'genotype']).T

#%% DI1 

# label = ['WT male']
# x = np.arange(len(label))  # the label locations
# width = 0.35  # the width of the bars

# fig, ax = plt.subplots()
# rects1 = ax.bar(x - width/2, infos_phase1[infos_phase1['genotype'] == 'WT_male']['DI1'].mean(), width, color = 'lightsteelblue', label='Encoding Phase')
# rects2 = ax.bar(x + width/2, infos_phase2[infos_phase2['genotype'] == 'WT_male']['DI1'].mean(), width, color = 'royalblue', label='Recall Phase')
# pval = np.vstack([(infos_phase1[infos_phase1['genotype'] == 'WT_male']['DI1'].values), (infos_phase2[infos_phase2['genotype'] == 'WT_male']['DI1'].values)])
# x2 = [x-width/2, x+width/2]
# plt.plot(x2, np.vstack(pval), 'o-', color = 'k')
# plt.errorbar(x - width/2, infos_phase1[infos_phase1['genotype'] == 'WT_male']['DI1'].mean(), yerr= infos_phase1[infos_phase1['genotype'] == 'WT_male']['DI1'].sem(), fmt = 'o', color = 'r', capsize = 5)
# plt.errorbar(x + width/2, infos_phase2[infos_phase2['genotype'] == 'WT_male']['DI1'].mean(), yerr= infos_phase2[infos_phase2['genotype'] == 'WT_male']['DI1'].sem(), fmt = 'o', color = 'r', capsize = 5)
# plt.axhline(0, linestyle = '--', color = 'silver')
# # Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_ylabel('Discrimination index')
# ax.set_title('WT male')
# ax.set_xticks(x)
# ax.legend(loc = 'upper left')
# ax.set_box_aspect(1)
# fig.tight_layout()

#%% WT male

# t_wtm, p_wtm = mannwhitneyu(infos_phase1[infos_phase1['genotype'] == 'WT_male']['DI1'].astype('float64'), infos_phase2[infos_phase2['genotype'] == 'WT_male']['DI1'].astype('float64'))
# z1_wtm, p1_wtm = wilcoxon(infos_phase1[infos_phase1['genotype'] == 'WT_male']['DI1'].values.astype('float64')-0)
# z2_wtm, p2_wtm = wilcoxon(infos_phase2[infos_phase2['genotype'] == 'WT_male']['DI1'].values.astype('float64')-0)
# z_pair_wtm, p_pair_wtm = wilcoxon(infos_phase1[infos_phase1['genotype'] == 'WT_male']['DI1'].values.astype('float64'), infos_phase2[infos_phase2['genotype'] == 'WT_male']['DI1'].values.astype('float64'))

#%% 

# fig, ax = plt.subplots()
# rects1 = ax.bar(x - width/2, infos_phase1[infos_phase1['genotype'] == 'KO_male']['DI1'].mean(), width, color = 'lightcoral', label='Encoding Phase')
# rects2 = ax.bar(x + width/2, infos_phase2[infos_phase2['genotype'] == 'KO_male']['DI1'].mean(), width, color = 'indianred', label='Recall Phase')
# pval = np.vstack([(infos_phase1[infos_phase1['genotype'] == 'KO_male']['DI1'].values), (infos_phase2[infos_phase2['genotype'] == 'KO_male']['DI1'].values)])
# x2 = [x-width/2, x+width/2]
# plt.plot(x2, np.vstack(pval), 'o-', color = 'k')
# plt.errorbar(x - width/2, infos_phase1[infos_phase1['genotype'] == 'KO_male']['DI1'].mean(), yerr= infos_phase1[infos_phase1['genotype'] == 'KO_male']['DI1'].sem(), fmt = 'o', color = 'r', capsize = 5)
# plt.errorbar(x + width/2, infos_phase2[infos_phase2['genotype'] == 'KO_male']['DI1'].mean(), yerr= infos_phase2[infos_phase2['genotype'] == 'KO_male']['DI1'].sem(), fmt = 'o', color = 'r', capsize = 5)
# plt.axhline(0, linestyle = '--', color = 'silver')
# # Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_ylabel('Discrimination index')
# ax.set_title('KO male')
# ax.set_xticks(x)
# ax.legend(loc = 'upper left')
# ax.set_box_aspect(1)
# fig.tight_layout()

#%% KO male

# t_kom, p_kom = mannwhitneyu(infos_phase1[infos_phase1['genotype'] == 'KO_male']['DI1'].astype('float64'), infos_phase2[infos_phase2['genotype'] == 'KO_male']['DI1'].astype('float64'))
# z1_kom, p1_kom = wilcoxon(infos_phase1[infos_phase1['genotype'] == 'KO_male']['DI1'].values.astype('float64')-0)
# z2_kom, p2_kom = wilcoxon(infos_phase2[infos_phase2['genotype'] == 'KO_male']['DI1'].values.astype('float64')-0)
# z_pair_kom, p_pair_kom = wilcoxon(infos_phase1[infos_phase1['genotype'] == 'KO_male']['DI1'].values.astype('float64'), infos_phase2[infos_phase2['genotype'] == 'KO_male']['DI1'].values.astype('float64'))

#%% 

# fig, ax = plt.subplots()
# rects1 = ax.bar(x - width/2, infos_phase1[infos_phase1['genotype'] == 'WT_female']['DI1'].mean(), width, color = 'mediumorchid', label='Encoding Phase')
# rects2 = ax.bar(x + width/2, infos_phase2[infos_phase2['genotype'] == 'WT_female']['DI1'].mean(), width, color = 'violet', label='Recall Phase')
# pval = np.vstack([(infos_phase1[infos_phase1['genotype'] == 'WT_female']['DI1'].values), (infos_phase2[infos_phase2['genotype'] == 'WT_female']['DI1'].values)])
# x2 = [x-width/2, x+width/2]
# plt.plot(x2, np.vstack(pval), 'o-', color = 'k')
# plt.errorbar(x - width/2, infos_phase1[infos_phase1['genotype'] == 'WT_female']['DI1'].mean(), yerr= infos_phase1[infos_phase1['genotype'] == 'WT_female']['DI1'].sem(), fmt = 'o', color = 'r', capsize = 5)
# plt.errorbar(x + width/2, infos_phase2[infos_phase2['genotype'] == 'WT_female']['DI1'].mean(), yerr= infos_phase2[infos_phase2['genotype'] == 'WT_female']['DI1'].sem(), fmt = 'o', color = 'r', capsize = 5)
# plt.axhline(0, linestyle = '--', color = 'silver')
# # Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_ylabel('Discrimination index')
# ax.set_title('WT female')
# ax.set_xticks(x)
# ax.legend(loc = 'upper left')
# ax.set_box_aspect(1)
# fig.tight_layout()

#%% WT female

# t_wtf, p_wtf = mannwhitneyu(infos_phase1[infos_phase1['genotype'] == 'WT_female']['DI1'].astype('float64'), infos_phase2[infos_phase2['genotype'] == 'WT_female']['DI1'].astype('float64'))
# z1_wtf, p1_wtf = wilcoxon(infos_phase1[infos_phase1['genotype'] == 'WT_female']['DI1'].values.astype('float64')-0)
# z2_wtf, p2_wtf = wilcoxon(infos_phase2[infos_phase2['genotype'] == 'WT_female']['DI1'].values.astype('float64')-0)
# z_pair_wtf, p_pair_wtf = wilcoxon(infos_phase1[infos_phase1['genotype'] == 'WT_female']['DI1'].values.astype('float64'), infos_phase2[infos_phase2['genotype'] == 'WT_female']['DI1'].values.astype('float64'))

#%% 

# fig, ax = plt.subplots()
# rects1 = ax.bar(x - width/2, infos_phase1[infos_phase1['genotype'] == 'KO_female']['DI1'].mean(), width, color = 'cadetblue', label='Encoding Phase')
# rects2 = ax.bar(x + width/2, infos_phase2[infos_phase2['genotype'] == 'KO_female']['DI1'].mean(), width, color = 'darkslategray', label='Recall Phase')
# pval = np.vstack([(infos_phase1[infos_phase1['genotype'] == 'KO_female']['DI1'].values), (infos_phase2[infos_phase2['genotype'] == 'KO_female']['DI1'].values)])
# x2 = [x-width/2, x+width/2]
# plt.plot(x2, np.vstack(pval), 'o-', color = 'k')
# plt.errorbar(x - width/2, infos_phase1[infos_phase1['genotype'] == 'KO_female']['DI1'].mean(), yerr= infos_phase1[infos_phase1['genotype'] == 'KO_female']['DI1'].sem(), fmt = 'o', color = 'r', capsize = 5)
# plt.errorbar(x + width/2, infos_phase2[infos_phase2['genotype'] == 'KO_female']['DI1'].mean(), yerr= infos_phase2[infos_phase2['genotype'] == 'KO_female']['DI1'].sem(), fmt = 'o', color = 'r', capsize = 5)
# plt.axhline(0, linestyle = '--', color = 'silver')
# # Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_ylabel('Discrimination index')
# ax.set_title('KO female')
# ax.set_xticks(x)
# ax.legend(loc = 'upper left')
# ax.set_box_aspect(1)
# fig.tight_layout()

#%% KO female

# t_kof, p_kof = mannwhitneyu(infos_phase1[infos_phase1['genotype'] == 'KO_female']['DI1'].astype('float64'), infos_phase2[infos_phase2['genotype'] == 'KO_female']['DI1'].astype('float64'))
# z1_kof, p1_kof = wilcoxon(infos_phase1[infos_phase1['genotype'] == 'KO_female']['DI1'].values.astype('float64')-0)
# z2_kof, p2_kof = wilcoxon(infos_phase2[infos_phase2['genotype'] == 'KO_female']['DI1'].values.astype('float64')-0)
# z_pair_kof, p_pair_kof = wilcoxon(infos_phase1[infos_phase1['genotype'] == 'KO_female']['DI1'].values.astype('float64'), infos_phase2[infos_phase2['genotype'] == 'KO_female']['DI1'].values.astype('float64'))

#%% Across recall phase only

# t1, p_wtm_kom = mannwhitneyu(infos_phase1[infos_phase2['genotype'] == 'WT_male']['DI1'].astype('float64'), infos_phase2[infos_phase2['genotype'] == 'KO_male']['DI1'].astype('float64'))
# t2, p_wtm_wtf = mannwhitneyu(infos_phase1[infos_phase2['genotype'] == 'WT_male']['DI1'].astype('float64'), infos_phase2[infos_phase2['genotype'] == 'WT_female']['DI1'].astype('float64'))
# t3, p_wtm_kof = mannwhitneyu(infos_phase1[infos_phase2['genotype'] == 'WT_male']['DI1'].astype('float64'), infos_phase2[infos_phase2['genotype'] == 'KO_female']['DI1'].astype('float64'))
# t4, p_kom_wtf = mannwhitneyu(infos_phase1[infos_phase2['genotype'] == 'KO_male']['DI1'].astype('float64'), infos_phase2[infos_phase2['genotype'] == 'WT_female']['DI1'].astype('float64'))
# t5, p_kom_kof = mannwhitneyu(infos_phase1[infos_phase2['genotype'] == 'KO_male']['DI1'].astype('float64'), infos_phase2[infos_phase2['genotype'] == 'KO_female']['DI1'].astype('float64'))
# t6, p_wtf_kof = mannwhitneyu(infos_phase1[infos_phase2['genotype'] == 'WT_female']['DI1'].astype('float64'), infos_phase2[infos_phase2['genotype'] == 'KO_female']['DI1'].astype('float64'))

#%% DI2

# label = ['WT male']
# x = np.arange(len(label))  # the label locations
# width = 0.35  # the width of the bars

# fig, ax = plt.subplots()
# rects1 = ax.bar(x - width/2, infos_phase1[infos_phase1['genotype'] == 'WT_male']['DI2'].mean(), width, color = 'lightsteelblue', label='Encoding Phase')
# rects2 = ax.bar(x + width/2, infos_phase2[infos_phase2['genotype'] == 'WT_male']['DI2'].mean(), width, color = 'royalblue', label='Recall Phase')
# pval = np.vstack([(infos_phase1[infos_phase1['genotype'] == 'WT_male']['DI2'].values), (infos_phase2[infos_phase2['genotype'] == 'WT_male']['DI2'].values)])
# x2 = [x-width/2, x+width/2]
# plt.plot(x2, np.vstack(pval), 'o-', color = 'k')
# plt.errorbar(x - width/2, infos_phase1[infos_phase1['genotype'] == 'WT_male']['DI2'].mean(), yerr= infos_phase1[infos_phase1['genotype'] == 'WT_male']['DI2'].sem(), fmt = 'o', color = 'r', capsize = 5)
# plt.errorbar(x + width/2, infos_phase2[infos_phase2['genotype'] == 'WT_male']['DI2'].mean(), yerr= infos_phase2[infos_phase2['genotype'] == 'WT_male']['DI2'].sem(), fmt = 'o', color = 'r', capsize = 5)
# plt.axhline(0.5, linestyle = '--', color = 'silver')
# # Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_ylabel('Discrimination index')
# ax.set_title('WT male')
# ax.set_xticks(x)
# ax.legend(loc = 'upper left')
# ax.set_box_aspect(1)
# fig.tight_layout()

#%% WT male

# t_wtm, p_wtm = mannwhitneyu(infos_phase1[infos_phase1['genotype'] == 'WT_male']['DI2'].astype('float64'), infos_phase2[infos_phase2['genotype'] == 'WT_male']['DI2'].astype('float64'))
# z1_wtm, p1_wtm = wilcoxon(infos_phase1[infos_phase1['genotype'] == 'WT_male']['DI2'].values.astype('float64')-0.5)
# z2_wtm, p2_wtm = wilcoxon(infos_phase2[infos_phase2['genotype'] == 'WT_male']['DI2'].values.astype('float64')-0.5)
# z_pair_wtm, p_pair_wtm = wilcoxon(infos_phase1[infos_phase1['genotype'] == 'WT_male']['DI2'].values.astype('float64'), infos_phase2[infos_phase2['genotype'] == 'WT_male']['DI2'].values.astype('float64'))

#%% 

# fig, ax = plt.subplots()
# rects1 = ax.bar(x - width/2, infos_phase1[infos_phase1['genotype'] == 'KO_male']['DI2'].mean(), width, color = 'lightcoral', label='Encoding Phase')
# rects2 = ax.bar(x + width/2, infos_phase2[infos_phase2['genotype'] == 'KO_male']['DI2'].mean(), width, color = 'indianred', label='Recall Phase')
# pval = np.vstack([(infos_phase1[infos_phase1['genotype'] == 'KO_male']['DI2'].values), (infos_phase2[infos_phase2['genotype'] == 'KO_male']['DI2'].values)])
# x2 = [x-width/2, x+width/2]
# plt.plot(x2, np.vstack(pval), 'o-', color = 'k')
# plt.errorbar(x - width/2, infos_phase1[infos_phase1['genotype'] == 'KO_male']['DI2'].mean(), yerr= infos_phase1[infos_phase1['genotype'] == 'KO_male']['DI2'].sem(), fmt = 'o', color = 'r', capsize = 5)
# plt.errorbar(x + width/2, infos_phase2[infos_phase2['genotype'] == 'KO_male']['DI2'].mean(), yerr= infos_phase2[infos_phase2['genotype'] == 'KO_male']['DI2'].sem(), fmt = 'o', color = 'r', capsize = 5)
# plt.axhline(0.5, linestyle = '--', color = 'silver')
# # Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_ylabel('Discrimination index')
# ax.set_title('KO male')
# ax.set_xticks(x)
# ax.legend(loc = 'upper left')
# ax.set_box_aspect(1)
# fig.tight_layout()

#%% KO male

# t_kom, p_kom = mannwhitneyu(infos_phase1[infos_phase1['genotype'] == 'KO_male']['DI2'].astype('float64'), infos_phase2[infos_phase2['genotype'] == 'KO_male']['DI2'].astype('float64'))
# z1_kom, p1_kom = wilcoxon(infos_phase1[infos_phase1['genotype'] == 'KO_male']['DI2'].values.astype('float64')-0.5)
# z2_kom, p2_kom = wilcoxon(infos_phase2[infos_phase2['genotype'] == 'KO_male']['DI2'].values.astype('float64')-0.5)
# z_pair_kom, p_pair_kom = wilcoxon(infos_phase1[infos_phase1['genotype'] == 'KO_male']['DI2'].values.astype('float64'), infos_phase2[infos_phase2['genotype'] == 'KO_male']['DI2'].values.astype('float64'))

#%% 

# fig, ax = plt.subplots()
# rects1 = ax.bar(x - width/2, infos_phase1[infos_phase1['genotype'] == 'WT_female']['DI2'].mean(), width, color = 'mediumorchid', label='Encoding Phase')
# rects2 = ax.bar(x + width/2, infos_phase2[infos_phase2['genotype'] == 'WT_female']['DI2'].mean(), width, color = 'violet', label='Recall Phase')
# pval = np.vstack([(infos_phase1[infos_phase1['genotype'] == 'WT_female']['DI2'].values), (infos_phase2[infos_phase2['genotype'] == 'WT_female']['DI2'].values)])
# x2 = [x-width/2, x+width/2]
# plt.plot(x2, np.vstack(pval), 'o-', color = 'k')
# plt.errorbar(x - width/2, infos_phase1[infos_phase1['genotype'] == 'WT_female']['DI2'].mean(), yerr= infos_phase1[infos_phase1['genotype'] == 'WT_female']['DI2'].sem(), fmt = 'o', color = 'r', capsize = 5)
# plt.errorbar(x + width/2, infos_phase2[infos_phase2['genotype'] == 'WT_female']['DI2'].mean(), yerr= infos_phase2[infos_phase2['genotype'] == 'WT_female']['DI2'].sem(), fmt = 'o', color = 'r', capsize = 5)
# plt.axhline(0.5, linestyle = '--', color = 'silver')
# # Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_ylabel('Discrimination index')
# ax.set_title('WT female')
# ax.set_xticks(x)
# ax.legend(loc = 'upper left')
# ax.set_box_aspect(1)
# fig.tight_layout()

#%% WT female

# t_wtf, p_wtf = mannwhitneyu(infos_phase1[infos_phase1['genotype'] == 'WT_female']['DI2'].astype('float64'), infos_phase2[infos_phase2['genotype'] == 'WT_female']['DI2'].astype('float64'))
# z1_wtf, p1_wtf = wilcoxon(infos_phase1[infos_phase1['genotype'] == 'WT_female']['DI2'].values.astype('float64')-0.5)
# z2_wtf, p2_wtf = wilcoxon(infos_phase2[infos_phase2['genotype'] == 'WT_female']['DI2'].values.astype('float64')-0.5)
# z_pair_wtf, p_pair_wtf = wilcoxon(infos_phase1[infos_phase1['genotype'] == 'WT_female']['DI2'].values.astype('float64'), infos_phase2[infos_phase2['genotype'] == 'WT_female']['DI2'].values.astype('float64'))

#%% 

# fig, ax = plt.subplots()
# rects1 = ax.bar(x - width/2, infos_phase1[infos_phase1['genotype'] == 'KO_female']['DI2'].mean(), width, color = 'cadetblue', label='Encoding Phase')
# rects2 = ax.bar(x + width/2, infos_phase2[infos_phase2['genotype'] == 'KO_female']['DI2'].mean(), width, color = 'darkslategray', label='Recall Phase')
# pval = np.vstack([(infos_phase1[infos_phase1['genotype'] == 'KO_female']['DI2'].values), (infos_phase2[infos_phase2['genotype'] == 'KO_female']['DI2'].values)])
# x2 = [x-width/2, x+width/2]
# plt.plot(x2, np.vstack(pval), 'o-', color = 'k')
# plt.errorbar(x - width/2, infos_phase1[infos_phase1['genotype'] == 'KO_female']['DI2'].mean(), yerr= infos_phase1[infos_phase1['genotype'] == 'KO_female']['DI2'].sem(), fmt = 'o', color = 'r', capsize = 5)
# plt.errorbar(x + width/2, infos_phase2[infos_phase2['genotype'] == 'KO_female']['DI2'].mean(), yerr= infos_phase2[infos_phase2['genotype'] == 'KO_female']['DI2'].sem(), fmt = 'o', color = 'r', capsize = 5)
# plt.axhline(0.5, linestyle = '--', color = 'silver')
# # Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_ylabel('Discrimination index')
# ax.set_title('KO female')
# ax.set_xticks(x)
# ax.legend(loc = 'upper left')
# ax.set_box_aspect(1)
# fig.tight_layout()

#%% KO female

# t_kof, p_kof = mannwhitneyu(infos_phase1[infos_phase1['genotype'] == 'KO_female']['DI2'].astype('float64'), infos_phase2[infos_phase2['genotype'] == 'KO_female']['DI2'].astype('float64'))
# z1_kof, p1_kof = wilcoxon(infos_phase1[infos_phase1['genotype'] == 'KO_female']['DI2'].values.astype('float64')-0.5)
# z2_kof, p2_kof = wilcoxon(infos_phase2[infos_phase2['genotype'] == 'KO_female']['DI2'].values.astype('float64')-0.5)
# z_pair_kof, p_pair_kof = wilcoxon(infos_phase1[infos_phase1['genotype'] == 'WT_female']['DI2'].values.astype('float64'), infos_phase2[infos_phase2['genotype'] == 'WT_female']['DI2'].values.astype('float64'))

#%% DI1 boxplots 


plt.figure()
plt.boxplot(info2['DI1'][info2['Genotype'] == 'WT'], positions = [0], showfliers= False, patch_artist=True,boxprops=dict(facecolor='royalblue', color='royalblue'),
            capprops=dict(color='royalblue'),
            whiskerprops=dict(color='royalblue'),
            medianprops=dict(color='white', linewidth = 2))
plt.boxplot(info2['DI1'][info2['Genotype'] == 'KO'], positions = [0.3], showfliers= False, patch_artist=True,boxprops=dict(facecolor='indianred', color='indianred'),
            capprops=dict(color='indianred'),
            whiskerprops=dict(color='indianred'),
            medianprops=dict(color='white', linewidth = 2))

x1 = np.random.normal(0, 0.01, size=len(info2['DI1'][info2['Genotype'] == 'WT'][info2['Sex'] == 'M']))
x2 = np.random.normal(0, 0.01, size=len(info2['DI1'][info2['Genotype'] == 'WT'][info2['Sex'] == 'F']))
x3 = np.random.normal(0.3, 0.01, size=len(info2['DI1'][info2['Genotype'] == 'KO'][info2['Sex'] == 'M']))
x4 = np.random.normal(0.3, 0.01, size=len(info2['DI1'][info2['Genotype'] == 'KO'][info2['Sex'] == 'F']))
                      
plt.plot(x1, info2['DI1'][info2['Genotype'] == 'WT'][info2['Sex'] == 'M'], '.', color = 'k', fillstyle = 'none', markersize = 8, zorder =3, label = 'male')
plt.plot(x2, info2['DI1'][info2['Genotype'] == 'WT'][info2['Sex'] == 'F'], '.', color = 'k', fillstyle = 'full', markersize = 8, zorder =3, label = 'female')
plt.plot(x3, info2['DI1'][info2['Genotype'] == 'KO'][info2['Sex'] == 'M'], '.', color = 'k', fillstyle = 'none', markersize = 8, zorder =3)
plt.plot(x4, info2['DI1'][info2['Genotype'] == 'KO'][info2['Sex'] == 'F'], '.', color = 'k', fillstyle = 'full', markersize = 8, zorder =3)

plt.ylabel('Discrimination Index')
plt.axhline(0, linestyle = '--',  color = 'silver')
plt.legend(loc = 'upper left')
plt.xticks([0, 0.3],['WT', 'KO'])
plt.gca().set_box_aspect(1)

z_wt, p_wt = wilcoxon(np.array(info2['DI1'][info2['Genotype'] == 'WT'])-0)
z_ko, p_ko = wilcoxon(np.array(info2['DI1'][info2['Genotype'] == 'KO'])-0)

#%% 

# plt.figure()
# plt.boxplot(info2['DI2'][info2['Genotype'] == 'WT'], positions = [0], showfliers= False, patch_artist=True,boxprops=dict(facecolor='royalblue', color='royalblue'),
#             capprops=dict(color='royalblue'),
#             whiskerprops=dict(color='royalblue'),
#             medianprops=dict(color='white', linewidth = 2))
# plt.boxplot(info2['DI2'][info2['Genotype'] == 'KO'], positions = [0.3], showfliers= False, patch_artist=True,boxprops=dict(facecolor='indianred', color='indianred'),
#             capprops=dict(color='indianred'),
#             whiskerprops=dict(color='indianred'),
#             medianprops=dict(color='white', linewidth = 2))

# x1 = np.random.normal(0, 0.01, size=len(info2['DI2'][info2['Genotype'] == 'WT'][info2['Sex'] == 'M']))
# x2 = np.random.normal(0, 0.01, size=len(info2['DI2'][info2['Genotype'] == 'WT'][info2['Sex'] == 'F']))
# x3 = np.random.normal(0.3, 0.01, size=len(info2['DI2'][info2['Genotype'] == 'KO'][info2['Sex'] == 'M']))
# x4 = np.random.normal(0.3, 0.01, size=len(info2['DI2'][info2['Genotype'] == 'KO'][info2['Sex'] == 'F']))
                      
# plt.plot(x1, info2['DI2'][info2['Genotype'] == 'WT'][info2['Sex'] == 'M'], '.', color = 'k', fillstyle = 'none', markersize = 8, zorder =3, label = 'male')
# plt.plot(x2, info2['DI2'][info2['Genotype'] == 'WT'][info2['Sex'] == 'F'], '.', color = 'k', fillstyle = 'full', markersize = 8, zorder =3, label = 'female')
# plt.plot(x3, info2['DI2'][info2['Genotype'] == 'KO'][info2['Sex'] == 'M'], '.', color = 'k', fillstyle = 'none', markersize = 8, zorder =3)
# plt.plot(x4, info2['DI2'][info2['Genotype'] == 'KO'][info2['Sex'] == 'F'], '.', color = 'k', fillstyle = 'full', markersize = 8, zorder =3)

# plt.ylabel('Discrimination Index')
# plt.axhline(0.5, linestyle = '--',  color = 'silver')
# plt.legend(loc = 'upper left')
# plt.xticks([0, 0.3],['WT', 'KO'])
# plt.ylim([0,1])
# plt.yticks([0, 0.25, 0.5, 0.75, 1])
# plt.gca().set_box_aspect(1)

# z_wt, p_wt = wilcoxon(np.array(info2['DI2'][info2['Genotype'] == 'WT'])-0)
# z_ko, p_ko = wilcoxon(np.array(info2['DI2'][info2['Genotype'] == 'KO'])-0)

#%% 

# label = ['WT male']
# x = np.arange(len(label))  # the label locations
# width = 0.35

# plt.figure()
# plt.subplot(121)
# pval = np.vstack([info2['DI1'][info2['Genotype'] == 'WT'][info2['Sex'] == 'M'], info2['DI2'][info2['Genotype'] == 'WT'][info2['Sex'] == 'M']])
# x2 = [x-width/2, x+width/2]
# plt.plot(x2, np.vstack(pval), 'o-', color = 'k')
# plt.ylim([-1,1])

# plt.subplot(122)
# pval = np.vstack([info2['DI1'][info2['Genotype'] == 'KO'][info2['Sex'] == 'M'], info2['DI2'][info2['Genotype'] == 'KO'][info2['Sex'] == 'M']])
# x2 = [x-width/2, x+width/2]
# plt.plot(x2, np.vstack(pval), 'o-', color = 'k')
# plt.ylim([-1,1])
