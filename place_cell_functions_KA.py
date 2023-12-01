# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 00:36:10 2023

@author: kasum
"""

import numpy as np
from numba import jit
import pandas as pd
import neuroseries as nts
import sys
import scipy
from sklearn.manifold import Isomap
from matplotlib.colors import hsv_to_rgb
from pylab import *
from scipy.stats import circmean
from scipy.stats import circvar
from scipy.ndimage import gaussian_filter
from itertools import combinations
from pycircstat.descriptive import mean as circmean2
import astropy
import warnings
import pickle
from sklearn.preprocessing import StandardScaler
from scipy.signal import fftconvolve



def remove_box(num=2, frame=True):
    if num >2:
        gca().spines['right'].set_visible(False)
        gca().spines['left'].set_visible(False)
        gca().spines['top'].set_visible(False)
        gca().spines['bottom'].set_visible(False)
    else:
        gca().spines['right'].set_visible(False)
        gca().spines['top'].set_visible(False)
    if frame:
        gca().set_xticks([])
        gca().set_yticks([])
        
        

def path_plot(eps,position):
    #fig=figure(figsize=(15,16))
    fig=figure()
    for i in range(len(eps)):
        if len(eps)==1:
            ax=subplot()
        else:    
            ax=subplot(1,len(eps),i+1)
        ep=eps.iloc[i]
        ep=nts.IntervalSet(ep[0],ep[1])
        plot(position['x'].restrict(ep),position['z'].restrict(ep),color='red',label=str(i), alpha=0.5) 
        legend()


def shuffleByCircularSpikes(spikes, ep):
    shuffled = {}
    for n in spikes.keys():
        spk = spikes[n].restrict(ep)
        shift = np.random.uniform(20 * 1000000, (ep.loc[0, 'end'] - ep.loc[0, 'start'] - 20) * 1000000)
        spk_shifted = (spk.index.values + shift) % (ep.loc[0, 'end'] - ep.loc[0, 'start']) + ep.loc[0, 'start']
        shuffled[n] = nts.Ts(t=spk_shifted)
    return shuffled


def all_frate_maps(spikes,position,ep,cells, size= 24):
    """Returns the smooth rate map and masks unexplored bins
    """    
    tms={}
    GF, ext,occ = computePlaceFields(spikes, position[['x', 'z']], ep, size)#using 2.5 by 2.5cm set to 24
    for i,k in enumerate(cells):
       tms_tmp = gaussian_filter(GF[k].values,sigma = 1.5, mode = "nearest")
       masked_array = np.ma.masked_where(occ == 0, tms_tmp) #should work fine without it
       tms[i] = masked_array
    return tms


def computePlaceFields(spikes, position, ep, nb_bins = 100, frequency = 120.0):
    place_fields = {}
    position_tsd = position.restrict(ep)
    xpos = position_tsd.iloc[:,0]
    ypos = position_tsd.iloc[:,1]
    xbins = np.linspace(xpos.min(), xpos.max()+1e-6, nb_bins+1)
    ybins = np.linspace(ypos.min(), ypos.max()+1e-6, nb_bins+1) 
    occupancy, _, _ = np.histogram2d(ypos, xpos, [ybins,xbins])
    norm_occu = occupancy/sum(occupancy)
    for n in spikes:
        position_spike = position_tsd.realign(spikes[n].restrict(ep))
        spike_count,_,_ = np.histogram2d(position_spike.iloc[:,1].values, position_spike.iloc[:,0].values, [ybins,xbins])
        mean_spike_count = spike_count/(occupancy+1)
        place_field = mean_spike_count*frequency    
        place_fields[n] = pd.DataFrame(index = ybins[0:-1][::-1],columns = xbins[0:-1], data = place_field)
    extent = (xbins[0], xbins[-1], ybins[0], ybins[-1]) # USEFUL FOR MATPLOTLIB
    return place_fields, extent, norm_occu

def pc_stability(spikes,pos,ep):
    r=pd.DataFrame(index=spikes.keys(), columns=['spatial_corr','pval_corr'])
    
    ep1_start=ep.start[0]; ep1_end=ep.start[0]+(diff(ep)[0][0]/2)        
    ep2_start=ep1_end; ep2_end=ep.end[0]
    
    ep1=nts.IntervalSet(ep1_start,ep1_end)
    ep2=nts.IntervalSet(ep2_start,ep2_end)
    
    map1 = all_frate_maps(spikes, pos,ep1,spikes.keys())
    map2 = all_frate_maps(spikes, pos,ep2,spikes.keys())
    
    for k in spikes.keys():
        r.loc[k,['spatial_corr', 'pval_corr']]=scipy.stats.pearsonr(map1[k].flatten(),map2[k].flatten())        
    return r



def pc_sessionStability(spikes, pos, eps):
    r=pd.DataFrame(index=spikes.keys(), columns=['spatial_corr','pval_corr'])
    
    ep1_s = eps.start[0]; ep1_e =eps.end[0]
    ep1 = nts.IntervalSet(ep1_s,ep1_e)
    
    ep2_s = eps.start[1]; ep2_e =eps.end[1]
    ep2 = nts.IntervalSet(ep2_s,ep2_e)
    
    map1 = all_frate_maps(spikes, pos,ep1,spikes.keys())
    map2 = all_frate_maps(spikes, pos,ep2,spikes.keys())
    
    for k in spikes.keys():
        r.loc[k,['spatial_corr', 'pval_corr']]=scipy.stats.pearsonr(map1[k].flatten(),map2[k].flatten())        
    return r
    
    
    
    
    
    
    
    
    

def compute_2d_mutual_info(tc, features, ep, bitssec=False):
    """
    Mutual information as defined in 
        
    Skaggs, W. E., McNaughton, B. L., & Gothard, K. M. (1993). 
    An information-theoretic approach to deciphering the hippocampal code. 
    In Advances in neural information processing systems (pp. 1030-1037).

    Parameters
    ----------
    tc : dict or numpy.ndarray
        If array, first dimension should be the neuron
    features : TsdFrame
        The 2 columns features that were used to compute the tuning curves
    ep : IntervalSet, optional
        The epoch over which the tuning curves were computed
        If None, the epoch is the time support of the feature.
    bitssec: bool, optional
        By default, the function return bits per spikes.
        Set to true for bits per seconds

    Returns
    -------
    pandas.DataFrame
        Spatial Information (default is bits/spikes)
    """
    fx=np.array([tc[i] for i in tc.keys()])
    idx=list(tc.keys())
    
    nb_bins = (fx.shape[1]+1,fx.shape[2]+1)

    cols = features.columns

    bins = []
    for i, c in enumerate(cols):
        bins.append(np.linspace(np.min(features[c]), np.max(features[c]), nb_bins[i]))
           
    features = features.restrict(ep)

    occupancy, _, _ = np.histogram2d(features[cols[0]].values, features[cols[1]].values, [bins[0], bins[1]])
    occupancy = occupancy / occupancy.sum()

    fr = np.nansum(fx * occupancy, (1,2))
    fr = fr[:,np.newaxis,np.newaxis]
    fxfr = fx/fr
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        logfx = np.log2(fxfr)        
    logfx[np.isinf(logfx)] = 0.0
    SI = np.nansum(occupancy * fx * logfx, (1,2))

    if bitssec:
        SI = pd.DataFrame(index = idx, columns = ['info'], data = SI)    
        return SI
    else:
        SI = SI / fr[:,0,0]
        SI = pd.DataFrame(index = idx, columns = ['info'], data = SI)
        return SI 


def pc_frates(spikes,rate_map, ep):
    rates = pd.DataFrame(index=spikes.keys(), columns= ['mean','peak'])  
    for i in spikes.keys():
        rates.loc[i,'mean'] = len(spikes[i].restrict(ep))/ep.tot_length('s')
        rates.loc[i,'peak'] = rate_map[i].max()
    return rates


def find_place_cells(rate_map, spikes,position,ep):
        
    corr = pc_stability(spikes, position, ep)
    rates = pc_frates(spikes,rate_map,ep)
    si = compute_2d_mutual_info(rate_map, position[['x','z']], ep)
    
    # crit1 = corr['spatial_corr'] > -0.1
    crit2 = rates['peak'] >= 1
    crit3 = np.logical_and(0.15 < rates['mean'],rates['mean'] <= 7) 
    crit4 =si['info'] >= 0.27 #corresponds to 95th percentile of the shuffles[0.27]
    
    thres = crit2 & crit3 & crit4

    pcs = list(rates.index[thres].values)
    stats = pd.concat([rates, corr['spatial_corr'], si, thres],axis=1)

    return pcs, stats


def selectivity(rate_map, px):
    '''
    "The selectivity measure max(rate)/mean(rate)  of the cell. The more
    tightly concentrated  the cell's activity, the higher the selectivity.
    A cell with no spatial tuning at all will  have a  selectivity of 1" [2]_.

    Parameters
    ----------
    rate_map : numpy.ndarray
        A firing rate map, any number of dimensions.

    Returns
    -------
    out : float
        selectivity
    '''
    tmp_rate_map = rate_map.copy()
    tmp_rate_map[np.isnan(tmp_rate_map)] = 0
    avg_rate = np.sum(np.ravel(tmp_rate_map * px))
    max_rate = np.max(np.ravel(tmp_rate_map))
    return max_rate / avg_rate


def sparsity(rate_map, px):
    '''
    Compute sparsity of a rate map, The sparsity  measure is an adaptation
    to space. The adaptation measures the fraction of the environment  in which
    a cell is  active. A sparsity of, 0.1 means that the place field of the
    cell occupies 1/10 of the area the subject traverses [2]_

    Parameters
    ----------
    rate_map : normalized numpy.ndarray
        A firing rate map, any number of dimensions.

    Returns
    -------
    out : float
        sparsity

    References
    ----------
    .. [2] Skaggs, W. E., McNaughton, B. L., Wilson, M., & Barnes, C. (1996).
       Theta phase precession in hippocampal neuronal populations and the
       compression of temporal sequences. Hippocampus, 6, 149-172.
    '''
    tmp_rate_map = rate_map.copy()
    tmp_rate_map[np.isnan(tmp_rate_map)] = 0
    avg_rate = np.sum(np.ravel(tmp_rate_map * px))
    avg_sqr_rate = np.sum(np.ravel(tmp_rate_map**2 * px))
    return avg_rate**2 / avg_sqr_rate


def compute_sparsity(all_maps,position,ep):
    sparsity_dat = pd.DataFrame(index=all_maps.keys(), columns=['sparsity'])
    px = occupancy_prob(position,ep)
    for i in all_maps.keys():
        tmp_sparsity = sparsity(all_maps[i],px)
        sparsity_dat.loc[i,'sparsity'] = tmp_sparsity
    return sparsity_dat


def compute_selectivity(all_maps,position,ep):
    selectivity_dat = pd.DataFrame(index=all_maps.keys(), columns=['selectivity'])
    px = occupancy_prob(position,ep)
    for i in all_maps.keys():
        tmp_selectivity = selectivity(all_maps[i],px)
        selectivity_dat.loc[i,'selectivity'] = tmp_selectivity
    return selectivity_dat
        
    
def occupancy_prob(position,ep,nb_bins=24):
    pos= position[['x','z']]
    position_tsd = pos.restrict(ep)
    xpos = position_tsd.iloc[:,0]
    ypos = position_tsd.iloc[:,1]
    xbins = np.linspace(xpos.min(), xpos.max()+1e-6, nb_bins+1)
    ybins = np.linspace(ypos.min(), ypos.max()+1e-6, nb_bins+1) 
    occupancy, _, _ = np.histogram2d(ypos, xpos, [ybins,xbins])
    norm_occu = gaussian_filter(occupancy/sum(occupancy),sigma = 1.5, mode = "nearest")
    masked_array = np.ma.masked_where(occupancy == 0, norm_occu) 
    masked_array = np.flipud(masked_array)
    return masked_array




def spatial_autocorrelation(all_maps):
    """Compute the spatial autocorrelation of a place cell's firing rate map.

    Args:
        rate_map (ndarray): A 2D array representing the firing rate of the place cell at each location in the environment.

    Returns:
        ndarray: A 2D array representing the spatial autocorrelation of the place cell's firing rate map.
        
    """
    spatial_autocorr = {}
    
    for idx in all_maps:
        rate_map = all_maps[idx]

        # Compute the mean firing rate of the place cell.
        mean_rate = np.mean(rate_map)
    
        # Compute the deviation of the firing rate from the mean.
        rate_deviation = rate_map - mean_rate
    
        # Compute the autocorrelation of the rate deviation using the FFT method.
        autocorrelation = fftconvolve(rate_deviation, np.flip(rate_deviation), mode='same')
    
        # Compute the normalization factor for the spatial autocorrelation.
        normalization = np.sum(rate_deviation**2)
    
        # Normalize the autocorrelation.
        spatial_autocorrelation = autocorrelation / normalization
        
        spatial_autocorr[idx] = spatial_autocorrelation

    return spatial_autocorr


def compute_population_vectors(rate_maps,cell_ids):
    # Get the dimensions of the rate maps
    num_cells = len(cell_ids)
    num_bins_x, num_bins_y = rate_maps[list(rate_maps.keys())[0]].shape

    # Create a 3D array to store the population vectors
    population_vectors = np.zeros((num_bins_x, num_bins_y, num_cells))

    # Populate the population vectors for each cell
    for i, cell_id in enumerate(cell_ids):
        population_vectors[:, :, i] = rate_maps[cell_id]

    # Compute the mean firing rate across cells for each spatial bin
    mean_rates = np.mean(population_vectors, axis=2)

    return mean_rates


def population_vector_correlation(pv1, pv2):
    """
    Computes the population vector correlation and correlation coefficient
    between two input population vectors.
    
    Args:
    pv1 (numpy.ndarray): The first population vector, represented as a 1D numpy array.
    pv2 (numpy.ndarray): The second population vector, represented as a 1D numpy array.
    
    Returns:
    A tuple containing the population vector correlation and the correlation coefficient.
    """
    assert pv1.shape == pv2.shape, "Population vectors must be of the same length."
        
        # Reshape the input matrices into 1D vectors
    pv1 = pv1.reshape(-1)
    pv2 = pv2.reshape(-1)
    
    # Compute the mean firing rates of each population vector
    mean_rate1 = np.mean(pv1)
    mean_rate2 = np.mean(pv2)
    
    # Subtract the mean firing rates from each population vector
    pv1 -= mean_rate1
    pv2 -= mean_rate2
    
    # Compute the dot product between the two population vectors
    dot_product = np.dot(pv1, pv2)
    
    # Compute the magnitudes of each population vector
    magnitude1 = np.sqrt(np.sum(pv1 ** 2))
    magnitude2 = np.sqrt(np.sum(pv2 ** 2))
    
    # Compute the population vector correlation
    pvc = dot_product / (magnitude1 * magnitude2)
    
    # Compute the correlation coefficient
    corr_coef = np.corrcoef(pv1, pv2)[0, 1]
    
    return pvc, corr_coef


def compute_PVcorrs (all_rates1,all_rates2,cell_ids):
    """returns the pv corr for two population vectors"""
    
    pv1 = compute_population_vectors(all_rates1,cell_ids)
    pv2 = compute_population_vectors(all_rates2,cell_ids)
    pvCorr = population_vector_correlation(pv1, pv2)[0]
    
    return pvCorr
    
    

def calculate_coherence(rate_map):
    """
    Calculates the coherence of a place cell's firing patterns given a normalized rate map.
    
    Parameters:
    rate_map (numpy.ndarray): A 2D numpy array representing the normalized firing rate map of the place cell.
    
    Returns:
    coherence (float): The coherence of the place cell's firing patterns, calculated using the spatial information content.
    """
    # Calculate the probability distribution of firing rates
    p = rate_map / np.sum(rate_map)
    
    # Calculate the mean firing rate
    pmean = np.mean(p)
    
    # Calculate the spatial information content
    SIC = np.sum(p * np.log2(p / pmean))
    
    # Calculate the coherence
    coherence = 1 - (SIC / np.log2(rate_map.size))
    
    return coherence

    



















