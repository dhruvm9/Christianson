import numpy as np 
import pandas as pd 
import nwbmatic as ntm
import scipy.io
import pynapple as nap 
import os, sys
import matplotlib.pyplot as plt 
import seaborn as sns
import warnings
from scipy.signal import butter, lfilter, filtfilt
from scipy.stats import mannwhitneyu, wilcoxon
from matplotlib.backends.backend_pdf import PdfPages    

#%% 

def multipage(filename, figs=None, dpi=200):
    pp = PdfPages(filename)
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()
    
    
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
    occupancy, _, _ = np.histogram2d(xpos, ypos, [xbins,ybins])
    
    if norm is True:
        occupancy = occupancy/np.sum(occupancy)
        
    masked_array = np.ma.masked_where(occupancy == 0, occupancy) 
    # masked_array = np.flipud(masked_array)
    return masked_array

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
    tmp_rate_map[np.isinf(tmp_rate_map)] = 0
    avg_rate = np.sum(np.ravel(tmp_rate_map * px))
    avg_sqr_rate = np.sum(np.ravel(tmp_rate_map**2 * px))
    return avg_rate**2 / avg_sqr_rate


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



def _butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def _butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = _butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def _butter_bandpass_zerophase_filter(data, lowcut, highcut, fs, order=4):
    b, a = _butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    """
    Bandpass filtering the LFP.
    
    Parameters
    ----------
    data : Tsd/TsdFrame
        Description
    lowcut : TYPE
        Description
    highcut : TYPE
        Description
    fs : TYPE
        Description
    order : int, optional
        Description
    
    Raises
    ------
    RuntimeError
        Description
    """
    time_support = data.time_support
    time_index = data.as_units('s').index.values
    if type(data) is nap.TsdFrame:
        tmp = np.zeros(data.shape)
        for i,c in enumerate(data.columns):
            tmp[:,i] = bandpass_filter(data[c], lowcut, highcut, fs, order)

        return nap.TsdFrame(
            t = time_index,
            d = tmp,
            time_support = time_support,
            time_units = 's',
            columns = data.columns)

    elif type(data) is nap.Tsd:
        flfp = _butter_bandpass_filter(data.values, lowcut, highcut, fs, order)
        return nap.Tsd(
            t=time_index,
            d=flfp,
            time_support=time_support,
            time_units='s')

    else:
        raise RuntimeError("Unknown format. Should be Tsd/TsdFrame")
        
def bandpass_filter_zerophase(data, lowcut, highcut, fs, order=4):
    """
    Bandpass filtering the LFP.
    
    Parameters
    ----------
    data : Tsd/TsdFrame
        Description
    lowcut : TYPE
        Description
    highcut : TYPE
        Description
    fs : TYPE
        Description
    order : int, optional
        Description
    
    Raises
    ------
    RuntimeError
        Description
    """
    time_support = data.time_support
    time_index = data.as_units('s').index.values
    if type(data) is nap.TsdFrame:
        tmp = np.zeros(data.shape)
        for i,c in enumerate(data.columns):
            tmp[:,i] = bandpass_filter(data[c], lowcut, highcut, fs, order)

        return nap.TsdFrame(
            t = time_index,
            d = tmp,
            time_support = time_support,
            time_units = 's',
            columns = data.columns)

    elif type(data) is nap.Tsd:
        flfp = _butter_bandpass_zerophase_filter(data.values, lowcut, highcut, fs, order)
        return nap.Tsd(
            t=time_index,
            d=flfp,
            time_support=time_support,
            time_units='s')

    else:
        raise RuntimeError("Unknown format. Should be Tsd/TsdFrame")