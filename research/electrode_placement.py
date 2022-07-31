"""Efficient way to optimize electrode placement in reduced electrode array"""

import numpy as np
from scipy.io import loadmat
from scipy import stats
from scipy import ndimage
import matplotlib.pyplot as plt
from lempel_ziv_complexity import lempel_ziv_complexity


def normalize_data(data):
    """Z-score each channel independently"""

    return stats.zscore(data, axis=1)


def find_perturbations(timeseries, thresh):
    """Find locations of intrinsic high-amplitude timeseries perturbations"""
    chan_idxs, time_idxs = np.nonzero(abs(timeseries) > thresh)

    mask = np.zeros_like(timeseries, dtype=int)
    mask[chan_idxs, time_idxs] = 1
    labels, n_peaks = ndimage.label(mask)
    peak_positions = ndimage.maximum_position(abs(timeseries), labels=labels,
                                              index=range(n_peaks))
    _, perturbation_idxs = zip(*peak_positions)

    return sorted(perturbation_idxs)


def get_data_windows(timeseries, win_beginnings, win_length):
    """Get windows of data across all channels beginning at each onset time"""

    # prevent overlapping windows
    previous = [win_beginnings[0]]
    for win_beginning in win_beginnings[1:]:
        if (win_beginning > previous[-1] + win_length and
                win_beginning + win_length <= timeseries.shape[1]):
            previous.append(win_beginning)

    data_windows = np.stack([timeseries[:, win_beginning] for win_beginning in
                             previous]).T

    return data_windows


def binarize_data(timeseries, thresh):
    """Threshold and convert each electrode channel to a binary string"""
    n_chan = timeseries.shape[0]

    binary_strings = list()
    for chan_idx in range(n_chan):
        chan_timeseries = timeseries[chan_idx, :]
        chan_thresh = (abs(chan_timeseries) > thresh).astype(int).astype(str)
        binary_strings.append(''.join(chan_thresh.tolist()))

    return binary_strings


def opt_channel_subset(data_binary, n_chan_subset):
    """Find optimal subset of electrode channels using brute force approach"""
    n_chan = len(data_binary)

    for chan_idx, binary_str in enumerate(data_binary):
        while n_chan_subset > 0:
            compliment_chans = list(range(chan_idx))
            opt_channel_subset(data_binary)

    chan_idxs, complexities = None, None

    return chan_idxs, complexities


def est_channel_subset(data_binary, n_chan_subset):
    """Estimate optimal subset of electrode channels using fast approach"""
    n_chan = len(data_binary)

    chan_idxs, complexities = None, None

    return chan_idxs, complexities


def plot_electrode_complexities(placements, complexities):
    fig, ax = plt.subplots(1, 1)

    return fig


if __name__ == "__main__":
    data_fname = '/home/ryan/Desktop/dataset_processing/EEG.mat'
    data = loadmat(data_fname)
    f_samp = data['EEG']['srate'][0][0].squeeze()  # sampling rate
    data = data['EEG']['data'][0][0]

    data_normalized = normalize_data(data)

    std_thresh = 4.
    perturbation_idxs = find_perturbations(timeseries=data_normalized,
                                           thresh=std_thresh)

    # sanity check: plot data alongside the identified "perturbation" peaks
    # fig, ax = plt.subplots(1, 1)
    # ax.pcolormesh(abs(data_normalized), vmin=std_thresh)
    # ax.vlines(perturbation_idxs, -1., 1.)

    win_length = f_samp * .5  # number of samples in 500 ms
    data_post_perturbation = get_data_windows(timeseries=data_normalized,
                                              win_beginnings=perturbation_idxs,
                                              win_length=win_length)
    std_thresh = 2.
    data_binary = binarize_data(timeseries=data_post_perturbation,
                                thresh=std_thresh)

    n_chan_subset = 4
    opt_chan, opt_compl = opt_channel_subset(data_binary=data_binary,
                                             n_chan_subset=n_chan_subset)
    est_chan, est_compl = est_channel_subset(data_binary=data_binary,
                                             n_chan_subset=n_chan_subset)

    fig = plot_electrode_complexities(opt_chan, opt_compl)
    plt.show()
