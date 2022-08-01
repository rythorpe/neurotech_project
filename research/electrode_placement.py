"""Efficient way to optimize electrode placement in reduced electrode array"""

import numpy as np
from scipy.io import loadmat
from scipy import stats
from scipy import ndimage
from scipy.special import comb
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


def binarized_data(timeseries, thresh):
    """Threshold and convert each electrode channel to a binary string"""
    n_chan = timeseries.shape[0]

    binary_strings = list()
    for chan_idx in range(n_chan):
        chan_timeseries = timeseries[chan_idx, :]
        chan_thresh = (abs(chan_timeseries) > thresh).astype(int).astype(str)
        binary_strings.append(''.join(chan_thresh.tolist()))

    return binary_strings


def get_combinations(element_idxs, combination_size):
    """Compute all combinations recursively"""

    combs = list()

    for el_idx in element_idxs:
        if combination_size > 1:
            # find the compliment subset
            other_element_idxs = [other_idx for other_idx in element_idxs if
                                  other_idx > el_idx]
            # get all sub-combinations of the compliment set
            sub_combs = get_combinations(other_element_idxs,
                                         combination_size - 1)
            # append each sub-combination to a copy of the current item
            for sub_comb in sub_combs:
                combs.append([el_idx] + sub_comb)
        else:
            combs.append([el_idx])

    return combs


def opt_channel_subset(data_binary, n_chan_subset):
    """Find optimal subset of electrode channels using brute force approach"""
    data_binary = np.array(data_binary)
    n_chan = data_binary.shape[0]

    idx_combinations = get_combinations(range(n_chan), n_chan_subset)
    complexities = np.zeros(len(idx_combinations))

    for idx, combination in enumerate(idx_combinations):
        selected_channels = data_binary[combination].tolist()
        complexities[idx], _ = lempel_ziv_complexity(selected_channels)

    sorting_idxs = np.argsort(complexities)

    return complexities[sorting_idxs], complexities[sorting_idxs]


def est_channel_subset(data_binary, n_chan_subset):
    """Estimate optimal subset of electrode channels using fast approach"""
    n_chan = len(data_binary)

    chan_idxs, complexities = None, None

    return chan_idxs, complexities


def plot_electrode_complexities(complexities, ax=None, labels=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    else:
        fig = ax.get_figure()
    ax.plot(range(len(complexities)), complexities)
    ax.set_ylabel('complexity index')
    ax.set_xlabel('electrode configuration')

    return fig, ax


if __name__ == "__main__":
    data_fname = '/home/ryan/Desktop/dataset_processing/EEG.mat'
    data = loadmat(data_fname)
    f_samp = data['EEG']['srate'][0][0].squeeze()  # sampling rate
    data = data['EEG']['data'][0][0]

    data_normalized = normalize_data(data)

    std_thresh = 3.
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
    std_thresh = 1.5
    data_binary = binarized_data(timeseries=data_post_perturbation,
                                 thresh=std_thresh)

    n_chans = range(4, 6)
    fig, ax = plt.subplots(1, 1)
    for n_chan_subset in n_chans:
        opt_chans, opt_compl = opt_channel_subset(data_binary=data_binary,
                                                  n_chan_subset=n_chan_subset)
    #est_chans, est_compl = est_channel_subset(data_binary=data_binary,
    #                                          n_chan_subset=n_chan_subset)

        fig, ax = plot_electrode_complexities(opt_compl, ax=ax)
    labels = [f'{n_chan_subset} electrodes' for n_chan_subset in n_chans]
    ax.legend(labels)

    plt.show()
