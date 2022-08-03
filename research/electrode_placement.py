"""Efficient way to optimize electrode placement in reduced electrode array"""

import numpy as np
from scipy.io import loadmat
from scipy import stats
from scipy import ndimage
import matplotlib.pyplot as plt
from lempel_ziv_complexity import lempel_ziv_complexity, normalization_factor


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

    combinations = get_combinations(range(n_chan), n_chan_subset)
    complexities = np.zeros(len(combinations))

    for idx, combination in enumerate(combinations):
        selected_channels = data_binary[combination].tolist()
        complexity, _ = lempel_ziv_complexity(selected_channels)
        complexity *= normalization_factor(selected_channels)  # XXX fix
        complexities[idx] = complexity

    sorting_idxs = np.argsort(complexities)
    best_chan_subset = combinations[sorting_idxs[-1]]

    return best_chan_subset, complexities[sorting_idxs]


def est_channel_subset(data_binary, n_chan_subset, lr=0.05, epsilon=0.1,
                       max_iter=10000):
    """Estimate optimal subset of electrode channels using quasi e-greedy MAB"""
    data_binary = np.array(data_binary)
    n_chan = data_binary.shape[0]

    channels = np.arange(n_chan)
    reward_prob = np.ones((n_chan_subset, n_chan)) / n_chan  # uniform distr.
    chans_chosen = np.arange(n_chan_subset)  # initialize channel subset

    # compute initial complexity index
    chans_chosen_data = data_binary[chans_chosen].tolist()
    max_complexity, _ = lempel_ziv_complexity(chans_chosen_data)
    max_complexity *= normalization_factor(chans_chosen_data)  # XXX fix

    complexities = np.zeros(max_iter)

    for iter_idx in range(max_iter):
        # select channels
        for choice_idx in range(n_chan_subset):
            previous_choice = chans_chosen[choice_idx]
            # channels that don't already occupy a slot in the subset
            chans_avail = [chan for chan in channels if (chan is
                           previous_choice) or (chan not in chans_chosen)]

            # use best strategy: choose most rewarding channel that is 
            # available
            if np.random.random() > epsilon:
                choices_ranked = np.argsort(reward_prob[choice_idx])
                for choice in choices_ranked:
                    if choice in chans_avail:
                        chan_choice = choice
                chans_chosen[choice_idx] = chan_choice

            # random exploration
            else:
                chan_avail_idx = np.random.randint(len(chans_avail))
                chan_choice = chans_avail[chan_avail_idx]
                chans_chosen[choice_idx] = chan_choice

            # compute complexity index
            chans_chosen_data = data_binary[chans_chosen].tolist()
            complexity, _ = lempel_ziv_complexity(chans_chosen_data)
            complexity *= normalization_factor(chans_chosen_data)  # XXX fix
            complexities[iter_idx] = complexity

            if complexity > max_complexity:
                reward_prob[choice_idx, chans_chosen[choice_idx]] += lr
            reward_prob[choice_idx, :] /= np.sum(reward_prob[choice_idx, :])  # normalize
            max_complexity = complexity

    best_chan_subset = np.argmax(reward_prob, axis=1)

    return best_chan_subset, complexities


def plot_electrode_complexities(complexities, ax=None, labels=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    else:
        fig = ax.get_figure()
    ax.plot(range(len(complexities)), complexities)
    ax.set_ylabel('complexity index')
    ax.set_xlabel('iteration')
    fig.show(False)
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

    n_chans = range(4, 5)
    fig, ax = plt.subplots(1, 1)
    for n_chan_subset in n_chans:
        opt_chans, opt_compl = opt_channel_subset(data_binary=data_binary,
                                                  n_chan_subset=n_chan_subset)
        est_chans, est_compl = est_channel_subset(data_binary=data_binary,
                                                  n_chan_subset=n_chan_subset,
                                                  lr=0.05,
                                                  epsilon=0.01,
                                                  max_iter=1000)

        plot_electrode_complexities(opt_compl, ax=ax)
        plot_electrode_complexities(est_compl, ax=ax)
    labels = [f'{n_chan_subset} electrodes' for n_chan_subset in n_chans]
    ax.legend(labels)

    plt.show()
