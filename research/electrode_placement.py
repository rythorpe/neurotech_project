"""Efficient way to optimize electrode placement in reduced electrode array"""

import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from lempel_ziv_complexity import lempel_ziv_complexity


def optimal_placement(elec_data):
    """Find ground truth optimal placement using brute force approach"""
    n_elec = elec_data.shape[0]


def est_optimal_placement(elec_data):
    """Estimate optimal placement using fast approach"""
    pass


def plot_electrode_ci(data):
    fig, ax = plt.subplots((1, 1))

    return fig


if __name__ == "__main__":
    data_fname = '/home/ryan/Desktop/dataset_processing/EEG.mat'
    data = loadmat(data_fname)
    data = data['EEG']['data'][0][0]

    opt_place = optimal_placement(data)
    est_place = est_optimal_placement(data)

    
