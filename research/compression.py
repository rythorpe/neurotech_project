import numpy as np

from data import datasets
from typing import List
from lempel_ziv_complexity import lempel_ziv_complexity
from scipy import stats
from tqdm import tqdm
import matplotlib.pyplot as plt

def normalize_data(samples):
    """
    For each frequency band, compute its own z-score
    """

    normalized_output = np.zeros_like(samples)
    for freq in range(samples.shape[1]):
        single_freq_data = samples[:, freq]
        normalized_output[:, freq] = stats.zscore(single_freq_data)

    return normalized_output


def bin_sample(normalized_sample, thresholds:List[float]):
    return np.stack([(abs(normalized_sample) >= thresh).astype(int) for thresh in thresholds])


def data_matrices_to_string(data_matrices: np.ndarray):
    # stupid way for now
    return ''.join(map(str, data_matrices.flatten()))


def get_compression_complexity(normalized_data, thresholds: List[float]):
    """
    At the moment, this expects a freq based data, i.e. a num_samples X freqs arary
    """

    data_matrices = bin_sample(normalized_data, thresholds)
    raw_data = data_matrices_to_string(data_matrices)
    return lempel_ziv_complexity(raw_data)


def get_subject_complexity(samples, samples_per_bin:int, thresholds: List[float]):
    # for now, just sample over samples
    normalized_data = normalize_data(samples)
    return [get_compression_complexity(normalized_data[i], thresholds) for i in range(samples.shape[0])]


if __name__ == '__main__':
    samples_per_bin = 100
    thresholds = [4, 2.5]
    data = datasets.DATASETS['gabaergic_anesthetic']
    all_results = []
    max_length = 0
    for case in tqdm(data.get_cases()):
        if 'volunteer' in case.lower():
            result = get_subject_complexity(data.get_data(case)['freq'], samples_per_bin, thresholds)
            if len(result) > max_length:
                max_length = len(result)
            all_results.append(result)


    # average out all samples
    average_results = []
    for i in range(max_length):
        complexities = np.array([single_case_result[i] for single_case_result in all_results if len(single_case_result) > i])
        average_results.append(complexities.mean())


    # create data
    x = range(max_length)
    plt.plot(average_results)
    plt.xlabel('time')
    plt.ylabel('average lempel ziv complexity')
    plt.legend()
    plt.show()





