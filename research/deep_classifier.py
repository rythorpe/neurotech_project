import torch
import torch.nn as nn
import torch.nn.functional as F
from data import datasets
import numpy as np
from scipy import stats


class MLP(nn.Module):
    def __init__(self, depth=5, width=128, input_channels=100, output_ch=1):
        super(MLP, self).__init__()

        self.linear_layers = [nn.Linear(input_channels, width)] + [nn.Linear(width, width) for i in range(depth - 1)]
        self.output_layer = nn.Linear(width, output_ch)

    def forward(self, x):
        for layer in self.linear_layers:
            x = F.relu(layer(x))

        return self.output_layer(x)


def normalize_data(case_samples):
    """
    For each frequency band, compute its own z-score
    """

    normalized_output = np.zeros_like(case_samples)
    for freq in range(case_samples.shape[1]):
        single_freq_data = case_samples[:, freq]
        normalized_output[:, freq] = stats.zscore(single_freq_data)

    return normalized_output

def load_data_to_pytorch(use_only_volunteer_data=False):
    data = datasets.DATASETS['gabaergic_anesthetic']
    for case in (data.get_cases()):
        if use_only_volunteer_data:
            if 'volunteer' in case.lower():

        else:







