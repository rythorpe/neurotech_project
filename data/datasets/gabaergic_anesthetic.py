from pathlib import Path
from itertools import chain
from data.data_api import EEGDataset
import numpy as np


class GabaergicDataset(EEGDataset):

    def __init__(self):
        self.number_of_channels = 1

    def get_cases(self):
        cases = []
        data_path = Path(__file__).parent.joinpath(Path('gabaergic_anesthetic'))
        for case in chain(data_path.joinpath(Path("OR")).iterdir(),
                          data_path.joinpath(Path("Volunteer")).iterdir()):
            if 'Sdb' in str(case):
                case_name = str(case)
                cases.append(case_name[:case_name.rfind('_')])

        return cases

    def get_data(self, case):
        labels = np.genfromtxt(case + '_l.csv', delimiter=',')
        time = np.genfromtxt(case + '_t.csv', delimiter=',')
        freq = np.genfromtxt(case + '_Sdb.csv', delimiter=',').T

        return {
            'labels': labels,
            'time': time,
            'freq': freq
        }



