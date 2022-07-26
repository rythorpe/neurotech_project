from data.data_api import EEGDataset
from typing import Dict
from data.datasets import gabaergic_anesthetic

DATASETS: Dict[str, EEGDataset] = {
    'gabaergic_anesthetic': gabaergic_anesthetic.GabaergicDataset()
}