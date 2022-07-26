import unittest
from data import datasets


class DataTest(unittest.TestCase):

    def test_sanity(self):
        dataset = datasets.DATASETS['gabaergic_anesthetic']

        # get the cases, and load the first one
        cases = dataset.get_cases()
        data = dataset.get_data(cases[0])

        # check that the expected data was extracted
        assert 'time' in data.keys()
        assert 'freq' in data.keys()
        assert 'labels' in data.keys()


if __name__ == '__main__':
    unittest.main()
