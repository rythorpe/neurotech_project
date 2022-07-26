

class EEGDataset:
    number_of_channels: int

    def get_cases(self):
        """
        Returns a list that identifies all the recordings, where later an element from that list
        can be queried
        """
        raise NotImplemented()

    def get_data(self, case) -> dict:
        """
        Returns a dict that contains all the possible data that could be extracted (in numpy?)
        field name convention can be:
            'raw' : just the pure raw samples, i.e. a num_channels X num_samples array
            'time': the time of the samples
            'freq': the samples in the frequency domain
            'labels': the general label of the subject / the label for each sample
        """
        raise NotImplemented()



