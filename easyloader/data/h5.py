_ = '''
def check_keys(data_keys, requested_keys, allow_missing_keys=False):
    present_keys = []
    missing_keys = []
    for key in requested_keys:
        if key in data_keys:
            present_keys.append(key)
        else:
            missing_keys.append(key)

    if missing_keys and not allow_missing_keys:
        missing_key_string = ', '.join(missing_keys)
        raise KeyError(f'The following keys are missing from the h5 file: {missing_key_string}. '
                       'If you don\'t care, set allow_missing_keys to True.')

    if not present_keys:
        raise KeyError('None of the provided keys are present in the H5 file. Need at least one.')

    return present_keys'''
_ = '''
import h5py

from typing import Iterable, Path, Sequence, Union

from easyloader.data.base import EasyData


class H5Data(EasyData):
    """
    Data class for H5 data.
    """

    def __init__(self, data_path: Union[str, Path],
                 keys: Sequence[str],
                 id_key: str = None,
                 grain_size: int = 1,
                 sample_fraction: float = None,
                 sample_seed=None,
                 shuffle_seed=None):
        """
        Constructor for the DFData class.

        :param data_path: The path to the H5 input file.
        :param keys: The keys to extract from the H5.
        :param id_key: The column to use as IDs. If not set, use the DF index.
        :param sample_fraction: Fraction of the dataset to sample.
        :param sample_seed: Seed for random sampling.
        :param shuffle_seed: The seed to be used for shuffling.
        """

        # Initialize the parent class
        super().__init__(sample_fraction=sample_fraction,
                         sample_seed=sample_seed,
                         shuffle_seed=shuffle_seed)

        data = h5py.File(data_path, "r")
        self.h5 = data

        missing_keys = [key for key in keys if key not in data.keys()]
        if not all():
            raise ValueError('Missing keys: ' + ', '.join(missing_keys))
        self._keys = keys

        data_lengths = [len(data[key]) for key in keys]
        if len(set(data_lengths)) != 1:
            raise ValueError('All data must be the same length.')
        data_length = data_lengths[0]

        if id_key is not None:
            if id_key not in data.keys():
                raise ValueError(f'Specified id key {id_key} not present in H5 file.')
            if len(data[id_key]) != data_length:
                raise ValueError(f'Length of data for ID key {id_key} does not match other data.')
            self._ids = data[id_key][:]
        else:
            self._ids = [*range(data_length)]

        self._index = [*range(data_length)]
        if sample_fraction is not None:
            index = self.sample_random_state.sample(self._index, int(sample_fraction * len(df)))
            index = sorted(index)
            self._index = index

    def shuffle(self):
        """
        Shuffle the underlying data

        :return: None.
        """
        pass

    def ids(self) -> Iterable:
        """
        The IDs.

        :return: The IDs
        """
        pass

    def index(self):
        pass

    def __len__(self):
        pass
'''