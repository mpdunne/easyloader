import h5py
import math

from pathlib import Path
from typing import Iterable, Sequence, Union

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

        # Process keys
        missing_keys = [key for key in keys if key not in data.keys()]
        if len(missing_keys) != 0:
            raise ValueError('Missing keys: ' + ', '.join(missing_keys))
        self._keys = keys

        # Check lengths
        data_lengths = [len(data[key]) for key in keys]
        if len(set(data_lengths)) != 1:
            raise ValueError('All data must be the same length.')
        data_length = data_lengths[0]

        # Organise the IDs
        if id_key is not None:
            if id_key not in data.keys():
                raise ValueError(f'Specified id key {id_key} not present in H5 file.')
            if len(data[id_key]) != data_length:
                raise ValueError(f'Length of data for ID key {id_key} does not match that of other data.')
            self._ids = data[id_key][:]
        else:
            self._ids = [*range(data_length)]

        # Organise grains & perform sampling
        n_grains = int(math.ceil(data_length / grain_size))
        self.grain_size = grain_size
        self.n_grains = n_grains
        grains = [*range(n_grains)]
        if sample_fraction is not None:
            grains = self.sample_random_state.sample(grains, int(sample_fraction * n_grains))
            grains = sorted(grains)
        self._grain_index = grains

    def shuffle(self):
        """
        Shuffle the underlying data

        :return: None.
        """
        self.shuffle_random_state.shuffle(self.grain_index)

    @property
    def ids(self) -> Iterable:
        """
        The IDs.

        :return: The IDs
        """
        return [self._ids[i] for i in self.index]

    @property
    def index(self):
        return [ix for gix in self.grain_index for ix in range(gix * self.grain_size, (gix + 1) * self.grain_size)]

    @property
    def grain_index(self):
        return self._grain_index

    @property
    def keys(self):
        return self._keys

    def __len__(self):
        return len(self.index)
