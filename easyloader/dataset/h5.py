import h5py
import math
import numpy as np

from pathlib import Path
from typing import Hashable, Optional, Sequence, Union

from easyloader.dataset.base import EasyDataset
from easyloader.utils.grains import grab_slices_from_grains
from easyloader.utils.random import Seedable
from easyloader.utils.typing import is_hashable


class H5Dataset(EasyDataset):
    """
    Turn a H5 file into a PyTorch Data Set.

    """

    def __init__(self,
                 data_path: Union[str, Path],
                 keys: Optional[Union[Sequence[str], str]],
                 ids: Union[str, Sequence[Hashable]] = None,
                 grain_size: int = 1,
                 sample_fraction: float = 1.0,
                 sample_seed: int = None,
                 shuffle_seed: Seedable = None):

        """
        Constructor for the H5Dataset class.

        :param data_path: The path to the H5 file that you want to load.
        :param keys: The keys that you want to grab.
        :param sample_fraction: Fraction of the dataset to sample.
        :param sample_seed: Seed for random sampling.
        :param shuffle_seed: Seed for shuffling.
        """

        # Initialize the parent class
        super().__init__(sample_fraction=sample_fraction,
                         sample_seed=sample_seed,
                         shuffle_seed=shuffle_seed)

        h5 = h5py.File(data_path, "r")
        self.h5 = h5
        self._single, keys = self._process_keys(keys, h5.keys())
        self._keys = keys

        # Check lengths
        data_lengths = [len(h5[key]) for key in keys]
        if len(set(data_lengths)) != 1:
            raise ValueError('All data must be the same length.')
        data_length = data_lengths[0]

        # Organise the IDs
        if ids is not None:
            if isinstance(ids, str):
                if ids not in h5.keys():
                    raise ValueError(f'Specified id key {ids} not present in H5 file.')
                if len(h5[ids]) != data_length:
                    raise ValueError(f'Length of data for ID key {ids} does not match that of other data.')
                self._ids = h5[ids][:]
            elif isinstance(ids, Sequence):
                if len(ids) != data_length:
                    raise ValueError('If specified as a sequence, IDs must have the same length as the H5 data.')
                self._ids = ids
            else:
                raise TypeError('If set, IDs must either be a sequence or a key contained in the H5 file.')
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

    @staticmethod
    def _process_keys(keys, available_keys):
        """
        Check and reorganise the keys, 
        
        :param keys: 
        :return: 
        """
        # Process keys
        if isinstance(keys, (str, bytes)):
            if keys in available_keys:
                keys = [keys]
                single = True
            else:
                ValueError('Single key specified but is not present in the H5.')
        else:
            if not isinstance(keys, Sequence) and all(is_hashable(item) for item in keys):
                raise TypeError('Keys must be specified as a sequence of hashable H5 keys.')
            single = False

        # If any keys are specified but not in the H5, complain.
        missing_keys = [key for key in keys if (not isinstance(key, Hashable)) or (key not in available_keys)]
        if len(missing_keys) != 0:
            raise ValueError('Missing or invalid keys: ' + ', '.join(missing_keys))
        
        return single, keys

    def shuffle(self):
        """
        Shuffle the underlying data

        :return: None.
        """
        self.shuffle_random_state.shuffle(self.grain_index)

    @property
    def index(self):
        """
        The index, relative to the original data.

        :return: The index.
        """
        return [ix for gix in self.grain_index for ix in range(gix * self.grain_size, (gix + 1) * self.grain_size)]

    @property
    def grain_index(self):
        """
        The grain index.

        :return: The grain index.
        """
        return self._grain_index

    @property
    def keys(self) -> Sequence[str]:
        """
        The specified keys that we want to get out of the H5.

        :return: The keys.
        """
        return self._keys

    def __getitem__(self, ix: Union[int, slice]):
        """
        Get items, either by a single index or by a slice.

        :return: A subset of items.
        """
        values = []

        if isinstance(ix, int):
            for key in self.keys:
                values.append(self.h5[key][self.index[ix]])

        elif isinstance(ix, slice):
            ix_slices = grab_slices_from_grains(self.grain_index, self.grain_size, ix.start, ix.stop)
            for key in self.keys:
                values.append(np.concatenate([self.h5[key][ix_slice] for ix_slice in ix_slices]))

        else:
            raise ValueError('Index ix must either be an int or a slice.')

        if self._single:
            return values[0]
        else:
            return tuple(values)
