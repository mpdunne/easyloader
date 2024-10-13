import h5py
import numpy as np

from pathlib import Path
from typing import Hashable, Optional, Sequence, Union

from easyloader.dataset.base import EasyDataset
from easyloader.utils.grains import grab_slices_from_grains
from easyloader.utils.random import Seedable
from easyloader.utils.typing import is_hashable
from easyloader.utils.slices import merge_neighbouring_slices


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
                 sample_seed: Seedable = None,
                 shuffle_seed: Seedable = None):

        """
        Constructor for the H5Dataset class.

        :param data_path: The path to the H5 file that you want to load.
        :param keys: The keys that you want to grab.
        :param ids: A list of IDs.
        :param grain_size: The grain size. Defaults to 1, to not use graining.
        :param sample_fraction: Fraction of the dataset to sample.
        :param sample_seed: Seed for random sampling.
        :param shuffle_seed: Seed for shuffling.
        """

        h5 = h5py.File(data_path, "r")

        keys, self._single = self._process_keys(keys, h5.keys())
        data_length = self._get_input_data_length(h5, keys)
        self._ids = self._process_ids(ids, data_length, h5)

        self._h5 = h5
        self._keys = keys

        # Initialize the parent class
        super().__init__(data_length=data_length,
                         grain_size=grain_size,
                         sample_fraction=sample_fraction,
                         sample_seed=sample_seed,
                         shuffle_seed=shuffle_seed)

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
        
        return keys, single

    @staticmethod
    def _process_ids(ids: Union[str, Sequence[Hashable]], data_length: int, h5: h5py.File) -> Sequence[Hashable]:
        """
        Process the IDs, given as either None, columns, or a list.

        :param ids: The provided IDs.
        :param data_length: The length of the data that we're interested in.
        :param h5: The h5 File object.
        :return: A list of IDs.
        """
        if ids is not None:
            if isinstance(ids, str):
                if ids not in h5.keys():
                    raise ValueError(f'Specified id key {ids} not present in H5 file.')
                if len(h5[ids]) != data_length:
                    raise ValueError(f'Length of data for ID key {ids} does not match that of other data.')
                return h5[ids][:]
            elif isinstance(ids, Sequence):
                if len(ids) != data_length:
                    raise ValueError('If specified as a sequence, IDs must have the same length as the H5 data.')
                return ids
            else:
                raise TypeError('If set, IDs must either be a sequence or a key contained in the H5 file.')
        else:
            return [*range(data_length)]

    @staticmethod
    def _get_input_data_length(h5, keys) -> int:
        """
        Get the length of the inputted data.

        :return: The length
        """
        # Check lengths
        data_lengths = [len(h5[key]) for key in keys]
        if len(set(data_lengths)) != 1:
            raise ValueError('All data must be the same length.')
        return data_lengths[0]

    @property
    def index(self):
        """
        The index, relative to the original data.

        :return: The index.
        """
        return self._index

    @property
    def keys(self) -> Sequence[str]:
        """
        The specified keys that we want to get out of the H5.

        :return: The keys.
        """
        return self._keys

    def _update_data(self):
        """
        For H5 data set, this does nothing as the index is always used directly.

        :return:
        """
        pass

    def __getitem__(self, ix: Union[int, slice]):
        """
        Get items, either by a single index or by a slice.

        :return: A subset of items.
        """
        values = []

        if isinstance(ix, int):
            for key in self.keys:
                values.append(self._h5[key][self.index[ix]])

        elif isinstance(ix, slice):
            ix_slices = grab_slices_from_grains(self.grain_index, self._grain_size, ix.start, ix.stop)

            # The chances of having any neighbouring slices when shuffled is minimal, so avoid this slight overhead.
            # Otherwise, this should be a quick and easy way to access the H5 as efficiently as possible.
            if not self._shuffled:
                ix_slices = merge_neighbouring_slices(ix_slices)

            for key in self.keys:
                values.append(np.concatenate([self._h5[key][ix_slice] for ix_slice in ix_slices]))

        else:
            raise ValueError('Index ix must either be an int or a slice.')

        if self._single:
            return values[0]
        else:
            return tuple(values)
