import numpy as np

from typing import Any, Hashable, Optional, Sequence, Tuple, Union

from easyloader.dataset.base import EasyDataset
from easyloader.utils.random import Seedable


class ArrayDataset(EasyDataset):
    """
    Turn a list of numpy arrays into a PyTorch Data Set.
    """

    def __init__(self,
                 arrays: Union[np.ndarray, Sequence[np.ndarray]],
                 ids: Optional[Sequence[Any]] = None,
                 sample_fraction: float = 1.0,
                 sample_seed: Seedable = None,
                 shuffle_seed: Seedable = None):
        """
        Constructor for the ArrayDataset class.

        :param arrays: The arrays.
        :param sample_fraction: Fraction of the dataset to sample.
        :param sample_seed: Seed for random sampling.
        """

        # Initialize the parent class
        super().__init__(sample_fraction=sample_fraction,
                         sample_seed=sample_seed,
                         shuffle_seed=shuffle_seed)

        arrays, self._single = self._process_arrays(arrays)
        array_length = self._get_input_data_length(arrays)
        self._ids = self._process_ids(ids, array_length)

        # Perform sampling
        index = [*range(array_length)]
        if sample_fraction is not None:
            index = self.sample_random_state.sample(index, int(sample_fraction * array_length))
            index = sorted(index)
            self._arrays = [arr[index] for arr in arrays]
        else:
            self._arrays = arrays
        self._index = index

    @staticmethod
    def _process_arrays(arrays: Sequence[np.ndarray]) -> Tuple[bool, Sequence[np.ndarray]]:
        """
        Check the consistency of the arrays, and decide whether to treat the output as single or multi.

        :param arrays: The inputted arrays.
        :return:
        """
        if isinstance(arrays, np.ndarray):
            arrays = [arrays]
            single = True
        else:
            if not isinstance(arrays, Sequence) or not all(isinstance(arr, np.ndarray) for arr in arrays):
                raise TypeError('Data must be inputted as either a single array or a sequence of arrays.')
            single = False
        
        return arrays, single

    @staticmethod
    def _process_ids(ids: Optional[Sequence[Any]], array_length: int) -> Sequence[Hashable]:
        """
        Process the IDs, given as either None, or a list.

        :param ids: The Ids.
        :param array_length: The length of the array used for this Dataset.
        :return: A list of IDs.
        """
        # Organise the IDs
        if ids is not None:
            if len(ids) != array_length:
                raise ValueError('ID list must be the same length as the arrays.')
            return ids
        else:
            return [*range(array_length)]

    @staticmethod
    def _get_input_data_length(arrays) -> int:
        """
        Get the length of the inputted data.

        :return: The length.
        """
        # Check lengths
        array_lengths = [len(arr) for arr in arrays]
        if len(set(array_lengths)) != 1:
            raise ValueError('Arrays must all have the same length')
        return array_lengths[0]

    def shuffle(self):
        """
        Shuffle the underlying arrays.

        :return: None.
        """
        ixs = [*range(len(self.index))]
        self.shuffle_random_state.shuffle(ixs)
        self._arrays = [arr[ixs] for arr in self._arrays]
        self._index = list(np.array(self._index)[ixs])

    def __getitem__(self, ix: Union[int, slice]):
        """
        Get items, either by a single index or by a slice.

        :return: A subset of items.
        """
        if self._single:
            return self._arrays[0][ix]
        else:
            return tuple([arr[ix] for arr in self._arrays])
