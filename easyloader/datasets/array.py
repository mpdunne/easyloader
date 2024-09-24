import numpy as np

from typing import Any, Sequence

from easyloader.datasets.base import EasyDataset
from easyloader.data.array import ArrayData


class ArrayDataset(EasyDataset):
    """
    Turn a list of numpy arrays into a PyTorch Data Set.
    """

    def __init__(self,
                 arrays: Sequence[np.ndarray],
                 ids: Sequence[Any] = None,
                 sample_fraction: float = 1.0,
                 sample_seed: int = None):
        """
        Constructor for the ArrayDataset class.

        :param arrays: The arrays.
        :param sample_fraction: Fraction of the dataset to sample.
        :param sample_seed: Seed for random sampling.
        """
        # Initialize the parent class
        super().__init__(sample_fraction=sample_fraction,
                         sample_seed=sample_seed)

        self.data = ArrayData(arrays, ids=ids, sample_fraction=sample_fraction, sample_seed=sample_seed)

    def __len__(self) -> int:
        return self.data_length

    def __getitem__(self, ix: int):
        return tuple([arr[ix] for arr in self.arrays])
