import math

from abc import ABC, abstractmethod
from torch.utils.data import Dataset
from typing import Hashable, Sequence, Iterable, Union

from easyloader.utils.random import get_random_state, Seedable


class EasyDataset(Dataset, ABC):
    """
    Interface class for EasyLoader datasets with common functionality for sampling and indexing.
    """

    _ids: Sequence[Hashable]
    _index: Sequence[int]

    def __init__(self, data_length: int,
                 grain_size: int = 1,
                 sample_fraction: float = 1.0,
                 sample_seed: Seedable = None,
                 shuffle_seed: Seedable = None):
        """
        Constructor for the EasyDataset class Interface.

        :param data_length: The length of the data being considered.
        :param grain_size: The grain size. Defaults to 1, to not use graining.
        :param sample_fraction: Fraction of the dataset to sample.
        :param sample_seed: Seed for random sampling.
        :param shuffle_seed: Seed for shuffling.
        """

        super().__init__()

        self.sample_random_state = get_random_state(sample_seed)
        self.shuffle_random_state = get_random_state(shuffle_seed)

        self._data_length = data_length
        self._grain_size = grain_size

        n_grains = int(math.ceil(data_length / grain_size))
        self.grain_index = [*range(n_grains)]

        self.sample(sample_fraction)
        self._shuffled = False

    @abstractmethod
    def _update_data(self):
        """
        This method is called after shuffling and sampling, or any time the grain index (index) is
        changed. It is used to make any necessary updates to the underlying data.

        :return:
        """
        pass

    @abstractmethod
    def __getitem__(self, ix: Union[int, slice]):
        """
        Get items, either by a single index or by a slice.

        :return: A subset of items.
        """
        pass

    def __len__(self) -> int:
        """
        The length of the sampled data set.

        :return: The length of the sampled data set.
        """
        return len(self.index)

    @property
    def ids(self) -> Iterable:
        """
        The IDs, according to the id_column attribute.

        :return: The IDs
        """
        if self._ids is None:
            return self.index
        else:
            return [self._ids[i] for i in self.index]

    @property
    def grain_index(self):
        """
        The grain index. If grain_size=1 then this is the same as the index.
        :return:
        """
        return self._grain_index

    @grain_index.setter
    def grain_index(self, grain_index):
        """
        The setter for the grain index. If grain_size=1 then this is the same as the index.
        :return:
        """
        self._grain_index = grain_index

        if self._grain_size == 1:
            self._index = self._grain_index
        else:
            self._index = [ix for gix in self._grain_index for
                           ix in range(gix * self._grain_size, (gix + 1) * self._grain_size)
                           if ix < self._data_length]

        self._update_data()

    @property
    def index(self):
        """
        The index of the underlying data, relative to the original.

        :return: The length
        """
        return self._index

    def sample(self, sample_fraction: float, subsample=False):
        """
        Shuffle the underlying data.

        :param sample_fraction: The fraction at which to sample.
        :param subsample: Whether to sample the original data or the already sampled data.
        """
        if sample_fraction is not None:
            if subsample:
                grains = self.grain_index
            else:
                n_grains = int(math.ceil(self._data_length / self._grain_size))
                grains = [*range(n_grains)]
            grains = self.sample_random_state.sample(grains, int(sample_fraction * len(grains)))
            self.grain_index = sorted(grains)

    def shuffle(self):
        """
        Shuffle the underlying data.

        """
        self.grain_index = self.shuffle_random_state.sample(self.grain_index, len(self.grain_index))
        self._shuffled = True
