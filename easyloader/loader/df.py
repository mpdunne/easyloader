import pandas as pd
import torch

from typing import Sequence

from easyloader.loader.base import EasyDataLoader
from easyloader.data.df import DFData
from easyloader.utils.batch import get_n_batches
from easyloader.utils.random import Seedable


class DFDataLoader(EasyDataLoader):
    """
    Turn a Pandas data frame into a PyTorch Data Loader.

    """

    def __init__(self,
                 df: pd.DataFrame,
                 column_groups: Sequence[Sequence[str]],
                 id_column: str = None,
                 batch_size: int = 1,
                 sample_fraction: float = None,
                 shuffle: bool = False,
                 sample_seed: Seedable = None,
                 shuffle_seed: Seedable = None):
        """
        Constructor for the DFDataLoader class.

        :param df: The DF to use for the data loader.
        :param column_groups: The column groups to use.
        :param id_column: The column to use as IDs. If not set, use the DF index.
        :param batch_size: The batch size.
        :param sample_fraction: Fraction of the dataset to sample.
        :param shuffle: Whether to shuffle the data.
        :param sample_seed: Seed for random sampling.
        :param shuffle_seed: The seed to be used for shuffling.
        """

        # Initialize the parent class
        super().__init__(batch_size=batch_size,
                         sample_fraction=sample_fraction,
                         sample_seed=sample_seed,
                         shuffle=shuffle,
                         shuffle_seed=shuffle_seed)

        self.data = DFData(df, id_column=id_column, sample_seed=sample_seed,
                           sample_fraction=sample_fraction, shuffle_seed=shuffle_seed)
        self.column_groups = column_groups

    @property
    def index(self):
        """
        The numeric indices of the underlying DF, relative to the inputted one.

        :return: The indices.
        """
        return self.data.index

    @property
    def ids(self):
        """
        The IDs, according to the id_column attribute.

        :return: The IDs
        """
        return self.data.ids

    def __iter__(self):
        if self.shuffle:
            self.data.shuffle()

        self.i = 0
        return self

    def __next__(self):
        if self.i >= len(self):
            raise StopIteration

        batch = tuple(
            torch.Tensor(self.data.df[g].iloc[self.i * self.batch_size: (self.i + 1) * self.batch_size].to_numpy())
            for g in self.column_groups)

        self.i += 1
        return batch

    def __len__(self) -> int:
        return get_n_batches(len(self.data), self.batch_size)
