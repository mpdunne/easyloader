import pandas as pd
import random
import torch

from typing import Sequence
from torch.utils.data import Dataset, DataLoader

from uloops.loaders.common import get_n_batches

from easyloader.loaders.base import EasyDataLoader
from easyloader.common.df import sample_df


class DFDataLoader(EasyDataLoader):
    """
    Turn a pandas data frame into a PyTorch Data Loader.

    """

    def __init__(self,
                 df: pd.DataFrame,
                 column_groups: Sequence[Sequence[str]],
                 batch_size: int = 1,
                 sample_fraction: float = None,
                 sample_seed: int = None,
                 shuffle: bool = False,
                 shuffle_seed: bool = None):
        """

        :param df:
        :param column_groups:
        :param batch_size: The batch size.
        :param sample_fraction: Fraction of the dataset to sample.
        :param sample_seed: Seed for random sampling.
        :param shuffle: Whether to shuffle the data.
        :param shuffle_seed: The seed to be used for shuffling.
        """

        # Initialize the parent class
        super().__init__(batch_size=batch_size,
                         sample_fraction=sample_fraction,
                         sample_seed=sample_seed,
                         shuffle=shuffle,
                         shuffle_seed=shuffle_seed)

        # Store the DF. Do the initial sampling now.
        self.df = df
        self.data_length = int(sample_fraction * len(self.df))
        self.df = sample_df(df, n_samples=self.data_length, sample_seed=sample_seed, shuffle=False)
        self.column_groups = column_groups

        # These will be set when __iter__ is called
        self.groups = None
        self.index = None

    def __iter__(self):
        df = sample_df(self.df, shuffle=self.shuffle)
        self.index = df.index
        self.groups = [df[cg].to_numpy() for cg in self.column_groups]
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.data_length:
            raise StopIteration
        batch = tuple(torch.Tensor(g[self.i: self.i + self.batch_size]) for g in self.groups)
        self.i += self.batch_size
        return batch

    def __len__(self) -> int:
        return self.n_batches
