import pandas as pd
import random
import torch

from typing import Sequence
from torch.utils.data import Dataset, DataLoader

from uloops.loaders.common import get_n_batches


def sample_df(df: pd.DataFrame,
              n_samples: int = None,
              shuffle: bool = False,
              sample_seed: int = None,
              shuffle_seed: int = None) -> pd.DataFrame:

    if n_samples is not None and n_samples != len(df):
        random.seed(sample_seed)
        sample = random.sample([*range(len(df))], n_samples)
        sample = sorted(sample)
        df = df.iloc[sample]

    if shuffle:
        df = df.sample(frac=1, random_state=shuffle_seed)

    return df


class DFDataLoader(DataLoader):
    """
    Turn a pandas data frame into a PyTorch Data Loader.

    """

    def __init__(self,
                 df: pd.DataFrame,
                 column_groups: Sequence[Sequence[str]],
                 batch_size: int = 1,
                 sample_fraction: float = 1.0,
                 sample_seed: int = None,
                 shuffle: bool = False):

        # Store the DF. Do the initial sampling now.
        self.df = df
        self.data_length = int(sample_fraction * len(self.df))
        self.df = sample_df(df, n_samples=self.data_length, sample_seed=sample_seed, shuffle=False)
        self.column_groups = column_groups

        # Only do any shuffling at iter time.
        self.shuffle = shuffle

        # Calculate number of batches
        self.n_batches = get_n_batches(self.data_length, batch_size)
        self.batch_size = batch_size

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
