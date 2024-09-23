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


class DFDataset(Dataset):
    """
    Turn a pandas data frame into a PyTorch Data Set.

    """

    def __init__(self,
                 df: pd.DataFrame,
                 column_groups: Sequence[Sequence[str]],
                 sample_fraction: float = 1.0,
                 sample_seed: int = None,
                 shuffle: bool = False):

        self.data_length = int(sample_fraction * len(df))
        df = sample_df(df, n_samples=self.data_length, sample_seed=sample_seed, shuffle=shuffle)

        self.column_groups = column_groups
        self.groups = [df[cg].to_numpy() for cg in self.column_groups]
        self.index = df.index

    def __len__(self) -> int:
        return self.data_length

    def __getitem__(self, ix: int):
        return tuple([g[ix] for g in self.groups])