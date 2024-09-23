import pandas as pd

from typing import Sequence

from easyloader.datasets.base import EasyDataset
from easyloader.common.df import sample_df


class DFDataset(EasyDataset):
    """
    Turn a Pandas data frame into a PyTorch Data Set.

    """

    def __init__(self,
                 df: pd.DataFrame,
                 column_groups: Sequence[Sequence[str]],
                 sample_fraction: float = 1.0,
                 sample_seed: int = None,
                 shuffle: bool = False,
                 shuffle_seed: int = None):

        """
        Constructor for the DFDataset class.

        :param df: The DF to use for the data set.
        :param column_groups: The column groups to use.
        :param sample_fraction: Fraction of the dataset to sample.
        :param sample_seed: Seed for random sampling.
        :param shuffle: Whether to shuffle the data.
        :param shuffle_seed: The seed to be used for shuffling.
        """

        # Initialize the parent class
        super().__init__(sample_fraction=sample_fraction,
                         sample_seed=sample_seed,
                         shuffle=shuffle,
                         shuffle_seed=shuffle_seed)

        self.data_length = int(sample_fraction * len(df))
        df = sample_df(df, n_samples=self.data_length, sample_seed=sample_seed, shuffle=shuffle)

        self.column_groups = column_groups
        self.groups = [df[cg].to_numpy() for cg in self.column_groups]
        self.index = df.index

    def __len__(self) -> int:
        return self.data_length

    def __getitem__(self, ix: int):
        return tuple([g[ix] for g in self.groups])