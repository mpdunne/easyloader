import pandas as pd

from typing import Sequence

from easyloader.datasets.base import EasyDataset
from easyloader.data.df import DFData


class DFDataset(EasyDataset):
    """
    Turn a Pandas data frame into a PyTorch Data Set.

    """

    def __init__(self,
                 df: pd.DataFrame,
                 column_groups: Sequence[Sequence[str]],
                 id_column: str = None,
                 sample_fraction: float = 1.0,
                 sample_seed: int = None,
                 shuffle: bool = False,
                 shuffle_seed: int = None):

        """
        Constructor for the DFDataset class.

        :param df: The DF to use for the data set.
        :param column_groups: The column groups to use.
        :param id_column: The column to use as IDs. If not set, use the DF index.
        :param sample_fraction: Fraction of the dataset to sample.
        :param shuffle: Whether to shuffle the data.
        :param sample_seed: Seed for random sampling.
        :param shuffle_seed: The seed to be used for shuffling.
        """

        # Initialize the parent class
        super().__init__(sample_fraction=sample_fraction,
                         sample_seed=sample_seed,
                         shuffle=shuffle,
                         shuffle_seed=shuffle_seed)

        self.data = DFData(df, id_column=id_column, sample_seed=sample_seed, shuffle_seed=shuffle_seed)
        self.column_groups = column_groups

        if self.shuffle:
            self.data.shuffle()

        self.groups = [self.data[cg].to_numpy() for cg in self.column_groups]

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

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, ix: int):
        return tuple([g.iloc[ix] for g in self.groups])
