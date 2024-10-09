import pandas as pd
import numpy as np

from typing import Iterable, Sequence, Union

from easyloader.dataset.base import EasyDataset
from easyloader.utils.random import Seedable


class DFDataset(EasyDataset):
    """
    Turn a Pandas data frame into a PyTorch Data Set.

    """

    def __init__(self,
                 df: pd.DataFrame,
                 column_groups: Sequence[Sequence[str]] = None,
                 id_column: str = None,
                 sample_fraction: float = 1.0,
                 sample_seed: Seedable = None,
                 shuffle_seed: Seedable = None):

        """
        Constructor for the DFDataset class.

        :param df: The DF to use for the data set.
        :param column_groups: The column groups to use.
        :param id_column: The column to use as IDs. If not set, use the DF index.
        :param sample_fraction: Fraction of the dataset to sample.
        :param sample_seed: Seed for random sampling.
        :param shuffle_seed: Seed for shuffling.
        """

        # Initialize the parent class
        super().__init__(sample_fraction=sample_fraction,
                         sample_seed=sample_seed,
                         shuffle_seed=shuffle_seed)

        # Organise the IDs
        if id_column is not None and not isinstance(id_column, str) and id_column not in df.columns:
            raise ValueError('ID column must be a column in the DF.')
        self.id_column = id_column

        # Perform sampling
        self._index = [*range(len(df))]
        if sample_fraction is not None:
            index = self.sample_random_state.sample(self._index, int(sample_fraction * len(df)))
            index = sorted(index)
            self._index = index
            self.df = df.iloc[self._index]
        else:
            self.df = df

        if column_groups is None:
            column_groups = [df.columns]

        self.column_groups = column_groups

    def shuffle(self):
        """
        Shuffle the underlying DF.

        :return: None.
        """
        ixs = [*range(len(self.df))]
        self.shuffle_random_state.shuffle(ixs)
        self._index = list(np.array(self._index)[ixs])
        self.df = self.df.iloc[ixs]

    @property
    def ids(self) -> Iterable:
        """
        The IDs, according to the id_column attribute.

        :return: The IDs
        """
        if self.id_column is not None:
            return self.df[self.id_column]
        else:
            return self.df.index

    def __getitem__(self, ix: Union[int, slice]):
        return tuple([self.df[g].iloc[ix].to_numpy() for g in self.column_groups])
