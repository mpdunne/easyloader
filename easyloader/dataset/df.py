import pandas as pd
import numpy as np

from typing import Hashable, Iterable, Optional, Sequence, Union

from easyloader.dataset.base import EasyDataset
from easyloader.utils.random import Seedable


class DFDataset(EasyDataset):
    """
    Turn a Pandas data frame into a PyTorch Data Set.

    """

    def __init__(self,
                 df: pd.DataFrame,
                 columns: Optional[Union[Sequence[str], Sequence[Sequence[str]]]] = None,
                 ids: Union[str, Sequence[Hashable]] = None,
                 sample_fraction: float = 1.0,
                 sample_seed: Seedable = None,
                 shuffle_seed: Seedable = None):

        """
        Constructor for the DFDataset class.

        :param df: The DF to use for the data set.
        :param columns: The column groups to use.
        :param ids: The column to use as IDs. If not set, use the DF index.
        :param sample_fraction: Fraction of the dataset to sample.
        :param sample_seed: Seed for random sampling.
        :param shuffle_seed: Seed for shuffling.
        """

        # Initialize the parent class
        super().__init__(sample_fraction=sample_fraction,
                         sample_seed=sample_seed,
                         shuffle_seed=shuffle_seed)

        # Organise the IDs
        if ids is not None:
            if isinstance(ids, str):
                if ids not in df.columns:
                    raise ValueError('ID column must be a column in the DF.')
                else:
                    self._ids = df[ids]
            elif isinstance(ids, Sequence):
                if len(ids) != len(df):
                    raise ValueError('If specified as a sequence, IDs must have the same length as the DF.')
                self._ids = ids
            else:
                raise TypeError('IDs must either be specified as a list or a column name, or omitted.')
        else:
            self._ids = df.index

        # Perform sampling
        self._index = [*range(len(df))]
        if sample_fraction is not None:
            index = self.sample_random_state.sample(self._index, int(sample_fraction * len(df)))
            index = sorted(index)
            self._index = index
            self.df = df.iloc[self._index]
        else:
            self.df = df

        if columns is None:
            # TODO: Don't interpret "no columns" as a group.
            columns = [df.columns]

        self.column_groups = columns

    def shuffle(self):
        """
        Shuffle the underlying DF.

        :return: None.
        """
        ixs = [*range(len(self.df))]
        self.shuffle_random_state.shuffle(ixs)
        self._index = list(np.array(self._index)[ixs])
        self.df = self.df.iloc[ixs]

    def __getitem__(self, ix: Union[int, slice]):
        """
        Get items, either by a single index or by a slice.

        :return: A subset of items.
        """
        return tuple([self.df[g].iloc[ix].to_numpy() for g in self.column_groups])
