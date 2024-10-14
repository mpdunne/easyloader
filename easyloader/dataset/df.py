import pandas as pd
import numpy as np
import inspect

from typing import Hashable, Optional, Sequence, Union

from easyloader.dataset.base import EasyDataset
from easyloader.utils.random import Seedable
from easyloader.utils.typing import is_hashable


class DFDataset(EasyDataset):
    """
    Turn a Pandas data frame into a PyTorch Data Set.

    """

    def __init__(self,
                 df: pd.DataFrame,
                 columns: Optional[Union[Sequence[str], Sequence[Sequence[str]]]] = None,
                 ids: Union[str, Sequence[Hashable]] = None,
                 grain_size: int = 1,
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

        self._column_groups, self._single = self._process_columns(columns, df)
        self._ids = self._process_ids(ids, df)

        self._df = self._df_orig = df

        # Initialize the parent class
        super().__init__(data_length=len(df),
                         grain_size=grain_size,
                         sample_fraction=sample_fraction,
                         sample_seed=sample_seed,
                         shuffle_seed=shuffle_seed)

    @staticmethod
    def _process_columns(columns: Optional[Union[Sequence[str], Sequence[Sequence[str]]]],
                         df: pd.DataFrame):
        """
        Check the inputted columns, ensure they are well-defined and part of the DF,
        and decide whether the dataset should yield single or multi outputs.

        :param columns: A sequence of column names, or a sequence of sequences of column names.
        :param df: The DF against which to check the columns.
        :return: The column groups, and whether the output should be treated as single.
        """
        # If not specified, use all columns.
        if columns is None:
            columns = [*df.columns]

        if is_hashable(columns):
            # Special case where the "columns" is a tuple and that tuple is also a DF column.
            if isinstance(columns, tuple):
                if columns in df.columns:
                    return [columns], True
                else:
                    columns = list(columns)
            else:
                if columns not in df.columns:
                    raise ValueError('Single column specified but column is not present in the DF.')
                else:
                    return [columns], True

        if not isinstance(columns, Sequence):
            raise TypeError('Value for "columns" must be a sequence of column names, or of sequences of column names.')

        # Convert any tuples to lists, unless they are column names.
        columns_detupled = []
        for c in columns:
            if isinstance(c, tuple) and is_hashable(c) and c not in df.columns:
                c = list(c)
            columns_detupled.append(c)
        columns = columns_detupled

        # Next, check everything specified is in the DF columns.
        missing_columns = []
        for c in columns:
            if isinstance(c, list):
                for cc in c:
                    if not is_hashable(cc) or cc not in df.columns:
                        missing_columns.append(str(cc))
            else:
                if not is_hashable(c) or c not in df.columns:
                    missing_columns.append(str(c))

        if missing_columns:
            raise ValueError('Some specified columns are not in the DF: ' + ', '.join(missing_columns))

        if all(is_hashable(c) for c in columns):
            single = True
            column_groups = [columns]
        else:
            single = False
            column_groups = columns

        return column_groups, single

    @staticmethod
    def _process_ids(ids: Union[str, Sequence[Hashable]], df: pd.DataFrame) -> Sequence[Hashable]:
        """
        Process the IDs, given as either None, columns, or a list.

        :param ids: The provided IDs.
        :param df: The DataFrame
        :return: A list of IDs.
        """
        # Organise the IDs
        if ids is not None:
            if isinstance(ids, str):
                if ids not in df.columns:
                    raise ValueError('ID column must be a column in the DF.')
                else:
                    return df[ids]
            elif isinstance(ids, Sequence):
                if len(ids) != len(df):
                    raise ValueError('If specified as a sequence, IDs must have the same length as the DF.')
                return ids
            else:
                raise TypeError('IDs must either be specified as a list or a column name, or omitted.')
        else:
            return df.index

    @property
    def column_groups(self):
        """
        Return the list of column groups (or single column, or column group) being used.

        :return: The list of column groups (or single column, or column group)
        """
        return self._column_groups

    @staticmethod
    def _item_to_numpy(x):
        """
        df.to_numpy() is slightly faster than np.numpy(df). But single items are not
        always dfs, so be safe here. Type checking is very little overhead.

        :param x: The item to convert.
        :return: A numpy version.
        """
        if isinstance(x, pd.DataFrame):
            return x.to_numpy()
        else:
            return np.array(x)

    def _update_data(self):
        """
        This method is called after shuffling and sampling. It is used to make
        any necessary updates to the underlying data.

        :return:
        """
        self._groups = [self._df_orig[g].iloc[self._index] for g in self._column_groups]

    def __getitem__(self, ix: Union[int, slice]):
        """
        Get items, either by a single index or by a slice.

        :return: A subset of items.
        """
        if self._single:
            return self._item_to_numpy(self._groups[0].iloc[ix])
        else:
            return tuple([self._item_to_numpy(g.iloc[ix]) for g in self._groups])


DFDataset.__signature__ = inspect.signature(DFDataset.__init__)
