import pandas as pd

from typing import Sequence, Union

from easyloader.dataset.base import EasyDataset
from easyloader.data.df import DFData
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
                 sample_seed: Seedable = None):

        """
        Constructor for the DFDataset class.

        :param df: The DF to use for the data set.
        :param column_groups: The column groups to use.
        :param id_column: The column to use as IDs. If not set, use the DF index.
        :param sample_fraction: Fraction of the dataset to sample.
        :param sample_seed: Seed for random sampling.
        """

        # Initialize the parent class
        super().__init__(sample_fraction=sample_fraction,
                         sample_seed=sample_seed)

        self.data = DFData(df, id_column=id_column, sample_fraction=sample_fraction, sample_seed=sample_seed)

        if column_groups is None:
            raise NotImplemented('Currently need to specify column groups')
            # TODO: Allow specify no columns, or just "columns".

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

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, ix: Union[int, slice]):
        return tuple([self.data.df[g].iloc[ix].to_numpy() for g in self.column_groups])
