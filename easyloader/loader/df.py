import pandas as pd

from typing import Hashable, Optional, Sequence, Union

from easyloader.loader.base import EasyDataLoader
from easyloader.dataset.df import DFDataset
from easyloader.utils.random import Seedable


class DFDataLoader(EasyDataLoader):
    """
    Turn a Pandas data frame into a PyTorch Data Loader.

    """

    dataset: DFDataset

    def __init__(self,
                 df: pd.DataFrame,
                 columns: Optional[Union[Sequence[str], Sequence[Sequence[str]]]] = None,
                 ids: Union[str, Sequence[Hashable]] = None,
                 batch_size: int = 1,
                 sample_fraction: float = None,
                 shuffle: bool = False,
                 sample_seed: Seedable = None,
                 shuffle_seed: Seedable = None):
        """
        Constructor for the DFDataLoader class.

        :param df: The DF to use for the data loader.
        :param columns: The column groups to use.
        :param ids: The column to use as IDs. If not set, use the DF index.
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

        self.dataset = DFDataset(df, ids=ids, columns=columns, sample_seed=sample_seed,
                                 sample_fraction=sample_fraction, shuffle_seed=shuffle_seed)
