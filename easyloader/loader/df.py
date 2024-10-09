import pandas as pd
import torch

from typing import Sequence

from easyloader.loader.base import EasyDataLoader
from easyloader.dataset.df import DFDataset
from easyloader.utils.random import Seedable


class DFDataLoader(EasyDataLoader):
    """
    Turn a Pandas data frame into a PyTorch Data Loader.

    """

    def __init__(self,
                 df: pd.DataFrame,
                 column_groups: Sequence[Sequence[str]],
                 id_column: str = None,
                 batch_size: int = 1,
                 sample_fraction: float = None,
                 shuffle: bool = False,
                 sample_seed: Seedable = None,
                 shuffle_seed: Seedable = None):
        """
        Constructor for the DFDataLoader class.

        :param df: The DF to use for the data loader.
        :param column_groups: The column groups to use.
        :param id_column: The column to use as IDs. If not set, use the DF index.
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

        self.dataset = DFDataset(df, id_column=id_column, column_groups=column_groups, sample_seed=sample_seed,
                                 sample_fraction=sample_fraction, shuffle_seed=shuffle_seed)

    def __next__(self):
        """
        Get the next batch.

        :return: The next batch.
        """
        if self.i >= len(self):
            raise StopIteration

        batch = tuple(
            torch.Tensor(self.dataset.df[g].iloc[self.i * self.batch_size: (self.i + 1) * self.batch_size].to_numpy())
            for g in self.dataset.column_groups)

        self.i += 1
        return batch
