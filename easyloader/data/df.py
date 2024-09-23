import random

from typing import Iterable

from easyloader.utils.random import get_random_state


class DFData:
    """
    Data class for DF data.
    """

    def __init__(self, df,
                 id_column: str = None,
                 sample_fraction: float = None,
                 sample_seed=None,
                 shuffle_seed=None):
        """
        Constructor for the DFData class

        :param df: The DF to use for the data loader.
        :param id_column: The column to use as IDs. If not set, use the DF index.
        :param sample_fraction: Fraction of the dataset to sample.
        :param sample_seed: Seed for random sampling.
        :param shuffle_seed: The seed to be used for shuffling.
        """

        self.sample_random_state = get_random_state(sample_seed)
        self.shuffle_random_state = get_random_state(shuffle_seed)

        if id_column is not None and not isinstance(id_column, str) and id_column not in df.columns:
            raise ValueError('ID column must be a column in the DF.')

        self.id_column = id_column

        self._index = [*range(len(df))]
        if sample_fraction is not None:
            index = random.sample(self._index, int(sample_fraction * len(df)))
            index = sorted(index)
            self._index = index
            self.df = df.iloc[self._index]
        else:
            self.df = df

    def shuffle(self):
        """
        Shuffle the underlying DF.

        :return: None.
        """
        self.shuffle_random_state.shuffle(self.index)
        self.df = self.df.iloc[self.index]

    def ids(self) -> Iterable:
        """
        The IDs, according to the id_column attribute.

        :return: The IDs
        """
        if self.id_column is not None:
            return self.df[self.id_column]
        else:
            return self.df.index

    @property
    def index(self):
        return self._index

    def __len__(self):
        return len(self.df)
