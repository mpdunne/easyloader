from abc import ABC, abstractmethod
from torch.utils.data import DataLoader


class EasyDataLoader(DataLoader, ABC):
    """
    Interface class for EasyLoader dataloaders with common functionality for sampling and indexing.
    """

    def __init__(self,
                 batch_size: int = 1,
                 sample_fraction: float = None,
                 sample_seed: int = None,
                 shuffle: bool = False,
                 shuffle_seed: bool = None):
        """
        Initializes the dataloader with batching and shuffling options.

        :param batch_size: The batch size.
        :param sample_fraction: Fraction of the dataset to sample.
        :param sample_seed: Seed for random sampling.
        :param shuffle: Whether to shuffle the data.
        :param shuffle_seed: The seed to be used for shuffling.
        """

        self.batch_size = batch_size
        self.sample_fraction = sample_fraction
        self.sample_seed = sample_seed
        self.shuffle = shuffle
        self.shuffle_seed = shuffle_seed
        self.ids = None
        self.index = None

    @abstractmethod
    def __iter__(self):
        pass

    @abstractmethod
    def __next__(self):
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass
