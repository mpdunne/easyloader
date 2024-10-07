import numpy as np

from pathlib import Path
from typing import Sequence, Union

from easyloader.dataset.base import EasyDataset
from easyloader.data.h5 import H5Data
from easyloader.utils.grains import grab_slices_from_grains


class H5Dataset(EasyDataset):
    """
    Turn a H5 file into a PyTorch Data Set.

    """

    def __init__(self,
                 data_path: Union[str, Path],
                 keys: Sequence[str],
                 id_key: str = None,
                 grain_size: int = 1,
                 sample_fraction: float = 1.0,
                 sample_seed: int = None):

        """
        Constructor for the H5Dataset class.

        :param data_path: The path to the H5 file that you want to load.
        :param keys: The keys that you want to grab.
        :param sample_fraction: Fraction of the dataset to sample.
        :param sample_seed: Seed for random sampling.
        """

        # Initialize the parent class
        super().__init__(sample_fraction=sample_fraction,
                         sample_seed=sample_seed)

        self.data = H5Data(data_path, keys=keys, id_key=id_key, grain_size=grain_size,
                           sample_fraction=sample_fraction, sample_seed=sample_seed)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, ix: Union[int, slice]):

        values = []

        if isinstance(ix, int):
            for key in self.data.keys:
                values.append(self.data.h5[key][self.data.index[ix]])

        elif isinstance(ix, slice):
            ix_slices = grab_slices_from_grains(self.data.grain_index, self.data.grain_size, ix.start, ix.stop)
            for key in self.data.keys:
                values.append(np.concatenate([self.data.h5[key][ix_slice] for ix_slice in ix_slices]))

        else:
            raise ValueError('Index ix must either be an int or a slice.')

        return tuple(values)
