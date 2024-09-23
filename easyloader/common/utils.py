def get_n_batches(data_length, batch_size):
    n_batches, remainder = divmod(data_length, batch_size)
    if remainder > 0:
        n_batches += 1
    return n_batches


def get_tts_loaders(dataset_stub: str,
                    batch_size: int = 100,
                    keys: Sequence[str] = ['embedding', 'value'],
                    labels: Sequence[str] = ('train', 'test', 'val'),
                    data_fraction: float = 1.0,
                    seed: int = None,
                    shuffle: Union[bool, 'str'] = 'auto'):
    """
    E.g. get_data_loaders('../data/my_dataset')

    """
    data_files = {l: f'{dataset_stub}.{l}.h5' for l in labels}
    data_sets = {l: H5Dataset(file, keys=keys, sample_fraction=data_fraction, sample_seed=seed)
                 for l, file in data_files.items()}
    data_loaders = {
        l: DataLoader(ds, batch_size=batch_size, shuffle=((l == 'train') if shuffle == 'auto' else shuffle))
        for l, ds in data_sets.items()
    }
    return data_loaders
