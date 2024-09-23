def sample_ixs(ixs: Sequence[int],
               n_samples: int = None,
               shuffle: bool = False,
               sample_seed: int = None,
               shuffle_seed: int = None) -> Sequence[int]:

    if n_samples != None and n_samples != len(ixs):
        random.seed(sample_seed)
        ixs = random.sample([*range(len(ixs))], n_samples)
        ixs = sorted(ixs)

    if shuffle:
        random.seed(shuffle_seed)
        ixs = random.sample(ixs, len(ixs))

    return ixs