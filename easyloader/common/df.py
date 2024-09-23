def sample_df(df: pd.DataFrame,
              n_samples: int = None,
              shuffle: bool = False,
              sample_seed: int = None,
              shuffle_seed: int = None) -> pd.DataFrame:

    if n_samples is not None and n_samples != len(df):
        random.seed(sample_seed)
        sample = random.sample([*range(len(df))], n_samples)
        sample = sorted(sample)
        df = df.iloc[sample]

    if shuffle:
        df = df.sample(frac=1, random_state=shuffle_seed)

    return df

