
def get_n_batches(data_length, batch_size):
    n_batches, remainder = divmod(data_length, batch_size)
    if remainder > 0:
        n_batches += 1
    return n_batches
