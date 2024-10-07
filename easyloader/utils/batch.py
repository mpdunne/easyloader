import math


def get_n_batches(data_length: int, batch_size: int) -> int:
    """
    Get number of batches, based on the data length and the batch size.

    :param data_length:
    :param batch_size:
    :return: The number of batches
    """
    if not (isinstance(data_length, int) and data_length >= 0):
        raise ValueError('Haha nice try. That\'s not a data length!')

    if not (isinstance(batch_size, int) and batch_size > 0):
        raise ValueError('Batch size must be a positive integer.')

    return int(math.ceil(data_length / batch_size))

