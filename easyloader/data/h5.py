
def check_keys(data_keys, requested_keys, allow_missing_keys=False):
    present_keys = []
    missing_keys = []
    for key in requested_keys:
        if key in data_keys:
            present_keys.append(key)
        else:
            missing_keys.append(key)

    if missing_keys and not allow_missing_keys:
        missing_key_string = ', '.join(missing_keys)
        raise KeyError(f'The following keys are missing from the h5 file: {missing_key_string}. '
                       'If you don\'t care, set allow_missing_keys to True.')

    if not present_keys:
        raise KeyError('None of the provided keys are present in the H5 file. Need at least one.')

    return present_keys
