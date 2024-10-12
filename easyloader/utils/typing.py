from typing import Any


def is_hashable(obj: Any) -> bool:
    """
    Check whether an object is hashable. Works where isinstance(obj, Hashable)
    doesn't, for example ([],)

    :param obj: The object to check.
    :return: A bool indicating whether the object is hashable.
    """
    try:
        hash(obj)
        return True
    except TypeError:
        return False
