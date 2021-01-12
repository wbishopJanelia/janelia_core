""" Helper functions and objects for searching and finding stings, values, etc in differt types of objects. """


def dict_find(d: dict, vl: object):
    """ Searches a dictionary's fields, which must be sequences, and returns key to a field containing the value.

    This function assumes each field in the dictionary is a sequence.  It then searches the sequence associated with
    each key.  It returns the first key for which a matching value is found in its sequence.  If the value is not
    found in the sequences for any key, None is returned.

    Args:

        d: The dictionary to search.  Each field must be a sequence.

        vl: The value to search for equality for

    Returns:

        k: The key of the fist field found with a match.  If not match is found, None is returned.

    """

    keys = d.keys()
    for k in keys:
        vls = d[k]
        for v in vls:
            if v == vl:
                return k
    return None

