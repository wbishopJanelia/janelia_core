""" Module for working with point representation of spatial variables, such as cells. """


import numpy as np


class Point:

    def __init__(self, c: np.ndarray, cs_names: list = None):
        """ Create a new point object.

        Args:
            c: Of shape n_coord_systems*n_dims.  Each row gives the location of a point in a given coordinate
            system. For many applications, points may be specified in a single coordinate system in which case c
            would have only one row.

            cs_names: An optional list of names to associate with each coordinate system. If not none, the length of
            cs_names must equal the number of coordinate systems used to specify the position of the point.

        Raises:
            ValueError: If the length of cs_names does not the number of coordinate systems in c.
        """

        if c.ndim == 1:
            c = np.expand_dims(c, 0)

        n_coord_systems = c.shape[0]

        if cs_names is not None:
            if len(cs_names) != n_coord_systems:
                raise(ValueError('if cs_names is not None, cs_names must have exactly one entry for each coordinate system in c'))

        self.c = c
        self.cs_names = cs_names

    @classmethod
    def from_dict(cls, d:dict):
        """ Create a new point object from a dictionary.

        Args:
            d: A dictionary with the keys 'c' and 'cs_names'

        Returns:
            A new Point object
        """
        return cls(**d)

    def to_dict(self):
        """ Create a dictionary from a Point object.

        This is useful for saving the object in a manner which will still allow to be loaded in the
        future in case the class definition of Point changes.

        Returns:
            d: A dictionary with object data.
        """
        return vars(self)

