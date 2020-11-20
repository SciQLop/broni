"""Top-level package for broni."""

__author__ = """Patrick Boettcher"""
__email__ = 'p@yai.se'
__version__ = '0.1.0'

import numpy as np

from astropy.coordinates import cartesian_to_spherical
from astropy import units

from typing import List, Union

from .shapes import Shape


class Trajectory:
    def __init__(self,
                 x: units.quantity.Quantity,
                 y: units.quantity.Quantity,
                 z: units.quantity.Quantity,
                 time_index: np.array,
                 coordinate_system: str):

        if len(x) != len(y) and len(y) != len(z):
            raise ValueError("x, y and z array must have the same number of elements")

        if len(x) != len(time_index):
            raise ValueError("trajectory data and time list must have the same number of elements")

        self._x = x
        self._y = y
        self._z = z
        self._r = None
        self._lat = None
        self._lon = None

        self._time_index = time_index
        self.coordinate_system = coordinate_system

    @property
    def time_index(self):
        return self._time_index

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def z(self):
        return self._z

    @property
    def r(self):
        self._spherical()
        return self._r

    @property
    def lat(self):
        self._spherical()
        return self._lat

    @property
    def lon(self):
        self._spherical()
        return self._lon

    @property
    def cartesian(self):
        return np.array((self._x,
                         self._y,
                         self._z)).T * self._x.unit

    def _spherical(self):
        if self._r is None:
            # xy = self._x ** 2 + self._y ** 2
            # self._r = np.sqrt(xy + self._z ** 2)
            # self._lon = np.arctan2(np.sqrt(xy), self._z)
            # self._lat = np.arctan2(self._y, self._x)
            self._r, self._lat, self._lon = cartesian_to_spherical(self._x, self._y, self._z)


def _listify(v):
    if type(v) in [list, tuple]:
        return v
    else:
        return [v]


def _index_list_to_ranges(indices: List[int]):
    if len(indices) == 0:
        return []
    split_idx = np.flatnonzero(np.diff(indices, prepend=indices[0], append=indices[-1]) != 1)
    bounding_points = np.transpose([split_idx[:-1], split_idx[1:]])
    return [range(indices[n], indices[m - 1]) for n, m in bounding_points]


def intervals(trajectory: Trajectory, shps: Union[List[Shape], Shape]):
    masks = [shape.intersect(trajectory) for shape in _listify(shps)]
    if len(masks) == 0:
        return []
    ranges = _index_list_to_ranges(np.where(np.logical_and.reduce(masks))[0])
    return [(trajectory.time_index[r.start], trajectory.time_index[r.stop]) for r in ranges]
