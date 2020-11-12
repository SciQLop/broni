"""Top-level package for broni."""

__author__ = """Patrick Boettcher"""
__email__ = 'p@yai.se'
__version__ = '0.1.0'

import numpy as np

from typing import List, Union

from .shapes import Shape


class Trajectory:
    def __init__(self,
                 trajectory: np.array,
                 time: np.array,
                 coordinate_system: str):  # TODO maybe r_theta_phi-support?
        if trajectory.shape[1] != 3:
            raise ValueError("trajectory data must consists of a matrix with 3 columns (X, Y, Z)")

        if len(trajectory) != len(time):
            raise ValueError("trajectory data and time list must have the same number of elements")

        self.data = trajectory
        self.time = time
        self.coordinate_system = coordinate_system


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
    return [range(indices[n], indices[m-1]) for n, m in bounding_points.tolist()]


def intervals(trajectory: Trajectory, shps: Union[List[Shape], Shape]):
    masks = [shape.intersect(trajectory) for shape in _listify(shps)]
    if len(masks) == 0:
        return []
    ranges = _index_list_to_ranges(np.where(np.logical_and.reduce(masks))[0])
    return [(trajectory.time[r.start], trajectory.time[r.stop]) for r in ranges]
