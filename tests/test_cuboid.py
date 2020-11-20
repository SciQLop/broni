#!/usr/bin/env python

import unittest
from ddt import ddt, data, unpack

import numpy as np
from astropy.units import km

from broni.shapes.primitives import Cuboid
from broni import Trajectory


@ddt
class TestCuboid(unittest.TestCase):
    def test_invalid_ctor_args_cuboid_without_volume(self):
        with self.assertRaises(ValueError):
            assert Cuboid(0 * km, 0 * km, 0 * km, 0 * km, 0 * km, 0 * km)

    @data(
        ([[1, 1, 1]], (0, 0, 0, 10, 10, 10),
         [True]),  # one point
        ([[0, 0, 0]], (0, 0, 0, 10, 10, 10),
         [True]),  # edge point
        ([[-1, 0, 0]], (0, 0, 0, 10, 10, 10),
         [False]),  # outside
        ([[-1, 0, 0], [0, 0, 0], [1, 1, 1], [11, 11, 11]], (0, 0, 0, 10, 10, 10),
         [False, True, True, False]),  # multiple points
        ([[-1, 0, 0]], (-10, -10, -10, 0, 0, 0),
         [True]),  # a cuboid in another quadrant
        ([[0, 0, 0], [3, 3, 3], [2.9, 2.9, 2.9], [10, 10, 10],
          [10.1, 10.1, 10.1]], (10, 10, 10, 3, 3, 3),
         [False, True, False, True, False]),  # Cuboid points "inverted"
    )
    @unpack
    def test_cuboid_intersections(self, trajectory, shape_data, expected):
        shape = Cuboid(*shape_data * km)
        td = np.array(trajectory) * km

        np.testing.assert_array_equal(
            shape.intersect(
                Trajectory(td[:, 0],
                           td[:, 1],
                           td[:, 2],
                           np.arange(0, len(trajectory)),
                           'gse')),
            np.array(expected))
