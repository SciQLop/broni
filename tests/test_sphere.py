#!/usr/bin/env python

import unittest
from ddt import ddt, data, unpack

import numpy as np
from astropy.units import km

from broni.shapes.primitives import Sphere
from broni import Trajectory


@ddt
class TestSphere(unittest.TestCase):
    def test_invalid_ctor_args_sphere_zero_radius(self):
        with self.assertRaises(ValueError):
            assert Sphere(*(0, 0, 0, 0) * km)

    def test_invalid_ctor_args_sphere_negative_radius(self):
        with self.assertRaises(ValueError):
            assert Sphere(*(0, 0, 0, -1) * km)

    @data(
        ([[1, 1, 1]], (0, 0, 0, 10),
         [True]),  # one point
        ([[0, 0, 10], [0, 10, 0], [10, 0, 0], [20, 0, 0]], (0, 0, 0, 10),
         [True, True, True, False]),  # some edge vectors
        ([[0, 0, -10], [0, -10, 0], [-10, 0, 0]], (0, 0, 0, 10),
         [True, True, True]),  # some edge vectors "on the other side"
        ([[0, 0, -11], [0, -11, 0], [-11, 0, 0]], (0, 0, 0, 10),
         [False, False, False]),  # just outside
    )
    @unpack
    def test_sphere_intersections(self, trajectory, shape_data, expected):
        shape = Sphere(*shape_data * km)
        td = np.array(trajectory) * km

        np.testing.assert_array_equal(
            shape.intersect(
                Trajectory(td[:, 0],
                           td[:, 1],
                           td[:, 2],
                           np.arange(0, len(trajectory)),
                           'gse')),
            np.array(expected))
