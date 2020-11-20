#!/usr/bin/env python

import unittest
from ddt import ddt, data, unpack

import numpy as np

from broni.shapes.primitives import Sphere
from broni import Trajectory


@ddt
class TestSphere(unittest.TestCase):
    def test_invalid_ctor_args_sphere_zero_radius(self):
        with self.assertRaises(ValueError):
            assert Sphere(0, 0, 0, 0)

    def test_invalid_ctor_args_sphere_negative_radius(self):
        with self.assertRaises(ValueError):
            assert Sphere(0, 0, 0, -1)

    @data(
        ([[1, 1, 1]], Sphere(0, 0, 0, 10),
         [True]),  # one point
        ([[0, 0, 10], [0, 10, 0], [10, 0, 0]], Sphere(0, 0, 0, 10),
         [True, True, True]),  # some edge vectors
        ([[0, 0, -10], [0, -10, 0], [-10, 0, 0]], Sphere(0, 0, 0, 10),
         [True, True, True]),  # some edge vectors "on the other side"
        ([[0, 0, -11], [0, -11, 0], [-11, 0, 0]], Sphere(0, 0, 0, 10),
         [False, False, False]),  # just outside
    )
    @unpack
    def test_sphere_intersections(self, trajectory, shape, expected):
        np.testing.assert_array_equal(
            shape.intersect(Trajectory(np.array(trajectory),
                                       np.arange(0, len(trajectory)),
                                       "gse")),
            np.array(expected))
