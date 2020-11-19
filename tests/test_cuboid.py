#!/usr/bin/env python

import unittest
from ddt import ddt, data, unpack

import numpy as np
from broni.shapes.primitives import Cuboid


@ddt
class TestCuboid(unittest.TestCase):
    def test_invalid(self):
        with self.assertRaises(ValueError):
            assert Cuboid(0, 0, 0, 0, 0, 0)

    @data(
        (np.array([[1, 1, 1]]), Cuboid(0, 0, 0, 10, 10, 10),
         np.array([True])),  # one point
        (np.array([[0, 0, 0]]), Cuboid(0, 0, 0, 10, 10, 10),
         np.array([True])),  # edge point
        (np.array([[-1, 0, 0]]), Cuboid(0, 0, 0, 10, 10, 10),
         np.array([False])),  # outside
        (np.array([[-1, 0, 0], [0, 0, 0], [1, 1, 1], [11, 11, 11]]), Cuboid(0, 0, 0, 10, 10, 10),
         np.array([False, True, True, False], dtype=bool)),  # multiple points
        (np.array([[-1, 0, 0]]), Cuboid(-10, -10, -10, 0, 0, 0),
         np.array([True])),  # a cuboid in another quadrant
        (np.array([]), Cuboid(0, 0, 0, 1, 1, 1),
         np.array([])),  # empty trajectory
        (np.array([[0, 0, 0], [3, 3, 3], [2.9, 2.9, 2.9], [10, 10, 10],
                   [10.1, 10.1, 10.1]]), Cuboid(10, 10, 10, 3, 3, 3),
         np.array([False, True, False, True, False], dtype=bool)),  # Cuboid points "inverted"
    )
    @unpack
    def test_intersections(self, trajectory, shape, expected):
        np.testing.assert_array_equal(shape.intersect(trajectory), expected)
