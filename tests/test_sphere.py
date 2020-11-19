#!/usr/bin/env python

import unittest
from ddt import ddt, data, unpack

import numpy as np
from broni.shapes.primitives import Sphere


@ddt
class TestSphere(unittest.TestCase):
    def test_invalid_ctor_args_raise_exceptions(self):
        with self.assertRaises(ValueError):
            assert Sphere(0, 0, 0, 0)  # zero radius

        with self.assertRaises(ValueError):
            assert Sphere(0, 0, 0, -1)  # negative radius

    @data(
        (np.array([[1, 1, 1]]), Sphere(0, 0, 0, 10),
         np.array([True])),  # one point
        (np.array([[0, 0, 10], [0, 10, 0], [10, 0, 0]]), Sphere(0, 0, 0, 10),
         np.array([True, True, True])),  # some edge vectors
        (np.array([[0, 0, -10], [0, -10, 0], [-10, 0, 0]]), Sphere(0, 0, 0, 10),
         np.array([True, True, True])),  # some edge vectors "on the other side"
        (np.array([[0, 0, -11], [0, -11, 0], [-11, 0, 0]]), Sphere(0, 0, 0, 10),
         np.array([False, False, False])),  # just outside
    )
    @unpack
    def test_sphere_intersections(self, trajectory, shape, expected):
        np.testing.assert_array_equal(shape.intersect(trajectory), expected)
