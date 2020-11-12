#!/usr/bin/env python

import unittest
from ddt import ddt, data, unpack

import numpy as np
from broni.shapes.primitives import Sphere


@ddt
class TestSphere(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_invalid(self):
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
    def test_intersections(self, trajectory, shape, expected):
        np.testing.assert_array_equal(shape.intersect(trajectory), expected)

    @data(
        (Sphere(0, 0, 0, 10), [3, 20, 10])
    )
    @unpack
    def test_to_mesh(self, shape, expected):  # TODO, mesh is huge - function will be changed later on - most likely
        np.testing.assert_array_equal(shape.to_mesh().shape, expected)
