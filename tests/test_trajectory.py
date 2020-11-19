#!/usr/bin/env python

import unittest
from ddt import ddt, data, unpack

import numpy as np

import broni
from broni.shapes.primitives import Cuboid, Sphere


@ddt
class TestTrajectory(unittest.TestCase):
    def test_invalid_ctor_args_raise_exceptions(self):
        with self.assertRaises(ValueError):
            assert broni.Trajectory(
                np.array([[0, 0], [0, 0]], dtype=float),
                [2, 2],
                coordinate_system="GSE"
            )

        with self.assertRaises(ValueError):
            assert broni.Trajectory(
                np.array([[0, 0, 0], [0, 0, 0]], dtype=float),
                [1, 2, 3],
                coordinate_system="GSE"
            )

    @data(
        (np.array([[1, 1, 1]]), [Cuboid(0, 0, 0, 2, 2, 2)], np.array([[0, 0]])), # simple, one dot trajectory, objet-list
        (np.array([[1, 1, 1]]), Cuboid(0, 0, 0, 2, 2, 2), np.array([[0, 0]])),  # single object (listified)

        (np.array([[2, 2, 2]]), [Cuboid(0, 0, 0, 1, 1, 1)], np.array([])),  # all points are outside

        (np.array([[2, 2, 2]]), [], np.array([])),  # no selection objects

        (np.array([[-1, -1, -1], [1, 1, 1]]), [Cuboid(0, 0, 0, 2, 2, 2)], np.array([[1, 1]])),
        (np.array([[-1, -1, -1], [1, 1, 1]]), [Cuboid(0, 0, 0, 2, 2, 2)], np.array([[1, 1]])),
        (np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]), [Cuboid(0, 0, 0, 2, 2, 2)], np.array([[1, 2]])),

        (np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1], [-1, -1, -1], [0, 0, 0], [1, 1, 1]]),
         [Cuboid(0, 0, 0, 2, 2, 2)], np.array([[1, 2], [4, 5]])),

        (np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1], [-1, -1, -1], [0, 0, 0], [1, 1, 1]]),
         [Cuboid(0, 0, 0, 2, 2, 2), Sphere(1.5, 1.5, 1.5 , 1)], np.array([[2, 2], [5, 5]])),  # overlapping sphere and cuboid, logical_and
    )
    @unpack
    def test_trajectory_intervals_with_primitive_objects(self, trajectory, objects, expected):
        np.testing.assert_array_equal(
            broni.intervals(
                broni.Trajectory(trajectory, np.arange(0, len(trajectory)), coordinate_system="GSE"),
                objects),
            expected)
