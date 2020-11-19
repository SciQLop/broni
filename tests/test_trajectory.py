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
        # simple, one dot trajectory, object-list
        ([[1, 1, 1]], [Cuboid(0, 0, 0, 2, 2, 2)], [[0, 0]]),

        # same single object (listified)
        ([[1, 1, 1]], Cuboid(0, 0, 0, 2, 2, 2), [[0, 0]]),

        # all points are outside
        ([[2, 2, 2]], Cuboid(0, 0, 0, 1, 1, 1), []),

        # no selection objects -> gives empty interval-liste
        ([[2, 2, 2]], [], []),

        # single point interval with multi-point trajectory
        ([[-1, -1, -1], [1, 1, 1]], Cuboid(0, 0, 0, 2, 2, 2), [[1, 1]]),

        # multi-point interval with multi-point trajectory
        ([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], Cuboid(0, 0, 0, 2, 2, 2), [[1, 2]]),

        # trajectory leaving et re-entering the cuboid -> two intervals
        ([[-1, -1, -1], [0, 0, 0], [1, 1, 1], [3, 3, 3], [1, 1, 1], [0, 0, 0]],
         [Cuboid(0, 0, 0, 2, 2, 2)], [[1, 2], [4, 5]]),

        # overlapping sphere and cuboid, logical_and between them
        ([[-1, -1, -1], [0, 0, 0], [1, 1, 1], [-1, -1, -1], [0, 0, 0], [1, 1, 1]],
         [Cuboid(0, 0, 0, 2, 2, 2), Sphere(1.5, 1.5, 1.5, 1)], [[2, 2], [5, 5]]),
    )
    @unpack
    def test_trajectory_intervals_with_primitive_objects(self, trajectory, objects, expected):
        np.testing.assert_array_equal(
            broni.intervals(
                broni.Trajectory(np.array(trajectory), np.arange(0, len(trajectory)), coordinate_system="GSE"),
                objects),
            np.array(expected))
