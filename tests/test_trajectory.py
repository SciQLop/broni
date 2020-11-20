#!/usr/bin/env python

import unittest
from ddt import ddt, data, unpack

import numpy as np

import broni
from broni.shapes.primitives import Cuboid, Sphere


@ddt
class TestTrajectory(unittest.TestCase):
    def test_invalid_ctor_args_need_3_columns_are_needed(self):
        with self.assertRaises(ValueError):
            assert broni.Trajectory(
                np.array([[0, 0], [0, 0]], dtype=float),
                [2, 2],
                coordinate_system="gse"
            )

    def test_invalid_ctor_args_need_2_dim_array(self):
        with self.assertRaises(ValueError):
            assert broni.Trajectory(
                np.array([0, 0, 0], dtype=float),
                [2, 2, 2],
                coordinate_system="gse"
            )

    def test_invalid_ctor_args_empty_trajectory_not_supported(self):
        with self.assertRaises(ValueError):
            assert broni.Trajectory(
                np.array([[]], dtype=float),
                [],
                coordinate_system="gse"
            )

    def test_invalid_ctor_args_time_series_has_to_match_trajectory_length(self):
        with self.assertRaises(ValueError):
            assert broni.Trajectory(
                np.array([[0, 0, 0], [0, 0, 0]], dtype=float),
                [1, 2, 3],
                coordinate_system="gse"
            )

    @data(
        # simple, one dot trajectory, object-list
        ([[1, 1, 1]], [Cuboid(0, 0, 0, 2, 2, 2)], [[0, 0]]),

        # same single object (listified)
        ([[1, 1, 1]], Cuboid(0, 0, 0, 2, 2, 2), [[0, 0]]),

        # all points are outside
        ([[2, 2, 2]], Cuboid(0, 0, 0, 1, 1, 1), []),

        # no selection objects -> gives empty interval-list
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
    def test_trajectory_intervals_with_primitive_objects_no_time_index(self, trajectory, objects, expected):
        np.testing.assert_array_equal(
            broni.intervals(
                broni.Trajectory(np.array(trajectory),
                                 np.arange(0, len(trajectory)),
                                 'gse'),
                objects),
            np.array(expected))

    @data(
        # multi-point interval with multi-point trajectory
        ([[-1, -1, -1], [0, 0, 0], [1, 1, 1]],
         [1000, 1001, 1002],
         Cuboid(0, 0, 0, 2, 2, 2),
         [[1001, 1002]])
    )
    @unpack
    def test_trajectory_intervals_with_dedicated_time_index(self, trajectory, time, objects, expected):
        np.testing.assert_array_equal(
            broni.intervals(
                broni.Trajectory(np.array(trajectory), time, 'gse'),
                objects),
            np.array(expected))
