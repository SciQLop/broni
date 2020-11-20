#!/usr/bin/env python

import unittest
from ddt import ddt, data, unpack

import numpy as np
from astropy.units import km

import broni
from broni.shapes.primitives import Cuboid, Sphere


@ddt
class TestTrajectory(unittest.TestCase):
    def test_invalid_ctor_args_with_different_x_y_length(self):
        with self.assertRaises(ValueError):
            assert broni.Trajectory(
                [1], [2, 2], [1],
                [],
                coordinate_system="gse")

    def test_invalid_ctor_args_with_different_x_z_length(self):
        with self.assertRaises(ValueError):
            assert broni.Trajectory(
                [1], [2], [1, 2],
                [],
                coordinate_system="gse")

    def test_invalid_ctor_args_with_different_y_z_length(self):
        with self.assertRaises(ValueError):
            assert broni.Trajectory(
                [1], [2], [1, 2],
                [],
                coordinate_system="gse")

    def test_invalid_ctor_args_time_series_has_to_match_trajectory_length(self):
        with self.assertRaises(ValueError):
            assert broni.Trajectory(
                [0, 0] * km, [0, 0] * km, [0, 0] * km,
                [1, 2, 3],
                coordinate_system="gse")

    def test_empty_trajectory(self):
        np.testing.assert_array_equal(
            broni.intervals(
                broni.Trajectory([] * km, [] * km, [] * km,
                                 [],
                                 'gse'),
                Cuboid(*(0, 0, 0, 2, 2, 2) * km)),
            [])

    @data(
        # simple, one dot trajectory, object-list
        ([[1, 1, 1]], [Cuboid(*(0, 0, 0, 2, 2, 2) * km)], [[0, 0]]),

        # same single object (listified)
        ([[1, 1, 1]], Cuboid(*(0, 0, 0, 2, 2, 2) * km), [[0, 0]]),

        # all points are outside
        ([[2, 2, 2]], Cuboid(*(0, 0, 0, 1, 1, 1) * km), []),

        # no selection objects -> gives empty interval-list
        ([[2, 2, 2]], [], []),

        # single point interval with multi-point trajectory
        ([[-1, -1, -1], [1, 1, 1]], Cuboid(*(0, 0, 0, 2, 2, 2) * km), [[1, 1]]),

        # multi-point interval with multi-point trajectory
        ([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], Cuboid(*(0, 0, 0, 2, 2, 2) * km), [[1, 2]]),

        # trajectory leaving et re-entering the cuboid -> two intervals
        ([[-1, -1, -1], [0, 0, 0], [1, 1, 1], [3, 3, 3], [1, 1, 1], [0, 0, 0]],
         [Cuboid(*(0, 0, 0, 2, 2, 2) * km)], [[1, 2], [4, 5]]),

        # overlapping sphere and cuboid, logical_and between them
        ([[-1, -1, -1], [0, 0, 0], [1, 1, 1], [-1, -1, -1], [0, 0, 0], [1, 1, 1]],
         [Cuboid(*(0, 0, 0, 2, 2, 2) * km), Sphere(*(1.5, 1.5, 1.5, 1) * km)], [[2, 2], [5, 5]]),
    )
    @unpack
    def test_trajectory_intervals_with_primitive_objects_no_time_index(self, trajectory, objects, expected):
        td = np.array(trajectory) * km

        np.testing.assert_array_equal(
            broni.intervals(
                broni.Trajectory(td[:, 0], td[:, 1], td[:, 2],
                                 np.arange(0, len(trajectory)),
                                 'gse'),
                objects),
            np.array(expected))

    @data(
        # multi-point interval with multi-point trajectory
        ([[-1, -1, -1], [0, 0, 0], [1, 1, 1]],
         [1000, 1001, 1002],
         Cuboid(*(0, 0, 0, 2, 2, 2) * km),
         [[1001, 1002]])
    )
    @unpack
    def test_trajectory_intervals_with_dedicated_time_index(self, trajectory, time, objects, expected):
        td = np.array(trajectory) * km

        np.testing.assert_array_equal(
            broni.intervals(
                broni.Trajectory(td[:, 0], td[:, 1], td[:, 2], time, 'gse'),
                objects),
            np.array(expected))
