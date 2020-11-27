#!/usr/bin/env python

import unittest
from ddt import ddt, data, unpack

import numpy as np
from astropy.units import km
from astropy.units.quantity import Quantity

from broni.shapes.callback import SphericalBoundary, Sheath
import broni


class SphereModel:
    def __init__(self, radius: Quantity):
        self.r = radius

    def __call__(self, theta, phi, **kwargs):
        assert (kwargs['base'] == 'spherical' or kwargs['base'] == 'cartesian')

        if kwargs['base'] == 'spherical':
            return np.full(theta.shape, self.r.to('km').value) * km, theta, phi
        # else:
        #     return self.r * np.sin(theta) * np.cos(phi), \
        #            self.r * np.sin(theta) * np.sin(phi), \
        #            self.r * np.cos(theta)


@ddt
class TestCallbacks(unittest.TestCase):
    def test_boundary_invalid_ctor_args_no_lower_or_upper_bound_given(self):
        with self.assertRaises(ValueError):
            assert SphericalBoundary(SphereModel(1))

    def test_boundary_invalid_ctor_args_lower_bound_greater_than_upper_bound(self):
        with self.assertRaises(ValueError):
            assert SphericalBoundary(SphereModel(1), 1, 0)

    def test_boundary_invalid_ctor_args_upper_bound_greater_than_lower_bound(self):
        with self.assertRaises(ValueError):
            assert SphericalBoundary(SphereModel(1), 0, -1)

    def test_sheath_invalid_ctor_args_margin_inner_none(self):
        with self.assertRaises(ValueError):
            assert Sheath(SphereModel(1), SphereModel(2), None, 1)

    def test_sheath_invalid_ctor_args_margin_outer_none(self):
        with self.assertRaises(ValueError):
            assert Sheath(SphereModel(1), SphereModel(2), 1, None)

    def test_sheath_invalid_ctor_args_margin_inner_less_zero(self):
        with self.assertRaises(ValueError):
            assert Sheath(SphereModel(1), SphereModel(2), -1, 1)

    def test_sheath_invalid_ctor_args_margin_outer_less_zero(self):
        with self.assertRaises(ValueError):
            assert Sheath(SphereModel(1), SphereModel(2), 1, -1)

    def test_boundary_kwargs_forwarding_and_creation(self):
        def func(theta, phi, **kwargs):
            assert ('test' in kwargs)
            assert (kwargs['test'] == 'value')

            assert ('base' in kwargs)
            assert (kwargs['base'] == 'spherical')

            return np.zeros(theta.shape), theta, phi

        shape = SphericalBoundary(func, 0, test='value')
        shape.intersect(broni.Trajectory([] * km, [] * km, [] * km, [], 'gse'))

    @data(
        ((1, -0.5, 0.5), [[-1, 0, 0], [0, 0, 0], [1, 0, 0], [2, 0, 0]], [True, False, True, False]),
        ((1, None, 0.5), [[-1, 0, 0], [0, 0, 0], [1, 0, 0], [2, 0, 0]], [True, True, True, False]),
        ((1, -0.5, None), [[-1, 0, 0], [0, 0, 0], [1, 0, 0], [2, 0, 0]], [True, False, True, True]),
        ((1, -10, 10), [[-1, 0, 0], [0, 0, 0], [1, 0, 0], [2, 0, 0]], [True, True, True, True]),
    )
    @unpack
    def test_boundary_with_sphere_model_intersections(self, model, trajectory, expected):
        shape = SphericalBoundary(SphereModel(model[0] * km),
                                  model[1] * km if model[1] is not None else None,
                                  model[2] * km if model[2] is not None else None)

        td = np.array(trajectory) * km

        np.testing.assert_array_equal(
            shape.intersect(
                broni.Trajectory(
                    td[:, 0], td[:, 1], td[:, 2],
                    np.arange(0, len(trajectory)),
                    "gse")
            ),
            np.array(expected))

    @data(
        ((1, 2, 0, 0), [[-1, 0, 0], [0, 0, 0], [1, 0, 0], [2, 0, 0]], [True, False, True, True]),
        ((1, 1.5, 0, 0), [[-1, 0, 0], [0, 0, 0], [1, 0, 0], [2, 0, 0]], [True, False, True, False]),
        ((1, 1.5, 1, 0), [[-1, 0, 0], [0, 0, 0], [1, 0, 0], [2, 0, 0]], [True, True, True, False]),
        ((1, 1.5, 0, 1), [[-1, 0, 0], [0, 0, 0], [1, 0, 0], [2, 0, 0]], [True, False, True, True]),
        ((1, 1.5, 1, 1), [[-1, 0, 0], [0, 0, 0], [1, 0, 0], [2, 0, 0]], [True, True, True, True]),

        # inverse inner/outer -> nothing
        ((1.5, 1, 0, 0), [[-1, 0, 0], [0, 0, 0], [1, 0, 0], [2, 0, 0]], [False, False, False, False]),
    )
    @unpack
    def test_sheath_with_2_spheres_intersections(self, model, trajectory, expected):
        shape = Sheath(SphereModel(model[0] * km),
                       SphereModel(model[1] * km),
                       model[2] * km,
                       model[3] * km)

        td = np.array(trajectory) * km

        np.testing.assert_array_equal(
            shape.intersect(
                broni.Trajectory(
                    td[:, 0], td[:, 1], td[:, 2],
                    np.arange(0, len(trajectory)),
                    "gse")
            ),
            np.array(expected))
