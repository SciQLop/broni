#!/usr/bin/env python3


from broni.shapes.primitives import Cuboid
from broni.shapes.callback import SphericalBoundary, Sheath
import broni

from astropy.units import km
from astropy.constants import R_earth

import datetime
import matplotlib.pyplot as plt

import numpy as np

from space.models.planetary import formisano1979, mp_formisano1979, bs_formisano1979


def cuboid_mesh(cuboid):
    p1 = cuboid.p1.value
    p2 = cuboid.p2.value
    p3 = cuboid.p3.value
    p4 = cuboid.p4.value

    X = [[p1[0], p2[0], p2[0], p1[0], p1[0]],
         [p1[0], p2[0], p2[0], p1[0], p1[0]],
         [p1[0], p2[0], p2[0], p1[0], p1[0]],
         [p1[0], p2[0], p2[0], p1[0], p1[0]]]

    Y = [[p1[1], p1[1], p3[1], p3[1], p1[1]],
         [p1[1], p1[1], p3[1], p3[1], p1[1]],
         [p1[1], p1[1], p1[1], p1[1], p1[1]],
         [p3[1], p3[1], p3[1], p3[1], p3[1]]]

    Z = [[p1[2], p1[2], p1[2], p1[2], p1[2]],
         [p4[2], p4[2], p4[2], p4[2], p4[2]],
         [p1[2], p1[2], p4[2], p4[2], p1[2]],
         [p1[2], p1[2], p4[2], p4[2], p1[2]]]

    return np.array((X, Y, Z), dtype=float)


class SphereModel:
    def __init__(self, radius: float):
        self.r = radius

    def __call__(self, theta, phi, **kwargs):
        base = kwargs.get('base', 'catersian')

        if base == 'spherical':
            return np.full(theta.shape, self.r.to('km').value) * km, theta, phi
        else:
            return self.r * np.sin(theta) * np.cos(phi), \
                   self.r * np.sin(theta) * np.sin(phi), \
                   self.r * np.cos(theta)


if __name__ == '__main__':
    coord_sys = "gse"

    X = np.arange(-200000, 200000, 10).flatten() * km

    orbit = broni.Trajectory(np.zeros(X.shape) * km,
                             X,
                             np.zeros(X.shape) * km,
                             np.arange(0, len(X)),
                             coordinate_system=coord_sys)

    boundary_model = SphericalBoundary(mp_formisano1979, 0 * R_earth, 1 * R_earth, scale=R_earth)
    # boundary_model = Sheath(mp_formisano1979, bs_formisano1979)

    # boundary_model = SphericalBoundary(SphereModel(100000 * km), -1 * R_earth, 0)
    # boundary_model = Sheath(SphereModel(100000 * km), SphereModel(110000 * km))

    intervals = broni.intervals(orbit, [boundary_model])

    print('found', len(intervals), 'intervals')
    for i in sorted(intervals):
        print('  ', i,
              datetime.datetime.fromtimestamp(i[0]).strftime('%c'), '-',
              datetime.datetime.fromtimestamp(i[1]).strftime('%c'), ';',
              i[0], '-', i[1])

    # plot
    th_1d = np.linspace(0, np.pi * 0.75, 20)
    ph_1d = np.linspace(0, 2 * np.pi, 20)
    th, ph = np.meshgrid(th_1d, ph_1d, indexing='ij')

    if boundary_model.__class__ == SphericalBoundary:
        x, y, z = boundary_model._cb.func(th, ph) * R_earth.to('km')
    else:
        x, y, z = boundary_model.inner_model._cb.func(th, ph) * R_earth.to('km')
        xb, yb, zb = boundary_model.outer_model._cb.func(th, ph) * R_earth.to('km')


    def make_fig(elev=None, azim=None, limit=False):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        # msphere
        ax.plot_surface(x, y, z, alpha=0.2, color='b')
        # bowshock
        if boundary_model.__class__ == Sheath:
            ax.plot_surface(xb, yb, zb, alpha=0.3, color='r')

        # orbit
        ax.plot(orbit.x, orbit.y, orbit.z, color='r')

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        if not None in [elev, azim]:
            ax.view_init(elev, azim)

        # intersection points
        for i in intervals:
            ax.plot(orbit.x[i[0]:i[1]],
                    orbit.y[i[0]:i[1]],
                    orbit.z[i[0]:i[1]],
                    color='g', linewidth=3)
            if limit:
                ax.set_xlim3d(min(orbit.x[i[0]:i[1]]) - R_earth.value, max(orbit.x[i[0]:i[1]]) + R_earth.value)
                ax.set_ylim3d(min(orbit.y[i[0]:i[1]]) - R_earth.value, max(orbit.y[i[0]:i[1]]) + R_earth.value)
                ax.set_zlim3d(min(orbit.z[i[0]:i[1]]) - R_earth.value, max(orbit.z[i[0]:i[1]]) + R_earth.value)

            len = np.sqrt(orbit.x[i[0]:i[1]] ** 2 + orbit.y[i[0]:i[1]] ** 2 + orbit.z[i[0]:i[1]] ** 2)
            # print(np.min(len), np.max(len)) #np.max(len) - np.min(len) - R_earth)
        return fig


    make_fig(limit=False)
    # make_fig(0, 0, False)  # zy
    # make_fig(90, 0, False)  # yx
    # make_fig(0, 90, False)  # xz

    plt.show()
