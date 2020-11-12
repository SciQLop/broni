#!/usr/bin/env python3

from broni.shapes.primitives import Cuboid, Sphere
import broni

from spwc import sscweb
from spwc.common import variable

import datetime
import matplotlib.pyplot as plt

import numpy as np


def cuboid_mesh(cuboid):
    p1 = cuboid.p1
    p2 = cuboid.p2
    p3 = cuboid.p3
    p4 = cuboid.p4

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


def sphere_mesh(sphere):
    u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
    x = sphere.center[0] + sphere.radius * np.cos(u) * np.sin(v)
    y = sphere.center[1] + sphere.radius * np.sin(u) * np.sin(v)
    z = sphere.center[2] + sphere.radius * np.cos(v)

    return np.array((x, y, z), dtype=float)


if __name__ == '__main__':
    ssc = sscweb.SscWeb()
    sv = ssc.get_orbit(product="mms1",
                       start_time="2020-10-10",
                       stop_time="2020-10-24",
                       coordinate_systems="GSE")
    df = sv.to_dataframe()

    orbit = broni.Trajectory(df.values[::2, 0:3],
                             df.index[::2],
                             coordinate_system="GSE")

    # sphere = Sphere(30000, 30000, 30000, 15000)
    # intervals = broni.intervals(orbit, sphere)

    # cuboid = Cuboid(10000, 10000, 10000,
    #                25000, 25000, 25000)
    # intervals += broni.intervals(orbit, cuboid)

    sphere = Sphere(30000, 30000, 30000, 25000)
    cuboid = Cuboid(10000, 10000, 10000,
                    25000, 25000, 25000)
    intervals = broni.intervals(orbit, [sphere, cuboid])

    print('found', len(intervals), 'intervals')
    for i in sorted(intervals):
        print('  ', i,
              datetime.datetime.fromtimestamp(i[0]).strftime('%c'), '-',
              datetime.datetime.fromtimestamp(i[1]).strftime('%c'), ';',
              i[0], '-', i[1])

    # intersection points
    slice1 = variable.merge([sv[interval[0]:interval[1]] for interval in intervals]).data

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.plot(df.values[:, 0], df.values[:, 1], df.values[:, 2], color='r')
    try:
        ax.scatter(slice1[:, 0], slice1[:, 1], slice1[:, 2], color='g')
    except:
        pass

    ax.set_xlim3d(5000, 32000)
    ax.set_ylim3d(5000, 32000)
    ax.set_zlim3d(5000, 32000)

    o = sphere_mesh(sphere)
    # ax.plot_wireframe(o[0], o[1], o[2], color="b")
    ax.plot_surface(o[0], o[1], o[2], color="b", alpha=0.2)

    o = cuboid_mesh(cuboid)
    ax.plot_surface(o[0], o[1], o[2], color="b", alpha=0.2)
    # ax.plot_wireframe(o[0], o[1], o[2], color="b")

    plt.show()
