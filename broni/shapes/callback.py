from . import Shape
from .. import Trajectory

from functools import partial
import numpy as np
from astropy.units.quantity import Quantity
from astropy.constants import R_earth
import astropy.units as u

from typing import Callable


class SphericalBoundary(Shape):
    """
    Calculates intersections of a trajectory with a spherical boundary.

    A callback-function has to be provided which is called with theta and phi angles (as np.array)
    and a keyword argument requesting spherical coordinates (r, thetha, phi) in return ('base'="spherical").

    kwargs passed to the constructor are forwarded to the callback-function (except for the base-keyword).

    A scale-factor (as a astropy.Quantity) can be provided. By default it is kilometer. This factor is
    applied to the radius-component returned by the callback-function.

    Optionally an upper and/or a lower bound can be specified to define a range around the boundary. In practice
    at least one of them should be defined - trajectory points are rarely exactly on the boundary.

    lower-bound has to be less than upper-bound if upper-bound is specified. A positive  lower-bound means that
    the selected range is actually outside the spherical object. A negative upper-bound means the range is
    inside the object.
    """

    def __init__(self, callback: Callable,
                 lower_bound: Quantity = None,
                 upper_bound: Quantity = None,
                 scale: Quantity = 1 * u.km,
                 **kwargs):
        if lower_bound is None and upper_bound is None:
            raise ValueError("At least of one of lower or upper bound has to be specified.")

        if scale is None:
            raise ValueError("Given scale-factor cannot be None, has to be a astropy.Quantity")

        self._lower = lower_bound
        self._upper = upper_bound
        self._scale = scale

        if self._lower is not None and self._upper is not None:
            if self._lower > self._upper:
                raise ValueError(
                    f"lower-bound-value ({lower_bound}) needs to be lower than upper-bound-value ({upper_bound})")

        kwargs.update({'base': 'spherical'})  # force spherical basis, overriding user's request
        self._cb = partial(callback, **kwargs)

    def intersect(self, trajectory: Trajectory):
        distances = trajectory.r - self._cb(trajectory.lon, trajectory.lat)[0] * self._scale

        print(distances, self._lower, self._upper, trajectory.r)

        return (distances >= self._lower if self._lower is not None else True) & \
               (distances <= self._upper if self._upper is not None else True)


class Sheath(Shape):
    """
    Intersections of a trajectory with a sheath-object are effectively all the points which are
    in-between two spherical-boundaries. This class does exactly that, it takes two callbacks representing
    the inner and outer boundary, creates two SphericalBoundary-instances and finds all the corresponding points.

    In addition an inner and/or an outer margin can be specified which will also find points just outside
    the sheath within the margin.
    """

    def __init__(self,
                 inner_callback: Callable,
                 outer_callback: Callable,
                 inner_margin: Quantity = 0,
                 outer_margin: Quantity = 0,
                 scale: Quantity = 1 * u.km,
                 **kwargs):
        if inner_margin is None or outer_margin is None or inner_margin < 0 or outer_margin < 0:
            raise ValueError("The margins have to be larger or equal to zero if specified.")

        self.inner_model = SphericalBoundary(inner_callback, -inner_margin, None, scale, **kwargs)
        self.outer_model = SphericalBoundary(outer_callback, None, outer_margin, scale, **kwargs)

    def intersect(self, trajectory: Trajectory):
        inner_mask = self.inner_model.intersect(trajectory)
        outer_mask = self.outer_model.intersect(trajectory)

        return np.logical_and(inner_mask, outer_mask)
