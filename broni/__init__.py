"""Top-level package for broni."""

__author__ = """Patrick Boettcher"""
__email__ = 'p@yai.se'
__version__ = '0.1.0'

import numpy as np


class Trajectory:
    def __init__(self,
                 trajectory: np.array,
                 coordinate_system: str):    # TODO maybe r_theta_phi-support?
        self.data = trajectory
        self.coordinate_system = coordinate_system
