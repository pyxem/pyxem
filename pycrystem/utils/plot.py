# -*- coding: utf-8 -*-
# Copyright 2017 The PyCrystEM developers
#
# This file is part of PyCrystEM.
#
# PyCrystEM is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PyCrystEM is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PyCrystEM.  If not, see <http://www.gnu.org/licenses/>.

import itertools
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.axisartist.floating_axes import GridHelperCurveLinear, \
    FloatingSubplot
from matplotlib.projections import PolarAxes
from matplotlib.transforms import Affine2D
from scipy.interpolate import griddata
from pymatgen.transformations.standard_transformations \
    import RotationTransformation
from transforms3d.euler import euler2axangle

# from . import Structure, ElectronDiffractionCalculator
from .utils import correlate

from ipywidgets import interact


def manual_orientation(data,  #: np.ndarray,
                       structure,  #: Structure,
                       calculator,  #: ElectronDiffractionCalculator,
                       ax=None):
    """
    
    """
    if ax is None:
        ax = plt.figure().add_subplot(111)
    dimension = data.shape[0] / 2
    extent = [-dimension, dimension] * 2
    ax.imshow(data, extent=extent, interpolation='none', origin='lower')
    text = plt.text(dimension, dimension, "Loading...")
    p = plt.scatter([0, ], [0, ], s=0)

    def plot(alpha=0., beta=0., gamma=0., calibration=1., reciprocal_radius=1.0):
        calibration /= 100
        orientation = euler2axangle(alpha, beta, gamma, 'rzyz')
        rotation = RotationTransformation(orientation[0], orientation[1],
                                          angle_in_radians=True).apply_transformation(
            structure)
        electron_diffraction = calculator.calculate_ed_data(rotation, reciprocal_radius)
        electron_diffraction.calibration = calibration
        nonlocal p
        p.remove()
        p = plt.scatter(
            electron_diffraction.calibrated_coordinates[:, 0],
            electron_diffraction.calibrated_coordinates[:, 1],
            s=electron_diffraction.intensities,
            facecolors='none',
            edgecolors='r'
        )
        text.set_text('\n'.join([str(correlate(data, electron_diffraction)), str(calibration)]))
        ax.set_xlim(-dimension, dimension)
        ax.set_ylim(-dimension, dimension)
        plt.show()

    interact(plot, alpha=(-np.pi, np.pi, 0.01), beta=(-np.pi, np.pi, 0.01),
             gamma=(-np.pi, np.pi, 0.01), calibration=(1e-2, 1e1, 1e-2),
             reciprocal_radius=(1e-1, 5., 1e-1))

#HACK
def find_max_length_peaks(peaks):
    x_size,y_size = peaks.axes_manager.navigation_shape[0],peaks.axes_manager.navigation_shape[1]
    length_of_longest_peaks_list = 0
    for x in np.arange(0,x_size):
            for y in np.arange(0,y_size):
                if peaks.data[x,y].shape[0] > length_of_longest_peaks_list:
                    length_of_longest_peaks_list = peaks.data[x,y].shape[0]
    return length_of_longest_peaks_list  


def generate_marker_inputs_from_peaks(peaks):
   
    max_peak_len = find_max_length_peaks(peaks)
    #Hardcoded into 2D
    pad = np.array(list(itertools.zip_longest(*np.concatenate(peaks.data),fillvalue=[np.nan,np.nan])))
    # Not tested for non-square signal, return flat to square
    pad = pad.reshape((max_peak_len),peaks.data.shape[0],peaks.data.shape[1],2)
    xy_cords = np.transpose(pad,[3,0,1,2]) #move the x,y pairs to the front 
    #taking care with x,y ordering
    x = xy_cords[1] 
    y = xy_cords[0]
    
    return x,y 