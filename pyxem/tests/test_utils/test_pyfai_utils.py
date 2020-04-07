# -*- coding: utf-8 -*-
# Copyright 2017-2020 The pyXem developers
#
# This file is part of pyXem.
#
# pyXem is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# pyXem is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with pyXem.  If not, see <http://www.gnu.org/licenses/>.

import pytest

from pyxem.utils.pyfai_utils import _get_radial_extent,get_azimuthal_integrator,_get_displacements,_get_curved_setup, _get_flat_setup
from pyFAI.detectors import Detector
from pyFAI.azimuthalIntegrator import AzimuthalIntegrator
import numpy as np

class Test_PyFai_utils:

    def test_get_azimuthal_integrator(self):
        dect = Detector(pixel1=1e-4, pixel2=1e-4,max_shape=(20,20))
        ai = get_azimuthal_integrator(detector=dect, detector_distance=.001, shape=(20,20), center=(10.5,10.5))
        print(ai.get_correct_solid_angle_for_spline())
        import matplotlib.pyplot as plt
        plt.imshow(ai.solidAngleArray())
        plt.show()
        print()
        ai_mask = get_azimuthal_integrator(detector=dect, detector_distance=1, shape=(20, 20), center=(10.5, 10.5),
                                           mask=np.zeros((20,20)))
        aff = [[1,0,0],[0,1,0],[0,0,1]]
        ai_affine = get_azimuthal_integrator(detector=dect, detector_distance=1, shape=(20, 20), center=(10.5, 10.5),
                                             mask=np.zeros((20,20)), affine=aff)

    def test_get_displacements(self):
        aff = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        dis = _get_displacements((10.5,10.5),shape=(20,20),affine=aff)
        np.testing.assert_array_equal(dis, np.zeros(shape=(2,21,21)))

    def test_get_extent(self):
        dect = Detector(pixel1=1e-4, pixel2=1e-4)
        ai = AzimuthalIntegrator(detector=dect, dist=0.1)
        ai.setFit2D(directDist=1000, centerX=50.5, centerY=50.5)
        extent = _get_radial_extent(ai=ai,shape=(100,100), unit="2th_rad")
        max_rad = 50*np.sqrt(2)
        calc_extent = np.arctan(max_rad*1e-4/1)
        np.testing.assert_almost_equal(extent[1], calc_extent,)

    def test_get_curved_setup_2th(self):
        _get_curved_setup(wavelength=1,pyxem_unit="2th_deg", pixel_scale=[1,1])

    def test_get_curved_setup_nm(self):
        _get_curved_setup(wavelength=1,pyxem_unit="q_nm^-1", pixel_scale=[1,1],radial_range=[0,1])


