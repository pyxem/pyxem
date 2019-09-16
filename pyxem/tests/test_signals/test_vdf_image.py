# -*- coding: utf-8 -*-
# Copyright 2019 The pyXem developers
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

import numpy as np
import pytest

from hyperspy.signals import Signal2D

from pyxem.generators.vdf_generator import (VDFGenerator,
                                            VDFSegmentGenerator)
from pyxem.signals.electron_diffraction2d import ElectronDiffraction2D
from pyxem.signals.diffraction_vectors import DiffractionVectors
from pyxem.signals.vdf_image import VDFSegment


@pytest.fixture(params=[
    np.array([np.array([0, 0]), np.array([3, 0]), np.array([3, 5]),
              np.array([0, 5]), np.array([0, 3]), np.array([3, 3]),
              np.array([5, 3]), np.array([5, 5])])
])
def unique_vectors(request):
    uv = DiffractionVectors(request.param)
    uv.axes_manager.set_signal_dimension(0)
    return uv


@pytest.fixture
def signal_data():
    s = ElectronDiffraction2D(np.zeros((4, 5, 6, 6)))
    s.inav[:2, :2].data[..., 0, 0] = 2
    s.inav[:2, :2].data[..., 0, 3] = 2
    s.inav[:2, :2].data[..., 3, 5] = 2

    s.inav[2:, :3].data[..., 3, 3] = 2
    s.inav[2:, :3].data[..., 3, 0] = 2

    s.inav[2, :2].data[..., 0, 0] = 1
    s.inav[2, :2].data[..., 0, 3] = 1
    s.inav[2, :2].data[..., 3, 5] = 1
    s.inav[2, :2].data[..., 3, 3] = 1
    s.inav[2, :2].data[..., 3, 0] = 1

    s.inav[:2, 2:].data[..., 5, 5] = 3
    s.inav[:2, 2:].data[..., 5, 0] = 3
    s.inav[:2, 2:].data[..., 5, 3] = 3

    s.inav[3:, 2:].data[..., 5, 5] = 3
    s.inav[3:, 2:].data[..., 5, 0] = 3
    return s


@pytest.fixture
def vdf_segments(signal_data, unique_vectors):
    vdfgen = VDFGenerator(signal_data, unique_vectors)
    vdfs = vdfgen.get_vector_vdf_images(radius=1)
    vdfseggen = VDFSegmentGenerator(vdfs)
    return vdfseggen.get_vdf_segments()


class TestVDFSegment:

    @pytest.mark.parametrize('corr_threshold, vector_threshold,'
                             'segment_threshold',
                             [(0.1, 1, 1),
                              (0.9, 3, 2)])
    def test_correlate_segments(self, vdf_segments: VDFSegment,
                                corr_threshold, vector_threshold,
                                segment_threshold):
        corrsegs = vdf_segments.correlate_segments(
            corr_threshold, vector_threshold, segment_threshold)
        assert isinstance(corrsegs.segments, Signal2D)
        assert isinstance(corrsegs.vectors_of_segments, DiffractionVectors)
        assert isinstance(corrsegs.intensities, np.ndarray)

    def test_get_virtual_electron_diffraction(self, vdf_segments: VDFSegment,
                                              ):
        corrsegs = vdf_segments.correlate_segments(0.1, 1, 1)
        vs = corrsegs.get_virtual_electron_diffraction(
            calibration=1, sigma=1,
            shape=signal_data().axes_manager.signal_shape)
        assert isinstance(vs, ElectronDiffraction2D)
