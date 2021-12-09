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

from pyxem.generators import VirtualDarkFieldGenerator
from pyxem.signals import ElectronDiffraction2D, DiffractionVectors, VDFSegment


@pytest.fixture(
    params=[
        np.array(
            [
                np.array([0, 0]),
                np.array([3, 0]),
                np.array([3, 5]),
                np.array([0, 5]),
                np.array([0, 3]),
                np.array([3, 3]),
                np.array([5, 3]),
                np.array([5, 5]),
            ]
        )
    ]
)
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
def vdf_generator_seg(signal_data, unique_vectors):
    return VirtualDarkFieldGenerator(signal_data, unique_vectors)


@pytest.fixture
def vdf_vector_images_seg(vdf_generator_seg):
    return vdf_generator_seg.get_virtual_dark_field_images(radius=1)


class TestVDFImage:
    @pytest.mark.parametrize(
        "min_distance, min_size, max_size,"
        "max_number_of_grains, marker_radius,"
        "threshold, exclude_border",
        [(1, 1, 20, 5, 1, False, 0), (2, 3, 200, 10, 2, True, 1)],
    )
    def test_get_vdf_segments(
        self,
        vdf_vector_images_seg,
        min_distance,
        min_size,
        max_size,
        max_number_of_grains,
        marker_radius,
        threshold,
        exclude_border,
    ):
        segs = vdf_vector_images_seg.get_vdf_segments(
            min_distance,
            min_size,
            max_size,
            max_number_of_grains,
            marker_radius,
            threshold,
            exclude_border,
        )
        assert isinstance(segs, VDFSegment)
        assert isinstance(segs.segments, Signal2D)
        assert isinstance(segs.vectors_of_segments, DiffractionVectors)
