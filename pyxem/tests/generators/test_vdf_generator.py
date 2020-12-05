# -*- coding: utf-8 -*-
# Copyright 2016-2020 The pyXem developers
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
import numpy as np

from pyxem.generators.virtual_image_generator import VirtualImageGenerator

from pyxem.signals.electron_diffraction2d import ElectronDiffraction2D
from pyxem.signals.diffraction_vectors import DiffractionVectors
from pyxem.signals.vdf_image import VDFImage


@pytest.fixture(params=[np.array([[1, 1], [2, 2]])])
def diffraction_vectors(request):
    dvec = DiffractionVectors(request.param)
    dvec.axes_manager.set_signal_dimension(1)
    return dvec


@pytest.fixture
def vdf_generator(diffraction_pattern, diffraction_vectors):
    diffraction_pattern.data = np.where(
        diffraction_pattern.data == 0, 0.01, diffraction_pattern.data
    )  # avoid divide by zeroes
    return VirtualImageGenerator(diffraction_pattern, diffraction_vectors)


class TestVirtualImageGenerator:
    def test_vdf_generator_init_with_vectors(self, diffraction_pattern):
        dvm = DiffractionVectors(
            np.array(
                [
                    [np.array([[1, 1], [2, 2]]), np.array([[1, 1], [2, 2], [1, 2]])],
                    [np.array([[1, 1], [2, 2]]), np.array([[1, 1], [2, 2]])],
                ],
                dtype=object,
            )
        )
        dvm.axes_manager.set_signal_dimension(0)

        vdfgen = VirtualImageGenerator(diffraction_pattern, dvm)
        assert isinstance(vdfgen.signal, ElectronDiffraction2D)
        assert isinstance(vdfgen.vectors, DiffractionVectors)

    def test_vdf_generator_init_without_vectors(self, diffraction_pattern):

        vdfgen = VirtualImageGenerator(diffraction_pattern)
        assert isinstance(vdfgen.signal, ElectronDiffraction2D)
        assert isinstance(vdfgen.vectors, type(None))

    def test_vector_vdfs_without_vectors(self, diffraction_pattern):
        vdfgen = VirtualImageGenerator(diffraction_pattern)
        with pytest.raises(
            ValueError, match="DiffractionVectors not specified by user"
        ):
            vdfgen.get_vector_vdf_images(radius=2.0)

    @pytest.mark.parametrize("radius, normalize", [(4.0, False), (4.0, True)])
    def test_get_vector_vdf_images(
        self, vdf_generator: VirtualImageGenerator, radius, normalize
    ):
        vdfs = vdf_generator.get_vector_vdf_images(radius, normalize)
        assert isinstance(vdfs, VDFImage)

    @pytest.mark.parametrize(
        "k_min, k_max, k_steps, normalize", [(0.0, 4.0, 2, False), (0.0, 4.0, 2, True)]
    )
    def test_get_concentric_virtual_images(
        self, vdf_generator: VirtualImageGenerator, k_min, k_max, k_steps, normalize
    ):
        vdfs = vdf_generator.get_concentric_virtual_images(
            k_min, k_max, k_steps, normalize)
        assert isinstance(vdfs, VDFImage)

    def test_calibration_vdf_images(self):
        dp = ElectronDiffraction2D(np.arange(500).reshape(5, 10, 10))
        nav_axis = dp.axes_manager.navigation_axes[0]
        nav_axis.scale = 0.2
        nav_axis.offset = 10
        nav_axis.units = 'nm'
        nav_axis.name = 'position'
        for sig_axis in dp.axes_manager.signal_axes:
            sig_axis.scale = 0.2
            sig_axis.offset = -1.0
            sig_axis.units = 'rad'
            sig_axis.name = 'Scattering Angle'
        virtual_image_generator = VirtualImageGenerator(dp)
        k_min, k_max, k_steps = 0.1, 0.6, 2
        vi = virtual_image_generator.get_concentric_virtual_images(
            k_min, k_max, k_steps)

        assert vi.data.shape == (2, 5)

        vi_nav_axis = vi.axes_manager[0]
        assert vi_nav_axis.name == 'Annular bins'
        assert vi_nav_axis.units == sig_axis.units
        assert vi_nav_axis.scale == (k_max - k_min) / k_steps
        assert vi_nav_axis.offset == k_min
        assert vi_nav_axis.size == k_steps

        vi_sig_axis = vi.axes_manager[1]
        for attr in ['scale', 'offset', 'units', 'name']:
            assert getattr(vi_sig_axis, attr) == getattr(nav_axis, attr)



def test_vdf_generator_from_map(diffraction_pattern):
    dvm = DiffractionVectors(
        np.array(
            [
                [np.array([[1, 1], [2, 2]]), np.array([[1, 1], [2, 2], [1, 2]])],
                [np.array([[1, 1], [2, 2]]), np.array([[1, 1], [2, 2]])],
            ],
            dtype=object,
        )
    )
    dvm.axes_manager.set_signal_dimension(0)

    vdfgen = VirtualImageGenerator(diffraction_pattern, dvm)
    assert isinstance(vdfgen, VirtualImageGenerator)
