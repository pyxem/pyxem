# -*- coding: utf-8 -*-
# Copyright 2016-2021 The pyXem developers
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
import hyperspy.api as hs

from pyxem.generators import VirtualImageGenerator, VirtualDarkFieldGenerator
from pyxem.signals import (
    Diffraction2D,
    ElectronDiffraction2D,
    DiffractionVectors,
    VirtualDarkFieldImage,
)


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
    return VirtualDarkFieldGenerator(diffraction_pattern, diffraction_vectors)


class TestVirtualDarkFieldGenerator:
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

        vdfgen = VirtualDarkFieldGenerator(diffraction_pattern, dvm)
        assert isinstance(vdfgen.signal, ElectronDiffraction2D)
        assert isinstance(vdfgen.vectors, DiffractionVectors)

    @pytest.mark.parametrize("radius, normalize", [(4.0, False), (4.0, True)])
    def test_get_virtual_dark_field_images(self, vdf_generator, radius, normalize):
        vdfs = vdf_generator.get_virtual_dark_field_images(radius, normalize)
        assert isinstance(vdfs, VirtualDarkFieldImage)


class TestVirtualImageGenerator:
    def setup_method(self, method):
        # Navigation dimension of the diffraction patterns
        diffraction_pattern = Diffraction2D(np.arange(2000).reshape(4, 5, 10, 10))
        virtual_image_generator = VirtualImageGenerator(diffraction_pattern)
        self.virtual_image_generator = virtual_image_generator
        self.diffraction_pattern = diffraction_pattern

    @pytest.mark.parametrize(
        "k_min, k_max, k_steps, normalize", [(0.0, 4.0, 2, False), (0.0, 4.0, 2, True)]
    )
    def test_get_concentric_virtual_images(self, k_min, k_max, k_steps, normalize):
        vdfs = self.virtual_image_generator.get_concentric_virtual_images(
            k_min, k_max, k_steps, normalize
        )
        assert isinstance(vdfs, VirtualDarkFieldImage)

    def test_calibration_vdf_images(self):
        diffraction_pattern = self.diffraction_pattern
        virtual_image_generator = self.virtual_image_generator
        nav_axis = diffraction_pattern.axes_manager.navigation_axes[0]
        nav_axis.scale = 0.2
        nav_axis.offset = 10
        nav_axis.units = "nm"
        nav_axis.name = "position"
        for sig_axis in diffraction_pattern.axes_manager.signal_axes:
            sig_axis.scale = 0.2
            sig_axis.offset = -1.0
            sig_axis.units = "rad"
            sig_axis.name = "Scattering Angle"
        k_min, k_max, k_steps = 0.1, 0.6, 2
        vi = virtual_image_generator.get_concentric_virtual_images(
            k_min, k_max, k_steps
        )

        assert vi.data.shape == (2, 4, 5)

        vi_nav_axis = vi.axes_manager[0]
        assert vi_nav_axis.name == "Annular bins"
        assert vi_nav_axis.units == sig_axis.units
        assert vi_nav_axis.scale == (k_max - k_min) / k_steps
        assert vi_nav_axis.offset == k_min
        assert vi_nav_axis.size == k_steps

        vi_sig_axis = vi.axes_manager[1]
        for attr in ["scale", "offset", "units", "name"]:
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


def test_vi_generator_set_ROI_mesh(diffraction_pattern):
    vi_generator = VirtualImageGenerator(diffraction_pattern)
    diffraction_pattern.plot()
    vi_generator.set_ROI_mesh(1.0, 1.2, 1.0)
    assert len(vi_generator.roi_list) == 3
    assert isinstance(vi_generator.roi_list[0], hs.roi.CircleROI)


@pytest.mark.parametrize("nav_shape", [[2], [2, 4]])
def test_vi_generator_get_virtual_images_from_mesh(nav_shape):
    n = np.prod(nav_shape) * 32 ** 2
    s = Diffraction2D(np.arange(n).reshape((*nav_shape, 32, 32)))
    vi_generator = VirtualImageGenerator(s)

    with pytest.raises(ValueError):
        vi_generator.get_virtual_images_from_mesh()

    for axis in s.axes_manager.signal_axes:
        axis.scale = 0.1
        axis.offset = -1.6
        axis.units = "1/nm"

    vi_generator.set_ROI_mesh(0.5, 0.6, 1.4)
    vi = vi_generator.get_virtual_images_from_mesh()
    vi_nav_axis = vi.axes_manager.navigation_axes[0]
    assert vi_nav_axis.size == 21
    assert vi_nav_axis.name == "ROI index"
    assert vi_nav_axis.scale == 1.0
    assert len(vi_generator.roi_list) == 21

    vi_generator.set_ROI_mesh(0.5, 0.6, 1.0)
    assert len(vi_generator.roi_list) == 11

    vi = vi_generator.get_virtual_images_from_mesh(normalize=True)


def test_vi_generator_get_virtual_images_from_mesh_nav_dim3():
    s = Diffraction2D(np.arange(2 * 4 * 10 * 32 ** 2).reshape((2, 4, 10, 32, 32)))
    for axis in s.axes_manager.signal_axes:
        axis.scale = 0.1
        axis.offset = -1.6
        axis.units = "1/nm"
    vi_generator = VirtualImageGenerator(s)

    vi_generator.set_ROI_mesh(0.5, 0.6, 0.6)
    vi = vi_generator.get_virtual_images_from_mesh()

    assert vi.axes_manager.navigation_shape == (2, 5)

    vi = vi_generator.get_virtual_images_from_mesh(out_signal_axes=[1, 2])

    assert vi.axes_manager.navigation_shape == (10, 5)
