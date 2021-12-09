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


from .dummy_data import (
    get_disk_shift_simple_test_signal,
    get_holz_simple_test_signal,
    get_holz_heterostructure_test_signal,
    get_single_ring_diffraction_signal,
    get_dead_pixel_signal,
    get_hot_pixel_signal,
    get_simple_dpc_signal,
    get_stripe_pattern_dpc_signal,
    get_square_dpc_signal,
    get_fem_signal,
    get_simple_fem_signal,
    get_generic_fem_signal,
    get_cbed_signal,
    get_simple_ellipse_signal_peak_array,
    get_nanobeam_electron_diffraction_signal,
)
from .make_diffraction_test_data import (
    EllipseRing,
    EllipseDisk,
    Circle,
    Disk,
    Ring,
    MakeTestData,
    generate_4d_data,
    DiffractionTestImage,
    DiffractionTestDataset,
)


__all__ = [
    "get_disk_shift_simple_test_signal",
    "get_holz_simple_test_signal",
    "get_holz_heterostructure_test_signal",
    "get_single_ring_diffraction_signal",
    "get_dead_pixel_signal",
    "get_hot_pixel_signal",
    "get_simple_dpc_signal",
    "get_stripe_pattern_dpc_signal",
    "get_square_dpc_signal",
    "get_fem_signal",
    "get_simple_fem_signal",
    "get_generic_fem_signal",
    "get_cbed_signal",
    "get_simple_ellipse_signal_peak_array",
    "get_nanobeam_electron_diffraction_signal",
    "EllipseRing",
    "EllipseDisk",
    "Circle",
    "Disk",
    "Ring",
    "MakeTestData",
    "generate_4d_data",
    "DiffractionTestImage",
    "DiffractionTestDataset",
]
