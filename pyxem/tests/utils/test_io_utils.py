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
import os

import hyperspy.api as hs

from pyxem.signals import (
    ElectronDiffraction1D,
    ElectronDiffraction2D,
    DiffractionVectors,
    DiffractionVectors2D,
    VirtualDarkFieldImage,
)


@pytest.mark.parametrize(
    "class_to_test,meta_string",
    [
        (ElectronDiffraction2D, "string1"),
        (DiffractionVectors, "string3"),
        (DiffractionVectors2D, "string3"),
        (ElectronDiffraction1D, "string5"),
        (VirtualDarkFieldImage, "string6"),
    ],
)
def test_load_function_core(class_to_test, meta_string):
    """
    Test the core; which is load a previously saved pyxem object.
    """
    to_save = class_to_test(np.zeros((2, 2, 2, 2)))
    to_save.metadata.Signal.tracker = meta_string
    if class_to_test is DiffractionVectors:
        to_save.axes_manager.set_signal_dimension(0)
    if class_to_test is DiffractionVectors2D:
        to_save.axes_manager.set_signal_dimension(2)
    to_save.save("tempfile_for_load_and_save.hspy", overwrite=True)
    from_save = hs.load("tempfile_for_load_and_save.hspy")
    assert isinstance(from_save, class_to_test)
    assert from_save.metadata.Signal.tracker == meta_string
    assert np.allclose(to_save.data, from_save.data)
    os.remove("tempfile_for_load_and_save.hspy")


@pytest.fixture()
def make_saved_dp(diffraction_pattern):
    """
    This makes use of conftest
    """
    diffraction_pattern.save("dp_temp")
    yield
    os.remove("dp_temp.hspy")


def test_load_ElectronDiffraction2D(diffraction_pattern, make_saved_dp):
    """
    This tests that our load function keeps .data, instance and metadata
    """
    dp = hs.load("dp_temp.hspy")
    assert np.allclose(dp.data, diffraction_pattern.data)
    assert isinstance(dp, ElectronDiffraction2D)
    assert (
        diffraction_pattern.metadata.Signal.found_from == dp.metadata.Signal.found_from
    )
