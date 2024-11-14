# -*- coding: utf-8 -*-
# Copyright 2016-2024 The pyXem developers
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
import hyperspy.models.model1d
import hyperspy.api as hs
import pytest

import numpy as np
import dask.array as da
from scipy.signal.windows import gaussian
from diffsims.simulations import Simulation1D
from orix.crystal_map import Phase
from diffpy.structure import Atom, Lattice, Structure

from pyxem.signals import ElectronDiffraction1D, LazyElectronDiffraction1D


class TestSimpleHyperspy:
    # Tests functions that assign to hyperspy metadata

    def test_set_experimental_parameters(self, electron_diffraction1d):
        electron_diffraction1d.set_experimental_parameters(
            accelerating_voltage=3,
            camera_length=3,
            scan_rotation=1,
            convergence_angle=1,
            rocking_angle=1,
            rocking_frequency=1,
            exposure_time=1,
        )
        assert isinstance(electron_diffraction1d, ElectronDiffraction1D)

    def test_set_scan_calibration(self, electron_diffraction1d):
        electron_diffraction1d.set_scan_calibration(19)
        assert isinstance(electron_diffraction1d, ElectronDiffraction1D)

    @pytest.mark.parametrize(
        "calibration",
        [
            1,
            0.017,
            0.5,
        ],
    )
    def test_set_diffraction_calibration(self, electron_diffraction1d, calibration):
        electron_diffraction1d.set_diffraction_calibration(calibration)
        dx = electron_diffraction1d.axes_manager.signal_axes[0]
        assert dx.scale == calibration


class TestComputeAndAsLazyElectron1D:
    def test_2d_data_compute(self):
        dask_array = da.random.random((100, 150), chunks=(50, 50))
        s = LazyElectronDiffraction1D(dask_array)
        scale0, scale1, metadata_string = 0.5, 1.5, "test"
        s.axes_manager[0].scale = scale0
        s.axes_manager[1].scale = scale1
        s.metadata.Test = metadata_string
        s.compute()
        assert isinstance(s, ElectronDiffraction1D)
        assert not hasattr(s.data, "compute")
        assert s.axes_manager[0].scale == scale0
        assert s.axes_manager[1].scale == scale1
        assert s.metadata.Test == metadata_string
        assert dask_array.shape == s.data.shape

    def test_3d_data_compute(self):
        dask_array = da.random.random((4, 10, 15), chunks=(1, 10, 15))
        s = LazyElectronDiffraction1D(dask_array)
        s.compute()
        assert isinstance(s, ElectronDiffraction1D)
        assert dask_array.shape == s.data.shape

    def test_2d_data_as_lazy(self):
        data = np.random.random((100, 150))
        s = ElectronDiffraction1D(data)
        scale0, scale1, metadata_string = 0.5, 1.5, "test"
        s.axes_manager[0].scale = scale0
        s.axes_manager[1].scale = scale1
        s.metadata.Test = metadata_string
        s_lazy = s.as_lazy()
        assert isinstance(s_lazy, LazyElectronDiffraction1D)
        assert hasattr(s_lazy.data, "compute")
        assert s_lazy.axes_manager[0].scale == scale0
        assert s_lazy.axes_manager[1].scale == scale1
        assert s_lazy.metadata.Test == metadata_string
        assert data.shape == s_lazy.data.shape

    def test_3d_data_as_lazy(self):
        data = np.random.random((4, 10, 15))
        s = ElectronDiffraction1D(data)
        s_lazy = s.as_lazy()
        assert isinstance(s_lazy, LazyElectronDiffraction1D)
        assert data.shape == s_lazy.data.shape


class TestDecomposition:
    def test_decomposition_class_assignment(self, electron_diffraction1d):
        electron_diffraction1d.decomposition()
        assert isinstance(electron_diffraction1d, ElectronDiffraction1D)


class TestModelFitting:
    @pytest.fixture()
    def simple_1d_plus_model(self):
        peaks = [20, 40, 60, 80]
        x = np.zeros(100)
        for i, p in enumerate(peaks):
            g = gaussian(100, 3)
            g = np.roll(g, p, axis=0)
            if i == 0:
                g = g * 2
            x += g
        s = ElectronDiffraction1D(x)
        p = Phase(
            name="al",
            space_group=225,
            structure=Structure(
                atoms=[Atom("al", [0, 0, 0])],
                lattice=Lattice(0.405, 0.405, 0.405, 90, 90, 90),
            ),
        )
        shifted_peaks = (np.array([20, 40, 60, 80]) + 50) % 101
        shifted_peaks = np.sort(shifted_peaks)
        sim = Simulation1D(
            phase=p,
            reciprocal_spacing=shifted_peaks,
            intensities=(1, 1, 2, 1),
            hkl=[[1, 1, 1], [2, 0, 0], [3, 1, 1], [4, 0, 0]],
            reciprocal_radius=100,
            wavelength=1e-9,
        )
        model = s.model_simulation1d(sim)
        return s, model, shifted_peaks, sim

    def test_create_model(self, simple_1d_plus_model):
        s, model, peaks, _ = simple_1d_plus_model
        assert isinstance(model, hyperspy.models.model1d.Model1D)
        assert len(model) == 5
        centers = np.sort(
            [
                m.centre.value
                for m in model
                if isinstance(m, hs.model.components1D.Gaussian)
            ]
        )
        np.testing.assert_array_almost_equal(centers, peaks, decimal=1)

    def test_model2theta_scale(self, simple_1d_plus_model):
        s, model, peaks, sim = simple_1d_plus_model
        scale = s.model2theta_scale(sim, model, 300)
        np.testing.assert_almost_equal(scale, 0.012768541396519572, decimal=6)

    def test_camera_length(self, simple_1d_plus_model):
        s, model, peaks, sim = simple_1d_plus_model
        scale = s.model2camera_length(sim, model, 300, 0.051077)
        np.testing.assert_almost_equal(scale, 4, decimal=4)
