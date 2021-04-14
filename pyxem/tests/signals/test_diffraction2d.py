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
import dask.array as da
import hyperspy.api as hs
from matplotlib import pyplot as plt
from numpy.random import default_rng
from skimage.draw import circle_perimeter_aa

from pyxem.signals import (
    Diffraction1D,
    Diffraction2D,
    LazyDiffraction2D,
    PolarDiffraction2D,
)


class TestComputeAndAsLazy2D:
    def test_2d_data_compute(self):
        dask_array = da.random.random((100, 150), chunks=(50, 50))
        s = LazyDiffraction2D(dask_array)
        scale0, scale1, metadata_string = 0.5, 1.5, "test"
        s.axes_manager[0].scale = scale0
        s.axes_manager[1].scale = scale1
        s.metadata.Test = metadata_string
        s.compute()
        assert isinstance(s, Diffraction2D)
        assert not hasattr(s.data, "compute")
        assert s.axes_manager[0].scale == scale0
        assert s.axes_manager[1].scale == scale1
        assert s.metadata.Test == metadata_string
        assert dask_array.shape == s.data.shape

    def test_4d_data_compute(self):
        dask_array = da.random.random((4, 4, 10, 15), chunks=(1, 1, 10, 15))
        s = LazyDiffraction2D(dask_array)
        s.compute()
        assert isinstance(s, Diffraction2D)
        assert dask_array.shape == s.data.shape

    def test_2d_data_as_lazy(self):
        data = np.random.random((100, 150))
        s = Diffraction2D(data)
        scale0, scale1, metadata_string = 0.5, 1.5, "test"
        s.axes_manager[0].scale = scale0
        s.axes_manager[1].scale = scale1
        s.metadata.Test = metadata_string
        s_lazy = s.as_lazy()
        assert isinstance(s_lazy, LazyDiffraction2D)
        assert hasattr(s_lazy.data, "compute")
        assert s_lazy.axes_manager[0].scale == scale0
        assert s_lazy.axes_manager[1].scale == scale1
        assert s_lazy.metadata.Test == metadata_string
        assert data.shape == s_lazy.data.shape

    def test_4d_data_as_lazy(self):
        data = np.random.random((4, 10, 15))
        s = Diffraction2D(data)
        s_lazy = s.as_lazy()
        assert isinstance(s_lazy, LazyDiffraction2D)
        assert data.shape == s_lazy.data.shape


class TestDecomposition:
    def test_decomposition_is_performed(self, diffraction_pattern):
        s = Diffraction2D(diffraction_pattern)
        s.decomposition()
        assert s.learning_results is not None

    def test_decomposition_class_assignment(self, diffraction_pattern):
        s = Diffraction2D(diffraction_pattern)
        s.decomposition()
        assert isinstance(s, Diffraction2D)


class TestAzimuthalIntegral1d:
    @pytest.fixture
    def ones(self):
        ones_diff = Diffraction2D(data=np.ones(shape=(10, 10)))
        ones_diff.axes_manager.signal_axes[0].scale = 0.1
        ones_diff.axes_manager.signal_axes[1].scale = 0.1
        ones_diff.axes_manager.signal_axes[0].name = "kx"
        ones_diff.axes_manager.signal_axes[1].name = "ky"
        ones_diff.unit = "2th_deg"
        return ones_diff

    def test_unit(self, ones):
        dif = Diffraction2D(data=[[1, 1], [1, 1]])
        assert dif.unit is None
        dif.unit = "!23"
        assert dif.unit is None

    def test_unit_set(self, ones):
        assert ones.unit == "2th_deg"

    @pytest.mark.parametrize(
        "unit", ["q_nm^-1", "q_A^-1", "k_nm^-1", "k_A^-1", "2th_deg", "2th_rad"]
    )
    def test_1d_azimuthal_integral_2th_units(self, ones, unit):
        ones.unit = unit
        ones.set_ai(wavelength=1e-9)
        az = ones.get_azimuthal_integral1d(
            npt=10, wavelength=1e-9, correctSolidAngle=False
        )
        np.testing.assert_array_equal(az.data[0:8], np.ones(8))

    def test_1d_azimuthal_integral_inplace(self, ones):
        ones.set_ai()
        az = ones.get_azimuthal_integral1d(
            npt=10,
            correctSolidAngle=False,
            inplace=True,
        )
        assert isinstance(ones, Diffraction1D)
        np.testing.assert_array_equal(ones.data[0:8], np.ones((8)))
        assert az is None

    def test_1d_azimuthal_integral_slicing(self, ones):
        ones.unit = "2th_rad"
        ones.set_ai(center=(5.5, 5.5))
        az = ones.get_azimuthal_integral1d(
            npt=10,
            method="BBox",
            correctSolidAngle=False,
            radial_range=[0.0, 1.0],
        )
        np.testing.assert_array_equal(az.data[0:7], np.ones(7))

    @pytest.mark.parametrize(
        "unit", ["q_nm^-1", "q_A^-1", "k_nm^-1", "k_A^-1", "2th_deg", "2th_rad"]
    )
    def test_1d_axes_continuity(self, ones, unit):
        ones.unit = unit
        ones.set_ai(center=(5.5, 5.5), wavelength=1e-9)
        az1 = ones.get_azimuthal_integral1d(
            npt=10,
            radial_range=[0.0, 1.0],
            method="splitpixel",
        )
        assert np.allclose(az1.axes_manager.signal_axes[0].scale, 0.1)

    @pytest.mark.parametrize("radial_range", [None, [0.0, 1.0]])
    @pytest.mark.parametrize("azimuth_range", [None, [-np.pi, np.pi]])
    @pytest.mark.parametrize("center", [None, [9, 9]])
    @pytest.mark.parametrize("affine", [None, [[1, 0, 0], [0, 1, 0], [0, 0, 1]]])
    def test_1d_integration(
        self,
        ones,
        radial_range,
        azimuth_range,
        center,
        affine,
    ):
        ones.set_ai(center=center, affine=affine, radial_range=radial_range)
        az = ones.get_azimuthal_integral1d(
            npt=10,
            method="BBox",
            radial_range=radial_range,
            azimuth_range=azimuth_range,
            correctSolidAngle=False,
            inplace=False,
        )
        assert isinstance(az, Diffraction1D)

    def test_1d_azimuthal_integral_mask(self, ones):
        from hyperspy.signals import BaseSignal

        aff = [[1, 1, 0], [0, 1, 0], [0, 0, 1]]
        center = [1, 1]
        mask = np.zeros((10, 10))
        mask_bs = BaseSignal(data=mask)
        ones.set_ai(center=center, affine=aff)
        ones.get_azimuthal_integral1d(
            npt=10,
            method="BBox",
            wavelength=1e-9,
            correctSolidAngle=False,
            mask=mask_bs,
        )

    def test_1d_azimuthal_integral_sum(self, ones):
        ones.set_ai()
        integration = ones.get_azimuthal_integral1d(
            npt=5, radial_range=[0, 0.5], sum=True
        )
        # 5^2*pi = 78.5
        np.testing.assert_almost_equal(integration.data.sum(), 78.5, decimal=0)

    @pytest.mark.parametrize(
        "shape", [(20, 16), (3, 20, 16), (4, 3, 20, 16), (6, 4, 3, 20, 16)]
    )
    def test_lazy_input_lazy_output_different_shapes(self, shape):
        chunks = [5] * len(shape)
        s = LazyDiffraction2D(da.ones(shape, chunks=chunks))
        s.unit = "2th_deg"
        s.set_ai()
        npt = 10
        s_a = s.get_azimuthal_integral1d(npt=npt)
        output_signal_shape = s.axes_manager.shape[:-2] + (npt,)
        output_data_shape = shape[:-2] + (npt,)
        assert s_a.axes_manager.shape == output_signal_shape
        assert s_a.data.shape == output_data_shape
        s_a.compute()
        assert s_a.axes_manager.shape == output_signal_shape
        assert s_a.data.shape == output_data_shape

    @pytest.mark.parametrize(
        "shape", [(20, 16), (3, 20, 16), (4, 3, 20, 16), (6, 4, 3, 20, 16)]
    )
    def test_non_lazy_input_lazy_output(self, shape):
        s = Diffraction2D(np.ones(shape))
        s.unit = "2th_deg"
        s.set_ai()
        npt = 10
        s_a = s.get_azimuthal_integral1d(npt=npt, lazy_result=True)
        output_signal_shape = s.axes_manager.shape[:-2] + (npt,)
        output_data_shape = shape[:-2] + (npt,)
        assert s_a.axes_manager.shape == output_signal_shape
        assert s_a.data.shape == output_data_shape
        s_a.compute()
        assert s_a.axes_manager.shape == output_signal_shape
        assert s_a.data.shape == output_data_shape

    @pytest.mark.parametrize(
        "shape", [(20, 16), (3, 20, 16), (4, 3, 20, 16), (6, 4, 3, 20, 16)]
    )
    def test_lazy_input_non_lazy_output(self, shape):
        chunks = [5] * len(shape)
        s = LazyDiffraction2D(da.ones(shape, chunks=chunks))
        s.unit = "2th_deg"
        s.set_ai()
        npt = 10
        s_a = s.get_azimuthal_integral1d(npt=npt, lazy_result=False)
        output_signal_shape = s.axes_manager.shape[:-2] + (npt,)
        output_data_shape = shape[:-2] + (npt,)
        assert s_a.axes_manager.shape == output_signal_shape
        assert s_a.data.shape == output_data_shape

    @pytest.mark.parametrize(
        "shape", [(20, 16), (3, 20, 16), (4, 3, 20, 16), (6, 4, 3, 20, 16)]
    )
    def test_lazy_input_lazy_result_inplace(self, shape):
        chunks = [5] * len(shape)
        s = LazyDiffraction2D(da.ones(shape, chunks=chunks))
        s.unit = "2th_deg"
        s.set_ai()
        npt = 10
        output_signal_shape = s.axes_manager.shape[:-2] + (npt,)
        output_data_shape = shape[:-2] + (npt,)
        s.get_azimuthal_integral1d(npt=npt, inplace=True, lazy_result=True)
        assert s.axes_manager.shape == output_signal_shape
        assert s.data.shape == output_data_shape
        s.compute()
        assert s.axes_manager.shape == output_signal_shape
        assert s.data.shape == output_data_shape

    @pytest.mark.parametrize(
        "shape", [(20, 16), (3, 20, 16), (4, 3, 20, 16), (6, 4, 3, 20, 16)]
    )
    def test_lazy_input_non_lazy_result_inplace(self, shape):
        chunks = [5] * len(shape)
        s = LazyDiffraction2D(da.ones(shape, chunks=chunks))
        s.unit = "2th_deg"
        s.set_ai()
        npt = 10
        output_signal_shape = s.axes_manager.shape[:-2] + (npt,)
        output_data_shape = shape[:-2] + (npt,)
        s.get_azimuthal_integral1d(npt=npt, inplace=True, lazy_result=False)
        assert s.axes_manager.shape == output_signal_shape
        assert s.data.shape == output_data_shape

    @pytest.mark.parametrize(
        "shape", [(20, 16), (3, 20, 16), (4, 3, 20, 16), (6, 4, 3, 20, 16)]
    )
    def test_non_lazy_input_lazy_result_inplace(self, shape):
        s = Diffraction2D(np.ones(shape))
        s.unit = "2th_deg"
        s.set_ai()
        npt = 10
        output_signal_shape = s.axes_manager.shape[:-2] + (npt,)
        output_data_shape = shape[:-2] + (npt,)
        s.get_azimuthal_integral1d(npt=npt, inplace=True, lazy_result=True)
        assert s.axes_manager.shape == output_signal_shape
        assert s.data.shape == output_data_shape
        s.compute()
        assert s.axes_manager.shape == output_signal_shape
        assert s.data.shape == output_data_shape


class TestVariance:
    @pytest.fixture
    def ones(self):
        ones_diff = Diffraction2D(data=np.ones(shape=(10, 10, 10, 10)))
        ones_diff.axes_manager.signal_axes[0].scale = 0.1
        ones_diff.axes_manager.signal_axes[1].scale = 0.1
        ones_diff.axes_manager.signal_axes[0].name = "kx"
        ones_diff.axes_manager.signal_axes[1].name = "ky"
        ones_diff.unit = "2th_deg"
        ones_diff.set_ai()
        return ones_diff

    @pytest.fixture
    def ones_zeros(self):
        data = np.ones(shape=(10, 10, 10, 10))
        data[0:10:2, :, :, :] = 2
        ones_diff = Diffraction2D(data=data)
        ones_diff.axes_manager.signal_axes[0].scale = 0.1
        ones_diff.axes_manager.signal_axes[1].scale = 0.1
        ones_diff.axes_manager.signal_axes[0].name = "kx"
        ones_diff.axes_manager.signal_axes[1].name = "ky"
        ones_diff.unit = "2th_deg"
        ones_diff.set_ai()
        return ones_diff

    @pytest.fixture
    def bulls_eye_noisy(self):
        x, y = np.mgrid[-25:25, -25:25]
        data = (x ** 2 + y ** 2) ** 0.5
        data = np.tile(data, (5, 5, 1, 1))
        # Electron is equal to 1 count in image
        rng = default_rng(seed=1)
        data = rng.poisson(lam=data)
        ones_diff = Diffraction2D(data=data)
        ones_diff.axes_manager.signal_axes[0].scale = 0.1
        ones_diff.axes_manager.signal_axes[1].scale = 0.1
        ones_diff.axes_manager.signal_axes[0].name = "kx"
        ones_diff.axes_manager.signal_axes[1].name = "ky"
        ones_diff.unit = "2th_deg"
        ones_diff.set_ai()
        return ones_diff

    def test_FEM_Omega(self, ones, ones_zeros):
        ones_variance = ones.get_variance(npt=5, method="Omega")
        # assert ones_variance.axes_manager[0].units == "2th_deg"
        np.testing.assert_array_almost_equal(ones_variance.data, np.zeros(5), decimal=3)
        ones_zeros_variance = ones_zeros.get_variance(5, method="Omega")
        np.testing.assert_array_almost_equal(
            ones_zeros_variance.data, np.ones(5) * 0.1111, decimal=3
        )

    def test_FEM_Omega_poisson_noise(self, bulls_eye_noisy):
        bulls_eye_variance = bulls_eye_noisy.get_variance(25, method="Omega", dqe=1)
        # We exclude small radii
        np.testing.assert_array_almost_equal(
            bulls_eye_variance.data[5:], np.zeros(20), decimal=2
        )
        # Testing for non dqe=1
        bulls_eye_variance = (bulls_eye_noisy * 10).get_variance(
            25, method="Omega", dqe=10
        )
        # This fails at small radii and might still fail because it is random...
        np.testing.assert_array_almost_equal(
            bulls_eye_variance.data[6:], np.zeros(19), decimal=2
        )

    def test_FEM_r(self, ones, ones_zeros, bulls_eye_noisy):
        ones_variance = ones.get_variance(npt=5, method="r")
        # assert ones_variance.axes_manager[0].units == "2th_deg"
        np.testing.assert_array_almost_equal(ones_variance.data, np.zeros(5), decimal=3)
        ones_zeros_variance = ones_zeros.get_variance(5, method="r")
        np.testing.assert_array_almost_equal(
            ones_zeros_variance.data, np.zeros(5), decimal=3
        )
        bulls_eye_variance = bulls_eye_noisy.get_variance(25, method="r", dqe=1)
        # We exclude small radii
        np.testing.assert_array_almost_equal(
            bulls_eye_variance.data[5:], np.zeros(20), decimal=2
        )
        # Testing for non dqe=1
        bulls_eye_variance = (bulls_eye_noisy * 10).get_variance(25, method="r", dqe=10)
        # We exclude small radii
        np.testing.assert_array_almost_equal(
            bulls_eye_variance.data[6:], np.zeros(19), decimal=2
        )

    def test_FEM_r_spatial_kwarg(self, ones, ones_zeros, bulls_eye_noisy):
        v, fv = ones.get_variance(npt=5, method="r", spatial=True)

    @pytest.mark.parametrize("dqe_choice", [None, 0.3])
    def test_FEM_VImage(self, ones, dqe_choice):
        v = ones.get_variance(npt=5, method="VImage", dqe=dqe_choice)
        if dqe_choice is None:
            np.testing.assert_array_almost_equal(v.data, np.zeros(5), decimal=3)

    def test_FEM_re(self, ones, ones_zeros, bulls_eye_noisy):
        ones_variance = ones.get_variance(npt=5, method="re")
        np.testing.assert_array_almost_equal(ones_variance.data, np.zeros(5), decimal=3)
        ones_zeros_variance = ones_zeros.get_variance(5, method="re")
        np.testing.assert_array_almost_equal(
            ones_zeros_variance.data, np.ones(5) * 0.1111, decimal=3
        )
        bulls_eye_variance = bulls_eye_noisy.get_variance(25, method="re", dqe=1)
        # This fails at small radii and might still fail because it is random...
        np.testing.assert_array_almost_equal(
            bulls_eye_variance.data[5:], np.zeros(20), decimal=2
        )
        # Testing for non dqe=1
        bulls_eye_variance = (bulls_eye_noisy * 10).get_variance(
            25, method="re", dqe=10
        )
        # This fails at small radii and might still fail because it is random...
        np.testing.assert_array_almost_equal(
            bulls_eye_variance.data[6:], np.zeros(19), decimal=1
        )

    def test_not_existing_method(self, ones):
        with pytest.raises(ValueError):
            ones.get_variance(npt=5, method="magic")


class TestAzimuthalIntegral2d:
    @pytest.fixture
    def ones(self):
        ones_diff = Diffraction2D(data=np.ones(shape=(10, 10)))
        ones_diff.axes_manager.signal_axes[0].scale = 0.1
        ones_diff.axes_manager.signal_axes[1].scale = 0.1
        ones_diff.axes_manager.signal_axes[0].name = "kx"
        ones_diff.axes_manager.signal_axes[1].name = "ky"
        ones_diff.unit = "2th_deg"
        return ones_diff

    @pytest.fixture
    def ring(self):
        ring_pattern = Diffraction2D(data=np.ones(shape=(100, 100)))
        rr, cc, val = circle_perimeter_aa(r=50, c=50, radius=30, shape=(100, 100))
        ring_pattern.data[rr, cc] = val * 100
        ring_pattern.axes_manager.signal_axes[0].scale = 0.1
        ring_pattern.axes_manager.signal_axes[1].scale = 0.1
        ring_pattern.axes_manager.signal_axes[0].name = "kx"
        ring_pattern.axes_manager.signal_axes[1].name = "ky"
        ring_pattern.unit = "k_nm^-1"
        return ring_pattern

    def test_2d_azimuthal_integral(self, ones):
        ones.set_ai()
        az = ones.get_azimuthal_integral2d(
            npt=10, npt_azim=10, method="BBox", correctSolidAngle=False
        )
        np.testing.assert_array_equal(az.data[0:8, :], np.ones((8, 10)))

    def test_2d_azimuthal_integral_scale(self,ring):
        ring.set_ai(wavelength=2.5e-12)
        az = ring.get_azimuthal_integral2d(npt=500)
        peak = np.argmax(az.sum(axis=0)).data * az.axes_manager[1].scale
        np.testing.assert_almost_equal(peak[0], 3,decimal=1)
        ring.unit="k_A^-1"
        ring.set_ai(wavelength=2.5e-12)
        az = ring.get_azimuthal_integral2d(npt=500)
        peak = np.argmax(az.sum(axis=0)).data * az.axes_manager[1].scale
        np.testing.assert_almost_equal(peak[0], 3,decimal=1)

    def test_2d_azimuthal_integral_fast_slicing(self, ones):
        ones.set_ai(center=(5.5, 5.5))
        az1 = ones.get_azimuthal_integral2d(
            npt=10,
            npt_azim=10,
            radial_range=[0.0, 1.0],
            method="splitpixel",
            correctSolidAngle=False,
        )
        np.testing.assert_array_equal(az1.data[8:, :], np.zeros(shape=(2, 10)))

    @pytest.mark.parametrize(
        "unit", ["q_nm^-1", "q_A^-1", "k_nm^-1", "k_A^-1", "2th_deg", "2th_rad"]
    )
    def test_2d_axes_continuity(self, ones, unit):
        ones.unit = unit
        ones.set_ai(wavelength=1e-9, center=(5.5, 5.5))
        az1 = ones.get_azimuthal_integral2d(
            npt=10,
            npt_azim=20,
            radial_range=[0.0, 1.0],
            method="splitpixel",
        )
        assert np.allclose(az1.axes_manager.signal_axes[1].scale, 0.1)

    def test_2d_azimuthal_integral_inplace(self, ones):
        ones.set_ai()
        az = ones.get_azimuthal_integral2d(
            npt=10,
            npt_azim=10,
            correctSolidAngle=False,
            inplace=True,
            method="BBox",
        )
        assert isinstance(ones, PolarDiffraction2D)
        np.testing.assert_array_equal(ones.data[0:8, :], np.ones((8, 10)))
        assert az is None

    @pytest.mark.parametrize("radial_range", [None, [0, 1.0]])
    @pytest.mark.parametrize("azimuth_range", [None, [-np.pi, 0]])
    @pytest.mark.parametrize("correctSolidAngle", [True, False])
    @pytest.mark.parametrize("center", [None, [7, 7]])
    @pytest.mark.parametrize("affine", [None, [[1, 0, 0], [0, 1, 0], [0, 0, 1]]])
    def test_2d_azimuthal_integral_params(
        self, ones, radial_range, azimuth_range, correctSolidAngle, center, affine
    ):
        ones.set_ai(
            center=center, affine=affine, radial_range=radial_range, wavelength=1e-9
        )
        az = ones.get_azimuthal_integral2d(
            npt=10,
            npt_azim=10,
            radial_range=radial_range,
            azimuth_range=azimuth_range,
            correctSolidAngle=correctSolidAngle,
            method="BBox",
        )
        assert isinstance(az, PolarDiffraction2D)

    def test_2d_azimuthal_integral_mask_iterate(self, ones):
        from hyperspy.signals import BaseSignal

        aff = [[1, 1, 0], [0, 1, 0], [0, 0, 1]]
        center = [1, 1]
        ones.set_ai(center=center, affine=aff, wavelength=1e-9)
        mask = np.zeros((10, 10))
        mask_bs = BaseSignal(data=mask)
        ones.get_azimuthal_integral2d(
            npt=10,
            npt_azim=10,
            method="BBox",
            correctSolidAngle=False,
            mask=mask_bs,
        )

    def test_2d_azimuthal_integral_sum(self, ones):
        ones.set_ai()
        integration = ones.get_azimuthal_integral2d(
            npt=10, npt_azim=15, radial_range=[0, 0.5], sum=True
        )
        #  mostly correct except for at the very center where things get weird...
        mask = np.ones((10, 10), dtype=bool)
        mask[0:5] = False
        integration2 = ones.get_azimuthal_integral2d(
            npt=10, npt_azim=15, radial_range=[0, 0.5], sum=True, mask=mask
        )


class TestPyFAIIntegration:
    @pytest.fixture
    def ones(self):
        ones_diff = Diffraction2D(data=np.ones(shape=(10, 10)))
        ones_diff.axes_manager.signal_axes[0].scale = 0.1
        ones_diff.axes_manager.signal_axes[1].scale = 0.1
        ones_diff.axes_manager.signal_axes[0].name = "kx"
        ones_diff.axes_manager.signal_axes[1].name = "ky"
        ones_diff.unit = "q_nm^-1"
        return ones_diff

    def test_integrate_radial(self, ones):
        ones.set_ai(center=(5.5, 5.5), wavelength=1e-9)
        integration = ones.get_radial_integral(
            npt=10,
            npt_rad=100,
            method="BBox",
            correctSolidAngle=False,
        )
        np.testing.assert_array_equal(integration, np.ones(10))
        integration = ones.get_radial_integral(
            npt=10,
            npt_rad=100,
            method="BBox",
            correctSolidAngle=False,
            sum=True,
        )
        integration = ones.get_radial_integral(
            npt=10, npt_rad=100, method="BBox", correctSolidAngle=False, inplace=True
        )
        np.testing.assert_array_equal(ones, np.ones(10))
        assert integration is None

    def test_integrate_med_filter(self, ones):
        ones.set_ai(center=(5.5, 5.5), wavelength=1e-9)
        integration = ones.get_medfilt1d(
            npt_rad=10, npt_azim=100, method="BBox", correctSolidAngle=False
        )
        np.testing.assert_array_equal(integration, np.ones(10))
        integration = ones.get_medfilt1d(
            npt_rad=10,
            npt_azim=100,
            method="BBox",
            correctSolidAngle=False,
            inplace=True,
        )
        np.testing.assert_array_equal(ones, np.ones(10))
        assert integration is None

    def test_integrate_sigma_clip(self, ones):
        ones.set_ai(center=(5.5, 5.5), wavelength=1e-9)
        integration = ones.sigma_clip(
            npt_rad=10, npt_azim=100, method="BBox", correctSolidAngle=False
        )
        np.testing.assert_array_equal(integration, np.ones(10))
        integration = ones.sigma_clip(
            npt_rad=10,
            npt_azim=100,
            method="BBox",
            correctSolidAngle=False,
            inplace=True,
        )
        np.testing.assert_array_equal(ones, np.ones(10))
        assert integration is None


class TestVirtualImaging:
    # Tests that virtual imaging runs without failure

    def test_plot_integrated_intensity(self, diffraction_pattern):
        roi = hs.roi.CircleROI(3, 3, 5)
        plt.ion()  # to make plotting non-blocking
        diffraction_pattern.plot_integrated_intensity(roi)
        plt.close("all")

    def test_get_integrated_intensity(self, diffraction_pattern):
        roi = hs.roi.CircleROI(3, 3, 5)
        vi = diffraction_pattern.get_integrated_intensity(roi)
        assert vi.data.shape == (2, 2)
        assert vi.axes_manager.signal_dimension == 2
        assert vi.axes_manager.navigation_dimension == 0

    @pytest.mark.parametrize("out_signal_axes", [None, (0, 1), (1, 2), ("x", "y")])
    def test_get_integrated_intensity_stack(self, diffraction_pattern, out_signal_axes):
        s = hs.stack([diffraction_pattern] * 3)
        s.axes_manager.navigation_axes[0].name = "x"
        s.axes_manager.navigation_axes[1].name = "y"

        roi = hs.roi.CircleROI(3, 3, 5)
        vi = s.get_integrated_intensity(roi, out_signal_axes)
        assert vi.axes_manager.signal_dimension == 2
        assert vi.axes_manager.navigation_dimension == 1
        if out_signal_axes == (1, 2):
            assert vi.data.shape == (2, 3, 2)
            assert vi.axes_manager.navigation_size == 2
            assert vi.axes_manager.signal_shape == (2, 3)
        else:
            assert vi.data.shape == (3, 2, 2)
            assert vi.axes_manager.navigation_size == 3
            assert vi.axes_manager.signal_shape == (2, 2)

    def test_get_integrated_intensity_out_signal_axes(self, diffraction_pattern):
        s = hs.stack([diffraction_pattern] * 3)
        roi = hs.roi.CircleROI(3, 3, 5)
        vi = s.get_integrated_intensity(roi, out_signal_axes=(0, 1, 2))
        assert vi.axes_manager.signal_dimension == 3
        assert vi.axes_manager.navigation_dimension == 0
        assert vi.metadata.General.title == "Integrated intensity"
        assert (
            vi.metadata.Diffraction.integrated_range[:25]
            == "CircleROI(cx=3, cy=3, r=5"
        )

    def test_get_integrated_intensity_error(
        self, diffraction_pattern, out_signal_axes=(0, 1, 2)
    ):
        roi = hs.roi.CircleROI(3, 3, 5)
        with pytest.raises(ValueError):
            _ = diffraction_pattern.get_integrated_intensity(roi, out_signal_axes)

    def test_get_integrated_intensity_linescan(self, diffraction_pattern):
        s = diffraction_pattern.inav[0, :]
        s.metadata.General.title = ""
        roi = hs.roi.CircleROI(3, 3, 5)
        vi = s.get_integrated_intensity(roi)
        assert vi.data.shape == (2,)
        assert vi.axes_manager.signal_dimension == 1
        assert vi.axes_manager.navigation_dimension == 0
        assert vi.metadata.Diffraction.integrated_range[:25] == "CircleROI(cx=3, cy=3, r=5"


class TestAzimuthalIntegrator:
    # Tests the setting of a Azimutal Integrator:
    @pytest.fixture
    def ones(self):
        ones_diff = Diffraction2D(data=np.ones(shape=(10, 10)))
        ones_diff.axes_manager.signal_axes[0].scale = 0.1
        ones_diff.axes_manager.signal_axes[1].scale = 0.1
        ones_diff.axes_manager.signal_axes[0].name = "kx"
        ones_diff.axes_manager.signal_axes[1].name = "ky"
        ones_diff.unit = "2th_deg"
        return ones_diff

    @pytest.mark.xfail()
    def test_set_ai_fail(self, ones):
        # Not enough parameters fed into .set_ai()
        ones.unit = "k_nm^-1"
        ai = ones.set_ai()

    @pytest.mark.xfail()
    def test_return_ai_fail(self, ones):
        # .ai hasn't been set
        ai = ones.ai


class TestGetDirectBeamPosition:
    def setup_method(self):
        px, py, dx, dy = 15, 10, 20, 26
        x_pos_list = np.linspace(dx / 2 - 5, dx / 2 + 5, px, dtype=np.int16)
        y_pos_list = np.linspace(dy / 2 - 7, dy / 2 + 7, py, dtype=np.int16)
        data = np.zeros((py, px, dy, dx), dtype=np.uint16)
        for ix in range(data.shape[1]):
            for iy in range(data.shape[0]):
                x_pos, y_pos = x_pos_list[ix], y_pos_list[iy]
                data[iy, ix, y_pos, x_pos] = 10
        s = Diffraction2D(data)
        self.s = s
        self.dx, self.dy = dx, dy
        self.x_pos_list, self.y_pos_list = x_pos_list, y_pos_list

    def test_blur(self):
        dx, dy = self.dx, self.dy
        s, x_pos_list, y_pos_list = self.s, self.x_pos_list, self.y_pos_list
        s_shift = s.get_direct_beam_position(method="blur", sigma=1)
        assert s.axes_manager.navigation_shape == s_shift.axes_manager.navigation_shape
        assert (-(x_pos_list - dx / 2) == s_shift.isig[0].data[0]).all()
        assert (-(y_pos_list - dy / 2) == s_shift.isig[1].data[:, 0]).all()

    def test_interpolate(self):
        dx, dy = self.dx, self.dy
        s, x_pos_list, y_pos_list = self.s, self.x_pos_list, self.y_pos_list
        s_shift = s.get_direct_beam_position(
            method="interpolate",
            sigma=1,
            upsample_factor=2,
            kind="nearest",
        )
        assert s.axes_manager.navigation_shape == s_shift.axes_manager.navigation_shape
        assert (-(x_pos_list - dx / 2) == s_shift.isig[0].data[0]).all()
        assert (-(y_pos_list - dy / 2) == s_shift.isig[1].data[:, 0]).all()

    def test_cross_correlate(self):
        s = self.s
        s_shift = s.get_direct_beam_position(
            method="cross_correlate", radius_start=0, radius_finish=1
        )

    def test_lazy_result_none_non_lazy_signal(self):
        s = self.s
        s_shift = s.get_direct_beam_position(method="blur", sigma=1)
        assert not s_shift._lazy

    def test_lazy_result_none_lazy_signal(self):
        s = self.s.as_lazy()
        s_shift = s.get_direct_beam_position(method="blur", sigma=1)
        assert s_shift._lazy

    def test_lazy_result(self):
        s = self.s
        s_shift = s.get_direct_beam_position(method="blur", sigma=1, lazy_result=True)
        assert hasattr(s_shift.data, "compute")
        s_shift.compute()

    def test_lazy_input_non_lazy_result(self):
        s = LazyDiffraction2D(da.from_array(self.s.data))
        s_shift = s.get_direct_beam_position(method="blur", sigma=1, lazy_result=False)
        assert not hasattr(s_shift.data, "compute")
        assert not hasattr(s_shift, "compute")

    def test_lazy_input_lazy_result(self):
        s = LazyDiffraction2D(da.from_array(self.s.data))
        s_shift = s.get_direct_beam_position(method="blur", sigma=1, lazy_result=True)
        assert hasattr(s_shift.data, "compute")
        s_shift.compute()

    def test_non_uniform_chunks(self):
        s = LazyDiffraction2D(da.from_array(self.s.data, chunks=(8, 7, 10, 12)))
        s_shift = s.get_direct_beam_position(method="blur", sigma=1, lazy_result=True)
        shift_data_shape = s.data.shape[:-2] + (2,)
        assert s_shift.data.shape == shift_data_shape
        s_shift.compute()
        assert s_shift.data.shape == shift_data_shape


class TestCenterDirectBeam:
    def setup_method(self):
        data = np.zeros((8, 6, 20, 16), dtype=np.int16)
        x_pos_list = np.random.randint(8 - 2, 8 + 2, 6, dtype=np.int16)
        x_pos_list[x_pos_list == 8] = 9
        y_pos_list = np.random.randint(10 - 2, 10 + 2, 8, dtype=np.int16)
        for ix in range(len(x_pos_list)):
            for iy in range(len(y_pos_list)):
                data[iy, ix, y_pos_list[iy], x_pos_list[ix]] = 9
        s = Diffraction2D(data)
        s.axes_manager[0].scale = 0.5
        s.axes_manager[1].scale = 0.6
        s.axes_manager[2].scale = 3
        s.axes_manager[3].scale = 4
        s_lazy = s.as_lazy()
        self.s = s
        self.s_lazy = s_lazy
        self.x_pos_list = x_pos_list
        self.y_pos_list = y_pos_list

    def test_non_lazy(self):
        s = self.s
        s.center_direct_beam(method="blur", sigma=1)
        assert s._lazy is False
        assert (s.data[:, :, 10, 8] == 9).all()
        # Make sure only the pixel we expect to change, has actually changed
        s.data[:, :, 10, 8] = 0
        assert not s.data.any()

    def test_non_lazy_lazy_result(self):
        s = self.s
        s.center_direct_beam(method="blur", sigma=1, lazy_result=True)
        assert s._lazy is True
        s.compute()
        assert (s.data[:, :, 10, 8] == 9).all()
        s.data[:, :, 10, 8] = 0
        assert not s.data.any()

    def test_lazy(self):
        s_lazy = self.s_lazy
        s_lazy.center_direct_beam(method="blur", sigma=1)
        assert s_lazy._lazy is True
        s_lazy.compute()
        assert (s_lazy.data[:, :, 10, 8] == 9).all()
        s_lazy.data[:, :, 10, 8] = 0
        assert not s_lazy.data.any()

    def test_lazy_not_lazy_result(self):
        s_lazy = self.s_lazy
        s_lazy.center_direct_beam(method="blur", sigma=1, lazy_result=False)
        assert s_lazy._lazy is False
        assert (s_lazy.data[:, :, 10, 8] == 9).all()
        s_lazy.data[:, :, 10, 8] = 0
        assert not s_lazy.data.any()

    def test_non_uniform_chunks(self):
        s_lazy = self.s_lazy
        s_lazy.data = s_lazy.data.rechunk((5, 4, 12, 14))
        s_lazy_shape = s_lazy.axes_manager.shape
        data_lazy_shape = s_lazy.data.shape
        s_lazy.center_direct_beam(method="blur", sigma=1, lazy_result=True)
        assert s_lazy.axes_manager.shape == s_lazy_shape
        assert s_lazy.data.shape == data_lazy_shape
        s_lazy.compute()
        assert s_lazy.axes_manager.shape == s_lazy_shape
        assert s_lazy.data.shape == data_lazy_shape

    def test_return_shifts_non_lazy(self):
        s = self.s
        s_shifts = s.center_direct_beam(method="blur", sigma=1, return_shifts=True)
        assert s_shifts._lazy is False
        nav_dim = s.axes_manager.navigation_dimension
        assert nav_dim == s_shifts.axes_manager.navigation_dimension
        x_pos_list, y_pos_list = self.x_pos_list, self.y_pos_list
        assert ((8 - x_pos_list) == s_shifts.isig[0].data[0]).all()
        assert ((10 - y_pos_list) == s_shifts.isig[1].data[:, 0]).all()

    def test_return_shifts_lazy(self):
        s_lazy = self.s_lazy
        s_shifts = s_lazy.center_direct_beam(method="blur", sigma=1, return_shifts=True)
        assert s_shifts._lazy is True
        s_shifts.compute()
        x_pos_list, y_pos_list = self.x_pos_list, self.y_pos_list
        assert ((8 - x_pos_list) == s_shifts.isig[0].data[0]).all()
        assert ((10 - y_pos_list) == s_shifts.isig[1].data[:, 0]).all()

    def test_return_axes_manager(self):
        s = self.s
        s.center_direct_beam(method="blur", sigma=1)
        assert s.axes_manager[0].scale == 0.5
        assert s.axes_manager[1].scale == 0.6
        assert s.axes_manager[2].scale == 3
        assert s.axes_manager[3].scale == 4

    def test_shifts_input(self):
        s = self.s
        s_shifts = s.get_direct_beam_position(method="blur", sigma=1, lazy_result=False)
        s.center_direct_beam(shifts=s_shifts)
        assert (s.data[:, :, 10, 8] == 9).all()
        s.data[:, :, 10, 8] = 0
        assert not s.data.any()

    def test_shifts_input_lazy(self):
        s = self.s
        s_shifts = s.get_direct_beam_position(method="blur", sigma=1, lazy_result=True)
        s.center_direct_beam(shifts=s_shifts)
        assert (s.data[:, :, 10, 8] == 9).all()
        s.data[:, :, 10, 8] = 0
        assert not s.data.any()

    def test_subpixel(self):
        s = self.s
        s_shifts = s.get_direct_beam_position(method="blur", sigma=1)
        s_shifts += 0.5
        s.change_dtype("float32")
        s.center_direct_beam(shifts=s_shifts, subpixel=True)
        assert (s.data[:, :, 10:12, 8:10] == 9 / 4).all()
        s.data[:, :, 10:12, 8:10] = 0.0
        assert not s.data.any()

    def test_not_subpixel(self):
        s = self.s
        s_shifts = s.get_direct_beam_position(method="blur", sigma=1)
        s_shifts += 0.3
        s.change_dtype("float32")
        s.center_direct_beam(shifts=s_shifts, subpixel=False)
        assert (s.data[:, :, 10, 8] == 9).all()
        s.data[:, :, 10, 8] = 0.0
        assert not s.data.any()

    @pytest.mark.parametrize(
        "shape", [(20, 20), (10, 20, 20), (8, 10, 20, 20), (6, 8, 10, 20, 20)]
    )
    def test_different_dimensions(self, shape):
        s = Diffraction2D(np.random.randint(0, 256, size=shape))
        s.center_direct_beam(method="blur", sigma=1)
        assert s.data.shape == shape

    def test_half_square_width(self):
        # Generating a larger dataset to check that half_square_width
        # works properly with the automatic chunking in dask_tools._get_dask_array
        s = Diffraction2D(np.zeros((10, 10, 200, 200)))
        x_pos_list = np.random.randint(100 - 2, 100 + 2, 10, dtype=np.int16)
        y_pos_list = np.random.randint(100 - 2, 100 + 2, 10, dtype=np.int16)
        for ix in range(len(x_pos_list)):
            for iy in range(len(y_pos_list)):
                s.data[iy, ix, y_pos_list[iy], x_pos_list[ix]] = 9
        s.data[:, :, 1, -1] = 1000

        s1 = s.deepcopy()
        s.center_direct_beam(method="blur", sigma=1)
        assert (s.data[:, :, 100, 100] == 1000).all()
        s1.center_direct_beam(method="blur", sigma=1, half_square_width=5)
        assert (s1.data[:, :, 100, 100] == 9).all()

    def test_align_kwargs(self):
        s = self.s
        s.data += 1
        s1 = s.deepcopy()
        s.center_direct_beam(method="blur", sigma=1)
        assert (s.data == 0).any()
        s1.center_direct_beam(method="blur", sigma=1, align_kwargs={"mode": "wrap"})
        assert not (s1.data == 0).any()

    def test_method_interpolate(self):
        s = self.s
        s.center_direct_beam(method="interpolate", sigma=1, upsample_factor=10, kind=1)

    def test_method_cross_correlate(self):
        s = self.s
        s.center_direct_beam(method="cross_correlate", radius_start=0, radius_finish=2)

    def test_parameter_both_method_and_shifts(self):
        s = self.s
        with pytest.raises(ValueError):
            s.center_direct_beam(method="blur", sigma=1, shifts=np.ones((8, 6, 2)))

    def test_parameter_neither_method_and_shifts(self):
        s = self.s
        with pytest.raises(ValueError):
            s.center_direct_beam()


class TestDiffraction2DVirtualAnnularDarkField:
    def test_simple(self):
        shape = (5, 9, 12, 14)
        s = Diffraction2D(np.zeros(shape))
        s1 = s.lazy_virtual_annular_dark_field(cx=6, cy=6, r_inner=2, r=5)
        assert s1.axes_manager.signal_shape == (shape[1], shape[0])
        assert s1.data.sum() == 0.0

    def test_r_smaller_than_r_inner(self):
        shape = (5, 9, 12, 14)
        s = Diffraction2D(np.zeros(shape))
        with pytest.raises(ValueError):
            s.lazy_virtual_annular_dark_field(cx=2, cy=2, r_inner=5, r=2)

    def test_one_value(self):
        shape = (5, 9, 12, 14)
        s = Diffraction2D(np.zeros(shape))
        s.data[:, :, 9, 9] = 1
        s1 = s.lazy_virtual_annular_dark_field(cx=6, cy=6, r_inner=2, r=5)
        assert s1.axes_manager.signal_shape == (shape[1], shape[0])
        assert (s1.data == 1.0).all()

    def test_lazy(self):
        shape = (5, 9, 12, 14)
        data = da.zeros((5, 9, 12, 14), chunks=(10, 10, 10, 10))
        s = LazyDiffraction2D(data)
        s1 = s.lazy_virtual_annular_dark_field(cx=6, cy=6, r_inner=2, r=5)
        assert s1.axes_manager.signal_shape == (shape[1], shape[0])


class TestDiffraction2DVirtualBrightField:
    def test_simple(self):
        shape = (5, 9, 12, 14)
        s = Diffraction2D(np.zeros(shape))
        s1 = s.lazy_virtual_bright_field()
        assert s1.axes_manager.signal_shape == (shape[1], shape[0])
        assert s1.data.sum() == 0.0

    def test_one_value(self):
        shape = (5, 9, 12, 14)
        s = Diffraction2D(np.zeros(shape))
        s.data[:, :, 10, 13] = 1
        s1 = s.lazy_virtual_bright_field()
        assert s1.axes_manager.signal_shape == (shape[1], shape[0])
        assert (s1.data == 1.0).all()

        s2 = s.lazy_virtual_bright_field(6, 6, 2)
        assert s2.axes_manager.signal_shape == (shape[1], shape[0])
        assert s2.data.sum() == 0

    def test_lazy(self):
        shape = (5, 9, 12, 14)
        data = da.zeros((5, 9, 12, 14), chunks=(10, 10, 10, 10))
        s = LazyDiffraction2D(data)
        s1 = s.lazy_virtual_bright_field(cx=6, cy=6, r=5)
        assert s1.axes_manager.signal_shape == (shape[1], shape[0])

    def test_lazy_result(self):
        data = da.ones((10, 10, 20, 20), chunks=(10, 10, 10, 10))
        s = LazyDiffraction2D(data)
        s_out = s.lazy_virtual_bright_field(lazy_result=True)
        assert s_out._lazy


class TestDiffraction2DFindPeaksLazy:

    method1 = ["dog", "log"]

    @pytest.mark.parametrize("methods", method1)
    @pytest.mark.xfail(reason="Non-lazy input")
    def test_simple(self, methods):
        s = Diffraction2D(np.random.randint(100, size=(3, 2, 10, 20)))
        peak_array = s.find_peaks_lazy(method=methods)

    def test_not_existing_method(self):
        s = LazyDiffraction2D(da.zeros((2, 2, 5, 5), chunks=(1, 1, 5, 5)))
        with pytest.raises(ValueError):
            s.find_peaks_lazy(method="magic")

    @pytest.mark.parametrize("methods", method1)
    def test_lazy_input(self, methods):
        data = np.random.randint(100, size=(3, 2, 10, 20))
        s = LazyDiffraction2D(da.from_array(data, chunks=(1, 1, 5, 10)))
        peak_array = s.find_peaks_lazy(method=methods)
        assert s.data.shape[:2] == peak_array.shape
        assert hasattr(peak_array, "compute")

    @pytest.mark.parametrize("methods", method1)
    def test_lazy_output(self, methods):
        data = np.random.randint(100, size=(3, 2, 10, 20))
        s = LazyDiffraction2D(da.from_array(data, chunks=(1, 1, 5, 10)))
        peak_array = s.find_peaks_lazy(method=methods, lazy_result=False)
        assert s.data.shape[:2] == peak_array.shape
        assert not hasattr(peak_array, "compute")

    @pytest.mark.parametrize("nav_dims", [0, 1, 2, 3, 4])
    @pytest.mark.parametrize("methods", method1)
    def test_different_dimensions(self, nav_dims, methods):
        shape = list(np.random.randint(2, 6, size=nav_dims))
        shape.extend([50, 50])
        s = Diffraction2D(np.random.random(size=shape)).as_lazy()
        peak_array = s.find_peaks_lazy(method=methods, lazy_result=False)
        assert peak_array.shape == tuple(shape[:-2])


class TestDiffraction2DIntensityPeaks:
    def test_non_lazy(self):
        s = Diffraction2D(np.random.rand(3, 2, 10, 20))
        peak_array = s.find_peaks(interactive=False)
        intensity_array = s.intensity_peaks(peak_array.data)
        assert s.data.shape[:2] == intensity_array.shape
        assert hasattr(intensity_array, "compute")

    def test_lazy_input(self):
        data = np.random.randint(100, size=(3, 2, 10, 20))
        s = LazyDiffraction2D(da.from_array(data, chunks=(1, 1, 5, 10)))
        peak_array = s.find_peaks_lazy()
        intensity_array = s.intensity_peaks(peak_array)
        assert s.data.shape[:2] == intensity_array.shape
        assert hasattr(intensity_array, "compute")

    def test_lazy_output(self):
        data = np.random.randint(100, size=(3, 2, 10, 20))
        s = LazyDiffraction2D(da.from_array(data, chunks=(1, 1, 5, 10)))
        peak_array = s.find_peaks_lazy()
        intensity_array = s.intensity_peaks(peak_array, lazy_result=False)
        assert s.data.shape[:2] == intensity_array.shape
        assert not hasattr(intensity_array, "compute")

    @pytest.mark.parametrize("nav_dims", [0, 1, 2, 3, 4])
    def test_different_dimensions(self, nav_dims):
        shape = list(np.random.randint(2, 6, size=nav_dims))
        shape.extend([50, 50])
        s = Diffraction2D(np.random.random(size=shape)).as_lazy()
        peak_array = s.find_peaks_lazy()
        intensity_array = s.intensity_peaks(peak_array, disk_r=1)
        assert intensity_array.shape == tuple(shape[:-2])


class TestDiffraction2DPeakPositionRefinement:
    def test_non_lazy(self):
        s = Diffraction2D(np.random.rand(3, 2, 10, 20))
        peak_array = s.find_peaks(interactive=False)
        refined_peak_array = s.peak_position_refinement_com(peak_array.data, 4)
        assert s.data.shape[:2] == refined_peak_array.shape
        assert hasattr(refined_peak_array, "compute")

    def test_wrong_square_size(self):
        s = Diffraction2D(np.random.randint(100, size=(3, 2, 10, 20)))
        peak_array = s.find_peaks(interactive=False)
        with pytest.raises(ValueError):
            s.peak_position_refinement_com(peak_array, square_size=5)

    def test_lazy_input(self):
        data = np.random.randint(100, size=(3, 2, 10, 20))
        s = LazyDiffraction2D(da.from_array(data, chunks=(1, 1, 5, 10)))
        peak_array = s.find_peaks_lazy()
        refined_peak_array = s.peak_position_refinement_com(peak_array, 4)
        assert s.data.shape[:2] == refined_peak_array.shape
        assert hasattr(refined_peak_array, "compute")

    def test_lazy_output(self):
        data = np.random.randint(100, size=(3, 2, 10, 20))
        s = LazyDiffraction2D(da.from_array(data, chunks=(1, 1, 5, 10)))
        peak_array = s.find_peaks_lazy()
        refined_peak_array = s.peak_position_refinement_com(
            peak_array, 4, lazy_result=False
        )
        assert s.data.shape[:2] == refined_peak_array.shape
        assert not hasattr(refined_peak_array, "compute")

    @pytest.mark.parametrize("nav_dims", [0, 1, 2, 3, 4])
    def test_different_dimensions(self, nav_dims):
        shape = list(np.random.randint(2, 6, size=nav_dims))
        shape.extend([50, 50])
        s = Diffraction2D(np.random.random(size=shape)).as_lazy()
        peak_array = s.find_peaks_lazy()
        refined_peak_array = s.peak_position_refinement_com(
            peak_array, 4, lazy_result=False
        )
        assert refined_peak_array.shape == tuple(shape[:-2])


class TestSubtractingDiffractionBackground:

    method1 = ["difference of gaussians", "median kernel", "radial median"]

    def test_simple_hdome(self):
        s = Diffraction2D(np.random.rand(3, 2, 200, 150))
        s_rem = s.subtract_diffraction_background(method="h-dome", h=0.25)
        assert s_rem.data.shape == s.data.shape
        assert not hasattr(s_rem.data, "compute")

    @pytest.mark.parametrize("methods", method1)
    def test_simple(self, methods):
        s = Diffraction2D(np.random.randint(100, size=(3, 2, 200, 150)))
        s_rem = s.subtract_diffraction_background(method=methods)
        assert s_rem.data.shape == s.data.shape
        assert hasattr(s_rem.data, "compute")

    @pytest.mark.parametrize("methods", method1)
    def test_lazy_input(self, methods):
        data = np.random.randint(100, size=(3, 2, 200, 150))
        s = LazyDiffraction2D(da.from_array(data, chunks=(1, 1, 20, 10)))
        s_rem = s.subtract_diffraction_background(method=methods)
        assert s.data.shape == s_rem.data.shape
        assert hasattr(s_rem.data, "compute")

    @pytest.mark.parametrize("methods", method1)
    def test_lazy_output(self, methods):
        data = np.random.randint(100, size=(3, 2, 200, 150))
        s = LazyDiffraction2D(da.from_array(data, chunks=(1, 1, 20, 20)))
        s_rem = s.subtract_diffraction_background(method=methods, lazy_result=False)
        assert s.data.shape == s_rem.data.shape
        assert not hasattr(s_rem.data, "compute")

    @pytest.mark.parametrize("methods", method1)
    def test_axes_manager_copy(self, methods):
        s = Diffraction2D(np.random.randint(100, size=(5, 5, 200, 200)))
        ax_sa = s.axes_manager.signal_axes
        ax_na = s.axes_manager.navigation_axes
        ax_sa[0].name, ax_sa[1].name = "Detector x", "Detector y"
        ax_sa[0].scale, ax_sa[1].scale = 0.2, 0.2
        ax_sa[0].offset, ax_sa[1].offset = 10, 20
        ax_sa[0].units, ax_sa[1].units = "mrad", "mrad"
        ax_na[0].name, ax_na[1].name = "Probe x", "Probe y"
        ax_na[0].scale, ax_na[1].scale = 35, 35
        ax_na[0].offset, ax_na[1].offset = 54, 12
        ax_na[0].units, ax_na[1].units = "nm", "nm"
        s_temp = s.subtract_diffraction_background(method=methods)
        assert s.data.shape == s_temp.data.shape
        ax_sa_t = s_temp.axes_manager.signal_axes
        ax_na_t = s_temp.axes_manager.navigation_axes
        assert ax_sa[0].name == ax_sa_t[0].name
        assert ax_sa[1].name == ax_sa_t[1].name
        assert ax_sa[0].scale == ax_sa_t[0].scale
        assert ax_sa[1].scale == ax_sa_t[1].scale
        assert ax_sa[0].offset == ax_sa_t[0].offset
        assert ax_sa[1].offset == ax_sa_t[1].offset
        assert ax_sa[0].units == ax_sa_t[0].units
        assert ax_sa[1].units == ax_sa_t[1].units

        assert ax_na[0].name == ax_na_t[0].name
        assert ax_na[1].name == ax_na_t[1].name
        assert ax_na[0].scale == ax_na_t[0].scale
        assert ax_na[1].scale == ax_na_t[1].scale
        assert ax_na[0].offset == ax_na_t[0].offset
        assert ax_na[1].offset == ax_na_t[1].offset
        assert ax_na[0].units == ax_na_t[0].units
        assert ax_na[1].units == ax_na_t[1].units

    @pytest.mark.parametrize("nav_dims", [0, 1, 2, 3, 4])
    @pytest.mark.parametrize("methods", method1)
    def test_different_dimensions(self, nav_dims, methods):
        shape = list(np.random.randint(2, 6, size=nav_dims))
        shape.extend([200, 200])
        s = Diffraction2D(np.random.random(size=shape))
        st = s.subtract_diffraction_background(method=methods)
        assert st.data.shape == tuple(shape)

    def test_exception_not_implemented_method(self):
        s = Diffraction2D(np.zeros((2, 2, 10, 10)))
        with pytest.raises(NotImplementedError):
            s.subtract_diffraction_background(method="magic")


@pytest.mark.slow
class TestFindHotPixels:
    @pytest.fixture()
    def hot_pixel_data_2d(self):
        """Values are 50, except [21, 11] and [5, 38]
        being 50000 (to represent a "hot pixel").
        """
        data = np.ones((40, 50)) * 50
        data[21, 11] = 50000
        data[5, 38] = 50000
        dask_array = da.from_array(data, chunks=(5, 5))
        return LazyDiffraction2D(dask_array)

    @pytest.mark.parametrize("lazy_case", (True, False))
    def test_2d(self, hot_pixel_data_2d, lazy_case):
        if not lazy_case:
            hot_pixel_data_2d.compute()
        s_hot_pixels = hot_pixel_data_2d.find_hot_pixels(lazy_result=False)
        assert not s_hot_pixels._lazy
        assert s_hot_pixels.data.shape == hot_pixel_data_2d.data.shape
        assert s_hot_pixels.data[21, 11]
        assert s_hot_pixels.data[5, 38]
        s_hot_pixels.data[21, 11] = False
        s_hot_pixels.data[5, 38] = False
        assert not s_hot_pixels.data.any()

    def test_3d(self):
        data = np.ones((5, 40, 50)) * 50
        data[2, 21, 11] = 50000
        data[1, 5, 38] = 50000
        dask_array = da.from_array(data, chunks=(5, 5, 5))
        s = LazyDiffraction2D(dask_array)
        s_hot_pixels = s.find_hot_pixels()
        assert s_hot_pixels.data.shape == s.data.shape

    def test_4d(self):
        data = np.ones((10, 5, 40, 50)) * 50
        data[4, 2, 21, 11] = 50000
        data[6, 1, 5, 38] = 50000
        dask_array = da.from_array(data, chunks=(5, 5, 5, 5))
        s = LazyDiffraction2D(dask_array)
        s_hot_pixels = s.find_hot_pixels()
        assert s_hot_pixels.data.shape == s.data.shape

    def test_lazy_result(self, hot_pixel_data_2d):
        s_hot_pixels = hot_pixel_data_2d.find_hot_pixels(lazy_result=True)
        assert s_hot_pixels._lazy

    def test_threshold_multiplier(self, hot_pixel_data_2d):
        s_hot_pixels = hot_pixel_data_2d.find_hot_pixels(threshold_multiplier=1000000)
        assert not s_hot_pixels.data.any()

    def test_mask_array(self, hot_pixel_data_2d):
        mask_array = np.ones_like(hot_pixel_data_2d.data, dtype=bool)
        s_hot_pixels = hot_pixel_data_2d.find_hot_pixels(mask_array=mask_array)
        assert not s_hot_pixels.data.any()


class TestFindDeadPixels:
    @pytest.fixture()
    def dead_pixel_data_2d(self):
        """Values are 50, except [14, 42] and [2, 12]
        being 0 (to represent a "dead pixel").
        """
        data = np.ones((40, 50)) * 50
        data[14, 42] = 0
        data[2, 12] = 0
        dask_array = da.from_array(data, chunks=(5, 5))
        return LazyDiffraction2D(dask_array)

    @pytest.mark.parametrize("lazy_case", (True, False))
    def test_2d(self, dead_pixel_data_2d, lazy_case):
        if not lazy_case:
            dead_pixel_data_2d.compute()
        s_dead_pixels = dead_pixel_data_2d.find_dead_pixels(lazy_result=False)
        assert not s_dead_pixels._lazy
        assert s_dead_pixels.data.shape == dead_pixel_data_2d.data.shape
        assert s_dead_pixels.data[14, 42]
        assert s_dead_pixels.data[2, 12]
        s_dead_pixels.data[14, 42] = False
        s_dead_pixels.data[2, 12] = False
        assert not s_dead_pixels.data.any()

    def test_3d(self):
        data = np.ones((5, 40, 50)) * 50
        data[:, 14, 42] = 0
        data[:, 2, 12] = 0
        dask_array = da.from_array(data, chunks=(5, 5, 5))
        s = LazyDiffraction2D(dask_array)
        s_dead_pixels = s.find_dead_pixels()
        assert s_dead_pixels.data.shape == s.data.shape[-2:]

    def test_4d(self):
        data = np.ones((10, 5, 40, 50)) * 50
        data[:, :, 14, 42] = 0
        data[:, :, 2, 12] = 0
        dask_array = da.from_array(data, chunks=(5, 5, 5, 5))
        s = LazyDiffraction2D(dask_array)
        s_dead_pixels = s.find_dead_pixels()
        assert s_dead_pixels.data.shape == s.data.shape[-2:]

    def test_lazy_result(self, dead_pixel_data_2d):
        s_dead_pixels = dead_pixel_data_2d.find_dead_pixels(lazy_result=True)
        assert s_dead_pixels._lazy

    def test_dead_pixel_value(self, dead_pixel_data_2d):
        s_dead_pixels = dead_pixel_data_2d.find_dead_pixels(dead_pixel_value=-10)
        assert not s_dead_pixels.data.any()

    def test_mask_array(self, dead_pixel_data_2d):
        mask_array = np.ones_like(dead_pixel_data_2d.data, dtype=bool)
        s_dead_pixels = dead_pixel_data_2d.find_dead_pixels(mask_array=mask_array)
        assert not s_dead_pixels.data.any()


class TestCorrectBadPixel:
    @pytest.fixture()
    def data(self):
        data = np.ones((2, 2, 100, 90))
        data[:, :, 9, 81] = 50000
        data[:, :, 41, 21] = 0
        return data

    @pytest.fixture()
    def bad_pixels(self):
        return np.asarray([[41, 21], [9, 81]])

    @pytest.mark.xfail(reason="This array shape is not currently supported")
    def test_lazy(self, data, bad_pixels):
        s_lazy = Diffraction2D(data).as_lazy()
        s_lazy.correct_bad_pixels(bad_pixels, lazy_result=True)

    @pytest.mark.parametrize("lazy_input", (True, False))
    @pytest.mark.parametrize("lazy_result", (True, False))
    def test_lazy_with_bad_pixel_finders(self, data, lazy_input,lazy_result):
        s = Diffraction2D(data).as_lazy()
        if not lazy_input:
            s.compute()

        hot = s.find_hot_pixels(lazy_result=True)
        dead = s.find_dead_pixels(lazy_result=True)
        bad = hot + dead

        s = s.correct_bad_pixels(bad, lazy_result=lazy_result)
        assert s._lazy == lazy_result
        if lazy_result:
            s.compute()
        assert np.isclose(s.data[0, 0, 9, 81], 1)
        assert np.isclose(s.data[0, 0, 41, 21], 1)


class TestMakeProbeNavigation:
    def test_fast(self):
        s = Diffraction2D(np.ones((6, 5, 12, 10)))
        s.make_probe_navigation(method="fast")
        s_nav = s._navigator_probe
        assert s.axes_manager.navigation_shape == s_nav.axes_manager.signal_shape
        assert np.all(s_nav.data == 1)

    def test_slow(self):
        s = Diffraction2D(np.ones((5, 5, 12, 10)))
        s.make_probe_navigation(method="slow")
        s_nav = s._navigator_probe
        assert s.axes_manager.navigation_shape == s_nav.axes_manager.signal_shape
        assert np.all(s_nav.data == 120)

    def test_fast_lazy(self):
        s = LazyDiffraction2D(da.ones((6, 4, 60, 50), chunks=(2, 2, 10, 10)))
        s.make_probe_navigation(method="fast")
        s_nav = s._navigator_probe
        assert s.axes_manager.navigation_shape == s_nav.axes_manager.signal_shape
        assert np.all(s_nav.data == 100)

    def test_slow_lazy(self):
        s = LazyDiffraction2D(da.ones((6, 4, 60, 50), chunks=(2, 2, 10, 10)))
        s.make_probe_navigation(method="slow")
        s_nav = s._navigator_probe
        assert s.axes_manager.navigation_shape == s_nav.axes_manager.signal_shape
        assert np.all(s_nav.data == 3000)

    def test_very_asymmetric_size(self):
        s = Diffraction2D(np.ones((50, 2, 120, 10)))
        s.make_probe_navigation(method="fast")
        s_nav = s._navigator_probe
        assert s.axes_manager.navigation_shape == s_nav.axes_manager.signal_shape

    def test_very_asymmetric_size_lazy(self):
        s = LazyDiffraction2D(da.ones((50, 2, 120, 10), chunks=(2, 2, 5, 5)))
        s.make_probe_navigation(method="fast")
        s_nav = s._navigator_probe
        assert s.axes_manager.navigation_shape == s_nav.axes_manager.signal_shape

    @pytest.mark.parametrize("shape", [(2, 10, 10), (3, 2, 10, 10)])
    def test_different_shapes(self, shape):
        s = Diffraction2D(np.ones(shape))
        s.make_probe_navigation(method="fast")
        s_nav = s._navigator_probe
        assert s.axes_manager.navigation_shape == s_nav.axes_manager.signal_shape

    @pytest.mark.parametrize("shape", [(10, 20), (2, 3, 4, 5, 6), (2, 3, 4, 5, 6, 7)])
    def test_wrong_navigation_dimensions(self, shape):
        s = Diffraction2D(np.ones(shape))
        with pytest.raises(ValueError):
            s.make_probe_navigation(method="fast")


@pytest.mark.slow
class TestPlotNavigator:
    @pytest.mark.parametrize(
        "shape", [(9, 8), (5, 9, 8), (4, 5, 9, 8), (8, 4, 5, 9, 8), (9, 8, 4, 5, 9, 8)]
    )
    def test_non_lazy(self, shape):
        s = Diffraction2D(np.random.randint(0, 256, shape), dtype=np.uint8)
        s.plot()
        plt.close("all")

    @pytest.mark.parametrize(
        "shape", [(9, 8), (5, 9, 8), (4, 5, 9, 8), (8, 4, 5, 9, 8), (9, 8, 4, 5, 9, 8)]
    )
    def test_lazy(self, shape):
        s = LazyDiffraction2D(da.random.randint(0, 256, shape), dtype=np.uint8)
        s.plot()
        plt.close("all")

    def test_navigator_kwarg(self):
        s = Diffraction2D(np.random.randint(0, 256, (8, 9, 10, 30), dtype=np.uint8))
        s_nav = Diffraction2D(np.zeros((8, 9)))
        s.plot(navigator=s_nav)
        plt.close("all")

    @pytest.mark.xfail(reason="Designed failure")
    def test_wrong_navigator_shape_kwarg(self):
        s = Diffraction2D(np.random.randint(0, 256, (8, 9, 10, 30), dtype=np.uint8))
        s_nav = Diffraction2D(np.zeros((2, 19)))
        s._navigator_probe = s_nav
        s.plot()
