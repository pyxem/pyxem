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

import pytest
import numpy as np
import dask.array as da
import hyperspy.api as hs
from matplotlib import pyplot as plt
from numpy.random import default_rng
from skimage.draw import circle_perimeter_aa
import scipy

from pyxem.signals import (
    Diffraction1D,
    Diffraction2D,
    LazyDiffraction2D,
    PolarDiffraction2D,
    DiffractionVectors,
)
from pyxem.data.dummy_data import make_diffraction_test_data as mdtd


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
        assert dif.unit is "px"
        dif.unit = "!23"
        assert dif.unit is "px"

    def test_unit_set(self, ones):
        assert ones.unit == "2th_deg"

    def test_1d_azimuthal_integral_pyxem(self, ones):
        ones.calibration.center = None
        ones.calibration.scale = 0.2
        az = ones.get_azimuthal_integral1d(
            npt=10,
            inplace=True,
            radial_range=[0.0, 0.8],
            mean=True,
        )
        assert isinstance(ones, Diffraction1D)
        np.testing.assert_array_almost_equal(ones.data[0:8], np.ones(8))
        assert az is None

    def test_mask_mean(self, ones):
        ones.calibration.center = None
        ones.calibration.scale = 0.2
        mask = np.zeros(ones.data.shape, dtype=bool)
        mask[0:5, 0:5] = True
        ones.calibration.mask = mask
        az = ones.get_azimuthal_integral1d(
            npt=10,
            inplace=True,
            radial_range=[0.0, 0.8],
            mean=True,
        )
        assert isinstance(ones, Diffraction1D)
        np.testing.assert_array_almost_equal(ones.data[0:8], np.ones(8))
        assert az is None

    def test_1d_azimuthal_integral_inplace(self, ones):
        az = ones.get_azimuthal_integral1d(
            npt=10,
            inplace=True,
        )
        assert isinstance(ones, Diffraction1D)
        assert az is None

    def test_1d_azimuthal_integral_slicing(self, ones):
        ones.unit = "2th_rad"
        ones.calibration.center = None
        az = ones.get_azimuthal_integral1d(npt=10, radial_range=(0.0, 1.0), mean=True)
        np.testing.assert_array_almost_equal(az.data[0:7], np.ones(7))

    @pytest.mark.parametrize(
        "shape", [(20, 16), (3, 20, 16), (4, 3, 20, 16), (6, 4, 3, 20, 16)]
    )
    def test_lazy_input_lazy_output_different_shapes(self, shape):
        chunks = [5] * len(shape)
        s = LazyDiffraction2D(da.ones(shape, chunks=chunks))
        s.unit = "2th_deg"
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
        npt = 10
        s_a = s.get_azimuthal_integral1d(npt=npt, lazy_output=True)
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
        npt = 10
        s_a = s.get_azimuthal_integral1d(npt=npt, lazy_output=False)
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
        npt = 10
        output_signal_shape = s.axes_manager.shape[:-2] + (npt,)
        output_data_shape = shape[:-2] + (npt,)
        s.get_azimuthal_integral1d(npt=npt, inplace=True, lazy_output=True)
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
        npt = 10
        output_signal_shape = s.axes_manager.shape[:-2] + (npt,)
        output_data_shape = shape[:-2] + (npt,)
        s.get_azimuthal_integral1d(npt=npt, inplace=True, lazy_output=False)
        assert s.axes_manager.shape == output_signal_shape
        assert s.data.shape == output_data_shape

    @pytest.mark.parametrize(
        "shape", [(20, 16), (3, 20, 16), (4, 3, 20, 16), (6, 4, 3, 20, 16)]
    )
    def test_non_lazy_input_lazy_result_inplace(self, shape):
        s = Diffraction2D(np.ones(shape))
        s.unit = "2th_deg"
        npt = 10
        output_signal_shape = s.axes_manager.shape[:-2] + (npt,)
        output_data_shape = shape[:-2] + (npt,)
        s.get_azimuthal_integral1d(npt=npt, inplace=True, lazy_output=True)
        assert s.axes_manager.shape == output_signal_shape
        assert s.data.shape == output_data_shape
        s.compute()
        assert s.axes_manager.shape == output_signal_shape
        assert s.data.shape == output_data_shape


class TestVariance:
    @pytest.fixture
    def ones(self):
        ones_diff = Diffraction2D(data=np.ones(shape=(10, 10, 10, 10)))
        ones_diff.calibration(scale=0.1, center=None)
        return ones_diff

    @pytest.fixture
    def ones_zeros(self):
        data = np.ones(shape=(10, 10, 10, 10))
        data[0:10:2, :, :, :] = 2
        ones_diff = Diffraction2D(data=data)
        ones_diff.calibration(scale=0.1, center=None)
        return ones_diff

    @pytest.fixture
    def bulls_eye_noisy(self):
        x, y = np.mgrid[-25:25, -25:25]
        data = (x**2 + y**2) ** 0.5
        data = np.tile(data, (5, 5, 1, 1))
        # Electron is equal to 1 count in image
        rng = default_rng(seed=1)
        data = rng.poisson(lam=data)
        ones_diff = Diffraction2D(data=data)
        ones_diff.calibration(scale=0.1, center=None)
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
            bulls_eye_variance.data[5:], np.zeros(20), decimal=1
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
    def arange(self):
        # signal looks as follows:
        # 0 1
        # 2 3
        arange_diff = Diffraction2D(data=np.arange(4).reshape(2, 2))
        arange_diff.axes_manager.signal_axes[0].scale = 1
        arange_diff.axes_manager.signal_axes[1].scale = 1
        arange_diff.axes_manager.signal_axes[0].name = "kx"
        arange_diff.axes_manager.signal_axes[1].name = "ky"
        arange_diff.unit = "2th_deg"
        return arange_diff

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

    def test_internal_azimuthal_integration(self, ring):
        ring.calibration(scale=1)
        az = ring.get_azimuthal_integral2d(npt=40, npt_azim=100, radial_range=(0, 40))
        ring_sum = np.sum(az.data, axis=1)
        assert ring_sum.shape == (40,)

    @pytest.mark.parametrize(
        "corner",
        [
            (0, 0),
            (0, -1),
            (-1, 0),
            (-1, -1),
        ],
    )
    @pytest.mark.parametrize(
        "shape",
        [
            (10, 10),  # Even square
            (9, 9),  # Odd square
            (4, 6),  # Even
            (5, 9),  # Odd
            (5, 10),  # Odd and even
            (10, 5),  # Even and odd
        ],
    )
    def test_internal_azimuthal_integration_data_range(self, corner, shape):
        # Test that the edges of the cartesian data are included in the polar transform

        max_val = 10
        data = np.zeros(shape)
        data[corner] = max_val

        signal = Diffraction2D(data)

        # Reality check
        assert np.allclose(np.nanmax(signal.data), max_val)

        # Use mean=True to conserve values
        pol = signal.get_azimuthal_integral2d(npt=20, mean=True)

        assert np.allclose(np.nanmax(pol.data), max_val)

    # polar unwrapping `arange` should look like [1 0 2 3]
    # since data looks like:
    # 0 1
    # 2 3
    # and the data gets unwrapped from the center to the right
    @pytest.mark.parametrize(
        [
            "azimuthal_range",
            "expected_output",
        ],
        [
            [
                (0 * np.pi / 2, 1 * np.pi / 2),
                1,
            ],
            [
                (1 * np.pi / 2, 2 * np.pi / 2),
                0,
            ],
            [
                (2 * np.pi / 2, 3 * np.pi / 2),
                2,
            ],
            [
                (3 * np.pi / 2, 4 * np.pi / 2),
                3,
            ],
        ],
    )
    def test_azimuthal_integration_range(
        self, arange, azimuthal_range, expected_output
    ):
        arange.calibration.center = None  # set center
        quadrant = arange.get_azimuthal_integral2d(
            npt=10, npt_azim=10, azimuth_range=azimuthal_range, mean=True
        )
        assert np.allclose(quadrant.data[~np.isnan(quadrant.data)], expected_output)


class TestVirtualImaging:
    # Tests that virtual imaging runs without failure

    def test_plot_integrated_intensity(self, diffraction_pattern):
        roi = hs.roi.CircleROI(3, 3, 5)
        plt.ion()  # to make plotting non-blocking
        diffraction_pattern.plot_integrated_intensity(roi)
        plt.close("all")

    @pytest.mark.parametrize("has_nan", [True, False])
    def test_get_integrated_intensity(self, diffraction_pattern, has_nan):
        roi = hs.roi.CircleROI(3, 3, 5)
        if has_nan:
            diffraction_pattern.isig[:2] = np.nan
        vi = diffraction_pattern.get_integrated_intensity(roi)
        assert vi.axes_manager.signal_dimension == 2
        assert vi.axes_manager.navigation_dimension == 0
        np.testing.assert_allclose(vi.data, np.array([[6.0, 6.0], [8.0, 10.0]]))

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
            vi.metadata.Diffraction.integrated_range[:25] == "CircleROI(cx=3, cy=3, r=5"
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
        assert (
            vi.metadata.Diffraction.integrated_range[:25] == "CircleROI(cx=3, cy=3, r=5"
        )


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

    @pytest.mark.parametrize(
        "sig_slice", (None, (2, 18, 2, 24), (2.0, 18.0, 2.0, 24.0))
    )
    @pytest.mark.parametrize(
        "method,kwargs",
        [
            ("cross_correlate", {"radius_start": 0, "radius_finish": 2}),
            (
                "blur",
                {
                    "sigma": 1,
                },
            ),
            (
                "interpolate",
                {
                    "sigma": 1,
                    "upsample_factor": 2,
                    "kind": "nearest",
                },
            ),
            ("center_of_mass", {"mask": (10, 13, 10)}),
        ],
    )
    def test_get_direct_beam(self, method, sig_slice, kwargs):
        s = self.s
        dx, dy = self.dx, self.dy
        s, x_pos_list, y_pos_list = self.s, self.x_pos_list, self.y_pos_list
        s_shift = s.get_direct_beam_position(
            method=method, signal_slice=sig_slice, **kwargs
        )
        if method == "cross_correlate":
            # shifted by half a pixel
            assert (-(x_pos_list - dx / 2) == s_shift.isig[0].data[0] + 0.5).all()
            assert (-(y_pos_list - dy / 2) == s_shift.isig[1].data[:, 0] + 0.5).all()
        else:
            np.testing.assert_array_almost_equal(
                -(x_pos_list - dx / 2), s_shift.isig[0].data[0]
            )
            np.testing.assert_array_almost_equal(
                -(y_pos_list - dy / 2), s_shift.isig[1].data[:, 0]
            )
        assert s.axes_manager.navigation_shape == s_shift.axes_manager.navigation_shape

    def test_fail_get_direct_beam(self):
        with pytest.raises(ValueError):
            s = self.s
            s_shift = s.get_direct_beam_position(
                method="blur",
                sigma=2,
                signal_slice=(5, 8, 5, 8),
                half_square_width=1,
            )

    def test_center_of_mass(self):
        dx, dy = self.dx, self.dy
        s, x_pos_list, y_pos_list = self.s, self.x_pos_list, self.y_pos_list
        s_shift = s.get_direct_beam_position(method="center_of_mass")
        assert s.axes_manager.navigation_shape == s_shift.axes_manager.navigation_shape
        assert (-(x_pos_list - dx / 2) == s_shift.isig[0].data[0]).all()
        assert (-(y_pos_list - dy / 2) == s_shift.isig[1].data[:, 0]).all()

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
        s_shift = s.get_direct_beam_position(method="blur", sigma=1, lazy_output=True)
        assert hasattr(s_shift.data, "compute")
        s_shift.compute()

    def test_lazy_input_non_lazy_result(self):
        s = LazyDiffraction2D(da.from_array(self.s.data))
        s_shift = s.get_direct_beam_position(method="blur", sigma=1, lazy_output=False)
        assert not hasattr(s_shift.data, "compute")
        assert not hasattr(s_shift, "compute")

    def test_lazy_input_lazy_result(self):
        s = LazyDiffraction2D(da.from_array(self.s.data))
        s_shift = s.get_direct_beam_position(method="blur", sigma=1, lazy_output=True)
        assert hasattr(s_shift.data, "compute")
        s_shift.compute()

    def test_non_uniform_chunks(self):
        s = LazyDiffraction2D(da.from_array(self.s.data, chunks=(8, 7, 10, 12)))
        s_shift = s.get_direct_beam_position(method="blur", sigma=1, lazy_output=True)
        shift_data_shape = s.data.shape[:-2] + (2,)
        assert s_shift.data.shape == shift_data_shape
        s_shift.compute()
        assert s_shift.data.shape == shift_data_shape


class TestDiffraction2DGetDirectBeamPositionCenterOfMass:
    def test_center_of_mass_0d(self):
        x_im, y_im, x, y = 7, 9, 2, 3
        array = np.zeros(shape=(y_im, x_im))
        array[y, x] = 1
        s = Diffraction2D(array)
        s_bs = s.get_direct_beam_position(method="center_of_mass")
        x_new, y_new = (x_im / 2) - x, (y_im / 2) - y
        assert (s_bs.isig[0].data == x_new).all()
        assert (s_bs.isig[1].data == y_new).all()
        assert s_bs.axes_manager.navigation_shape == ()
        assert s_bs.axes_manager.signal_shape == (2,)

    def test_center_of_mass_1d(self):
        x_im, y_im, x, y = 7, 9, 2, 3
        array = np.zeros(shape=(5, y_im, x_im))
        array[:, y, x] = 1
        s = Diffraction2D(array)
        s_bs = s.get_direct_beam_position(method="center_of_mass")
        x_new, y_new = (x_im / 2) - x, (y_im / 2) - y
        assert (s_bs.isig[0].data == x_new).all()
        assert (s_bs.isig[1].data == y_new).all()
        assert s_bs.axes_manager.navigation_shape == (5,)
        assert s_bs.axes_manager.signal_shape == (2,)

    def test_center_of_mass(self):
        x_im, y_im, x, y = 10, 10, 5, 7
        array = np.zeros(shape=(10, 10, y_im, x_im))
        array[:, :, y, x] = 1
        s = Diffraction2D(array)
        s_bs = s.get_direct_beam_position(method="center_of_mass")
        x_new, y_new = (x_im / 2) - x, (y_im / 2) - y
        assert (s_bs.isig[0].data == x_new).all()
        assert (s_bs.isig[1].data == y_new).all()
        assert s_bs.axes_manager.navigation_shape == (10, 10)
        assert s_bs.axes_manager.signal_shape == (2,)

    def test_center_of_mass_random_position(self):
        x_im, y_im = 10, 10
        array = np.zeros(shape=(10, 10, y_im, x_im))
        x_array = np.random.randint(0, 10, size=(y_im, x_im))
        y_array = np.random.randint(0, 10, size=(y_im, x_im))
        for i in range(10):
            for j in range(10):
                array[i, j, y_array[i, j], x_array[i, j]] = 1
        s = Diffraction2D(array)
        s_bs = s.get_direct_beam_position(method="center_of_mass")
        x_new, y_new = (x_im / 2) - x_array, (y_im / 2) - y_array
        assert (s_bs.isig[0].data == x_new.astype(float)).all()
        assert (s_bs.isig[1].data == y_new.astype(float)).all()

    def test_center_of_mass_different_shapes(self):
        x_nav, y_nav, x_im, y_im = 10, 5, 8, 15
        array = np.zeros(shape=(y_nav, x_nav, y_im, x_im))
        x_array = np.random.randint(1, 7, size=(5, 10))
        y_array = np.random.randint(1, 14, size=(5, 10))
        for i in range(5):
            for j in range(10):
                array[i, j, y_array[i, j], x_array[i, j]] = 1
        s = Diffraction2D(array)
        s_bs = s.get_direct_beam_position(method="center_of_mass")
        x_new, y_new = (x_im / 2) - x_array, (y_im / 2) - y_array
        assert (s_bs.isig[0].data == x_new).all()
        assert (s_bs.isig[1].data == y_new).all()
        assert s_bs.axes_manager.navigation_shape == (x_nav, y_nav)
        assert s_bs.axes_manager.signal_shape == (2,)

    def test_center_of_mass_different_shapes2(self):
        psX, psY = 11, 9
        s = mdtd.generate_4d_data(probe_size_x=psX, probe_size_y=psY, ring_x=None)
        s_bs = s.get_direct_beam_position(method="center_of_mass")
        assert s_bs.axes_manager.shape == (psX, psY, 2)

    def test_different_shape_no_blur_no_downscale(self):
        y, x = np.mgrid[75:83:9j, 85:95:11j]
        x_im, y_im = 160, 140
        s = mdtd.generate_4d_data(
            probe_size_x=11,
            probe_size_y=9,
            ring_x=None,
            image_size_x=x_im,
            image_size_y=y_im,
            disk_x=x,
            disk_y=y,
            disk_r=40,
            disk_I=20,
            blur=False,
            blur_sigma=1,
            downscale=False,
        )
        s_bs = s.get_direct_beam_position(method="center_of_mass")
        x_new, y_new = (x_im / 2) - x, (y_im / 2) - y
        assert (s_bs.isig[0].data == x_new).all()
        assert (s_bs.isig[1].data == y_new).all()

    def test_different_shape_no_downscale(self):
        y, x = np.mgrid[75:83:9j, 85:95:11j]
        x_im, y_im = 160, 140
        s = mdtd.generate_4d_data(
            probe_size_x=11,
            probe_size_y=9,
            ring_x=None,
            image_size_x=x_im,
            image_size_y=y_im,
            disk_x=x,
            disk_y=y,
            disk_r=40,
            disk_I=20,
            blur=True,
            blur_sigma=1,
            downscale=False,
        )
        s_bs = s.get_direct_beam_position(method="center_of_mass")
        x_new, y_new = (x_im / 2) - x, (y_im / 2) - y
        np.testing.assert_allclose(s_bs.isig[0].data, x_new)
        np.testing.assert_allclose(s_bs.isig[1].data, y_new)

    def test_mask(self):
        y, x = np.mgrid[75:83:9j, 85:95:11j]
        x_im, y_im = 160, 140
        s = mdtd.generate_4d_data(
            probe_size_x=11,
            probe_size_y=9,
            ring_x=None,
            image_size_x=x_im,
            image_size_y=y_im,
            disk_x=x,
            disk_y=y,
            disk_r=40,
            disk_I=20,
            blur=False,
            blur_sigma=1,
            downscale=False,
        )
        s.data[:, :, 15, 10] = 1000000
        s_bs0 = s.get_direct_beam_position(method="center_of_mass")
        s_bs1 = s.get_direct_beam_position(method="center_of_mass", mask=(90, 79, 60))
        x_new, y_new = (x_im / 2) - x, (y_im / 2) - y
        assert not (s_bs0.isig[0].data == x_new).all()
        assert not (s_bs0.isig[1].data == y_new).all()
        assert (s_bs1.isig[0].data == x_new).all()
        assert (s_bs1.isig[1].data == y_new).all()

    def test_mask_2(self):
        x, y = 60, 50
        x_im, y_im = 120, 100
        s = mdtd.generate_4d_data(
            probe_size_x=5,
            probe_size_y=5,
            ring_x=None,
            image_size_x=x_im,
            image_size_y=y_im,
            disk_x=x,
            disk_y=y,
            disk_r=20,
            disk_I=20,
            blur=False,
            downscale=False,
        )
        x_new, y_new = (x_im / 2) - x, (y_im / 2) - y

        # Add one large value
        s.data[:, :, 50, 30] = 200000  # Large value to the left of the disk

        # Center of mass should not be in center of the disk, due to the
        # large value.
        s_bs1 = s.get_direct_beam_position(method="center_of_mass")
        assert not (s_bs1.isig[0].data == x_new).all()
        assert (s_bs1.isig[1].data == y_new).all()

        # Here, the large value is masked
        s_bs2 = s.get_direct_beam_position(method="center_of_mass", mask=(60, 50, 25))
        assert (s_bs2.isig[0].data == x_new).all()
        assert (s_bs2.isig[1].data == y_new).all()

        # Here, the large value is right inside the edge of the mask
        s_bs3 = s.get_direct_beam_position(method="center_of_mass", mask=(60, 50, 31))
        assert not (s_bs3.isig[0].data == x_new).all()
        assert (s_bs3.isig[1].data == y_new).all()

        # Here, the large value is right inside the edge of the mask
        s_bs4 = s.get_direct_beam_position(method="center_of_mass", mask=(59, 50, 30))
        assert not (s_bs4.isig[0].data == x_new).all()
        assert (s_bs4.isig[1].data == y_new).all()

        s.data[:, :, 50, 30] = 0
        s.data[:, :, 80, 60] = 200000  # Large value under the disk

        # The large value is masked
        s_bs5 = s.get_direct_beam_position(method="center_of_mass", mask=(60, 50, 25))
        assert (s_bs5.isig[0].data == x_new).all()
        assert (s_bs5.isig[1].data == y_new).all()

        # The large value just not masked
        s_bs6 = s.get_direct_beam_position(method="center_of_mass", mask=(60, 50, 31))
        assert (s_bs6.isig[0].data == x_new).all()
        assert not (s_bs6.isig[1].data == y_new).all()

        # The large value just not masked
        s_bs7 = s.get_direct_beam_position(method="center_of_mass", mask=(60, 55, 25))
        assert (s_bs7.isig[0].data == x_new).all()
        assert not (s_bs7.isig[1].data == y_new).all()

    def test_threshold(self):
        x_im, y_im = 120, 100
        x, y = 60, 50
        s = mdtd.generate_4d_data(
            probe_size_x=4,
            probe_size_y=3,
            ring_x=None,
            image_size_x=x_im,
            image_size_y=y_im,
            disk_x=x,
            disk_y=y,
            disk_r=20,
            disk_I=20,
            blur=False,
            blur_sigma=1,
            downscale=False,
        )
        x_new, y_new = (x_im / 2) - x, (y_im / 2) - y
        s.data[:, :, 0:30, 0:30] = 5

        # The extra values are ignored due to thresholding
        s_bs0 = s.get_direct_beam_position(method="center_of_mass", threshold=2)
        assert (s_bs0.isig[0].data == x_new).all()
        assert (s_bs0.isig[1].data == y_new).all()

        # The extra values are not ignored
        s_bs1 = s.get_direct_beam_position(method="center_of_mass", threshold=1)
        assert not (s_bs1.isig[0].data == x_new).all()
        assert not (s_bs1.isig[1].data == y_new).all()

        # The extra values are not ignored
        s_bs2 = s.get_direct_beam_position(method="center_of_mass")
        assert not (s_bs2.isig[0].data == x_new).all()
        assert not (s_bs2.isig[1].data == y_new).all()

    def test_threshold_and_mask(self):
        x_im, y_im = 120, 100
        x, y = 60, 50
        s = mdtd.generate_4d_data(
            probe_size_x=4,
            probe_size_y=3,
            ring_x=None,
            image_size_x=x_im,
            image_size_y=y_im,
            disk_x=x,
            disk_y=y,
            disk_r=20,
            disk_I=20,
            blur=False,
            blur_sigma=1,
            downscale=False,
        )
        x_new, y_new = (x_im / 2) - x, (y_im / 2) - y
        s.data[:, :, 0:30, 0:30] = 5
        s.data[:, :, 1, -2] = 60

        # The extra values are ignored due to thresholding and mask
        s_bs0 = s.get_direct_beam_position(
            method="center_of_mass", threshold=3, mask=(60, 50, 50)
        )
        assert (s_bs0.isig[0].data == x_new).all()
        assert (s_bs0.isig[1].data == y_new).all()

        # The extra values are not ignored
        s_bs1 = s.get_direct_beam_position(method="center_of_mass", mask=(60, 50, 50))
        assert not (s_bs1.isig[0].data == x_new).all()
        assert not (s_bs1.isig[1].data == y_new).all()

        # The extra values are not ignored
        s_bs2 = s.get_direct_beam_position(method="center_of_mass", threshold=3)
        assert not (s_bs2.isig[0].data == x_new).all()
        assert not (s_bs2.isig[1].data == y_new).all()

        # The extra values are not ignored
        s_bs3 = s.get_direct_beam_position(method="center_of_mass")
        assert not (s_bs3.isig[0].data == x_new).all()
        assert not (s_bs3.isig[1].data == y_new).all()

    def test_1d_signal(self):
        x_im, y_im = 120, 100
        x = np.arange(45, 45 + 9).reshape((1, 9))
        y = np.arange(55, 55 + 9).reshape((1, 9))
        s = mdtd.generate_4d_data(
            probe_size_x=9,
            probe_size_y=1,
            ring_x=None,
            image_size_x=x_im,
            image_size_y=y_im,
            disk_x=x,
            disk_y=y,
            disk_r=20,
            disk_I=20,
            blur=False,
            blur_sigma=1,
            downscale=False,
        )
        x_new, y_new = (x_im / 2) - x, (y_im / 2) - y
        s_bs = s.inav[:, 0].get_direct_beam_position(method="center_of_mass")
        assert (s_bs.isig[0].data == x_new).all()
        assert (s_bs.isig[1].data == y_new).all()

    def test_0d_signal(self):
        x_im, y_im = 120, 100
        x, y = 40, 51
        s = mdtd.generate_4d_data(
            probe_size_x=1,
            probe_size_y=1,
            ring_x=None,
            image_size_x=x_im,
            image_size_y=y_im,
            disk_x=x,
            disk_y=y,
            disk_r=20,
            disk_I=20,
            blur=False,
            blur_sigma=1,
            downscale=False,
        )
        x_new, y_new = (x_im / 2) - x, (y_im / 2) - y
        s_bs = s.inav[0, 0].get_direct_beam_position(method="center_of_mass")
        assert (s_bs.isig[0].data == x_new).all()
        assert (s_bs.isig[1].data == y_new).all()

    def test_lazy(self):
        x_im, y_im = 160, 140
        y, x = np.mgrid[75:83:9j, 85:95:11j]
        s = mdtd.generate_4d_data(
            probe_size_x=11,
            probe_size_y=9,
            ring_x=None,
            image_size_x=x_im,
            image_size_y=y_im,
            disk_x=x,
            disk_y=y,
            disk_r=40,
            disk_I=20,
            blur=True,
            blur_sigma=1,
            downscale=False,
        )
        x_new, y_new = (x_im / 2) - x, (y_im / 2) - y
        s_lazy = LazyDiffraction2D(da.from_array(s.data, chunks=(1, 1, 140, 160)))
        s_lazy_bs = s_lazy.get_direct_beam_position(method="center_of_mass")
        np.testing.assert_allclose(s_lazy_bs.isig[0].data, x_new)
        np.testing.assert_allclose(s_lazy_bs.isig[1].data, y_new)

        s_lazy_1d = s_lazy.inav[0]
        s_lazy_1d_bs = s_lazy_1d.get_direct_beam_position(method="center_of_mass")
        np.testing.assert_allclose(s_lazy_1d_bs.isig[0].data, x_new[:, 0])
        np.testing.assert_allclose(s_lazy_1d_bs.isig[1].data, y_new[:, 0])

        s_lazy_0d = s_lazy.inav[0, 0]
        s_lazy_0d_bs = s_lazy_0d.get_direct_beam_position(method="center_of_mass")
        np.testing.assert_allclose(s_lazy_0d_bs.isig[0].data, x_new[0, 0])
        np.testing.assert_allclose(s_lazy_0d_bs.isig[1].data, y_new[0, 0])

    def test_compare_lazy_and_nonlazy(self):
        x_im, y_im = 160, 140
        y, x = np.mgrid[75:83:9j, 85:95:11j]
        s = mdtd.generate_4d_data(
            probe_size_x=11,
            probe_size_y=9,
            ring_x=None,
            image_size_x=x_im,
            image_size_y=y_im,
            disk_x=x,
            disk_y=y,
            disk_r=40,
            disk_I=20,
            blur=True,
            blur_sigma=1,
            downscale=False,
        )
        s_lazy = LazyDiffraction2D(da.from_array(s.data, chunks=(1, 1, 140, 160)))
        s_bs = s.get_direct_beam_position(method="center_of_mass")
        s_lazy_bs = s_lazy.get_direct_beam_position(method="center_of_mass")
        np.testing.assert_equal(s_bs.data, s_lazy_bs.data)

        bs_nav_extent = s_bs.axes_manager.navigation_extent
        lazy_bs_nav_extent = s_lazy_bs.axes_manager.navigation_extent
        assert bs_nav_extent == lazy_bs_nav_extent

        bs_sig_extent = s_bs.axes_manager.signal_extent
        lazy_bs_sig_extent = s_lazy_bs.axes_manager.signal_extent
        assert bs_sig_extent == lazy_bs_sig_extent

    def test_lazy_result(self):
        data = da.ones((10, 10, 20, 20), chunks=(10, 10, 10, 10))
        s_lazy = LazyDiffraction2D(data)
        s_lazy_bs = s_lazy.get_direct_beam_position(
            method="center_of_mass", lazy_output=True
        )
        assert s_lazy_bs._lazy
        assert s_lazy_bs.axes_manager.navigation_shape == (10, 10)

        s_lazy_1d = s_lazy.inav[0]
        s_lazy_1d_bs = s_lazy_1d.get_direct_beam_position(
            method="center_of_mass", lazy_output=True
        )
        assert s_lazy_1d_bs._lazy
        assert s_lazy_1d_bs.axes_manager.navigation_shape == (10,)

        s_lazy_0d = s_lazy.inav[0, 0]
        s_lazy_0d_bs = s_lazy_0d.get_direct_beam_position(
            method="center_of_mass", lazy_output=True
        )
        assert s_lazy_0d_bs._lazy
        assert s_lazy_0d_bs.axes_manager.navigation_shape == ()

    def test_center_of_mass_inplace(self):
        with pytest.raises(ValueError):
            d = Diffraction2D(np.zeros((10, 10, 20, 20)))
            d.get_direct_beam_position(method="center_of_mass", inplace=True)


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
        s.center_direct_beam(method="blur", sigma=1, lazy_output=True, inplace=True)
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
        s_lazy.center_direct_beam(method="blur", sigma=1, lazy_output=False)
        assert s_lazy._lazy is False
        assert (s_lazy.data[:, :, 10, 8] == 9).all()
        s_lazy.data[:, :, 10, 8] = 0
        assert not s_lazy.data.any()

    def test_non_uniform_chunks(self):
        s_lazy = self.s_lazy
        s_lazy.data = s_lazy.data.rechunk((5, 4, 12, 14))
        s_lazy_shape = s_lazy.axes_manager.shape
        data_lazy_shape = s_lazy.data.shape
        s_lazy.center_direct_beam(method="blur", sigma=1, lazy_output=True)
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
        s_shifts = s.get_direct_beam_position(method="blur", sigma=1, lazy_output=False)
        s.center_direct_beam(shifts=s_shifts)
        assert (s.data[:, :, 10, 8] == 9).all()
        s.data[:, :, 10, 8] = 0
        assert not s.data.any()

    def test_shifts_input_lazy(self):
        s = self.s
        s_shifts = s.get_direct_beam_position(method="blur", sigma=1, lazy_output=True)
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
        s.axes_manager.signal_axes[0].offset = 100
        s.axes_manager.signal_axes[1].offset = 50
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

    def test_method_center_of_mass(self):
        s = self.s
        s.center_direct_beam(method="center_of_mass")

    def test_parameter_both_method_and_shifts(self):
        s = self.s
        with pytest.raises(ValueError):
            s.center_direct_beam(method="blur", sigma=1, shifts=np.ones((8, 6, 2)))

    def test_parameter_neither_method_and_shifts(self):
        s = self.s
        with pytest.raises(ValueError):
            s.center_direct_beam()


class TestFindVectors:
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

    def test_find_vectors(self):
        s = self.s
        vectors = s.get_diffraction_vectors()
        assert isinstance(vectors, DiffractionVectors)
        assert vectors.column_names == ["ky", "kx", "intensity"]
        assert vectors.units == ["px", "px", "a.u."]


class TestSubtractingDiffractionBackground:
    method1 = ["difference of gaussians", "median kernel", "radial median", "h-dome"]

    def setup_method(self):
        self.data = np.random.rand(3, 2, 20, 15)
        self.data[:, :, 10:12, 7:9] = 100
        self.dp = Diffraction2D(self.data)

    @pytest.mark.parametrize("methods", method1)
    def test_subtract_backgrounds(self, methods):
        # test that background is mostly removed
        if methods == "h-dome":
            kwargs = {"h": 0.25}
        else:
            kwargs = {}
        subtracted = self.dp.subtract_diffraction_background(method=methods, **kwargs)
        assert isinstance(subtracted, Diffraction2D)
        assert subtracted.data.shape == self.data.shape

    def test_exception_not_implemented_method(self):
        s = Diffraction2D(np.zeros((2, 2, 10, 10)))
        with pytest.raises(NotImplementedError):
            s.subtract_diffraction_background(method="magic")


class TestFindHotPixels:
    @pytest.fixture()
    def hot_pixel_data(self):
        """Values are 50, except [21, 11] and [5, 38]
        being 50000 (to represent a "hot pixel").
        """
        data = np.ones((2, 2, 40, 50)) * 50
        data[:, :, 21, 11] = 50000
        data[:, :, 5, 38] = 50000
        dask_array = da.from_array(data, chunks=(1, 1, 40, 50))
        return LazyDiffraction2D(dask_array)

    @pytest.mark.parametrize("lazy", (True, False))
    def test_find_hot_pixels(self, hot_pixel_data, lazy):
        if not lazy:
            hot_pixel_data.compute()
            assert not hot_pixel_data._lazy
        s_hot_pixels = hot_pixel_data.find_hot_pixels()
        assert s_hot_pixels._lazy == lazy
        assert s_hot_pixels.data[0, 0, 21, 11]
        assert s_hot_pixels.data[0, 0, 5, 38]
        s_hot_pixels.data[:, :, 21, 11] = False
        s_hot_pixels.data[:, :, 5, 38] = False
        assert not s_hot_pixels.data.any()

    def test_threshold_multiplier(self, hot_pixel_data):
        s_hot_pixels = hot_pixel_data.find_hot_pixels(threshold_multiplier=1000000)
        assert not s_hot_pixels.data.any()

    def test_mask_array(self, hot_pixel_data):
        mask_array = Diffraction2D(np.ones_like(hot_pixel_data.data, dtype=bool))
        s_hot_pixels = hot_pixel_data.find_hot_pixels(mask=mask_array)
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
        s_dead_pixels = dead_pixel_data_2d.find_dead_pixels()
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

    def test_dead_pixel_value(self, dead_pixel_data_2d):
        s_dead_pixels = dead_pixel_data_2d.find_dead_pixels(dead_pixel_value=-10)
        assert not s_dead_pixels.data.any()

    def test_mask_array(self, dead_pixel_data_2d):
        mask_array = np.ones_like(dead_pixel_data_2d.data, dtype=bool)
        s_dead_pixels = dead_pixel_data_2d.find_dead_pixels(mask=mask_array)
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
    def test_lazy_with_bad_pixel_finders(self, data, lazy_input, lazy_result):
        s = Diffraction2D(data).as_lazy()
        if not lazy_input:
            s.compute()

        hot = s.find_hot_pixels(lazy_output=True)
        dead = s.find_dead_pixels()
        bad = hot + dead

        s = s.correct_bad_pixels(bad, lazy_output=lazy_result, inplace=False)
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


class TestFilter:
    @pytest.fixture
    def three_section(self):
        x = np.random.random((100, 50, 20, 20))
        x[0:20, :, 5:7, 5:7] = x[0:20, :, 5:7, 5:7] + 10
        x[20:60, :, 1:3, 14:16] = x[20:60, :, 1:3, 14:16] + 10
        x[60:100, :, 6:8, 10:12] = x[60:100, :, 6:8, 10:12] + 10
        d = Diffraction2D(x)
        return d

    @pytest.mark.parametrize("lazy", [True, False])
    def test_filter(self, three_section, lazy):
        if lazy:  # pragma: no cover
            dask_image = pytest.importorskip("dask_image")
            from dask_image.ndfilters import gaussian_filter as gaussian_filter

            three_section = three_section.as_lazy()
        else:
            from scipy.ndimage import gaussian_filter

        sigma = (3, 3, 3, 3)
        new = three_section.filter(func=gaussian_filter, sigma=sigma, inplace=False)
        three_section.filter(func=gaussian_filter, sigma=sigma, inplace=True)
        np.testing.assert_array_almost_equal(new.data, three_section.data)

    def test_filter_fail(self, three_section):
        def small_func(x):
            return x[1:, 1:, 1:, 1:]

        with pytest.raises(ValueError):
            new = three_section.filter(func=small_func, inplace=False)


class TestBlockwise:
    def test_blockwise(self):
        s = Diffraction2D(np.ones((2, 2, 10, 10)))
        s._blockwise(lambda x: x + 1, inplace=True)
        assert np.all(s.data == 2)
        assert s.axes_manager.signal_shape == (10, 10)

    def test_blockwise_change_signal(self):
        s = Diffraction2D(np.ones((2, 2, 10, 10)))

        def change_signal(x):
            return np.zeros((2, 2, 30, 40))

        new_s = s._blockwise(change_signal, inplace=False, signal_shape=(30, 40))
        assert np.all(new_s.data == 0)
        assert new_s.data.shape == (2, 2, 30, 40)
        assert new_s.axes_manager.signal_shape == (40, 30)

    def test_blockwise_partial_ragged(self):
        s = Diffraction2D(np.ones((2, 2, 10, 10)))

        def change_signal(x):
            exp = np.empty(
                (
                    2,
                    2,
                ),
                dtype=object,
            )
            exp[0, 0] = np.zeros((3, 4))
            exp[0, 1] = np.zeros((4, 4))
            exp[1, 0] = np.zeros((3, 4))
            exp[1, 1] = np.zeros((4, 4))
            return exp

        new_s = s._blockwise(
            change_signal, inplace=False, signal_shape=(), ragged=True, dtype=object
        )
        assert new_s.data.shape == (2, 2)
        assert new_s.axes_manager.signal_shape == ()
        assert new_s.data[0, 0].shape == (3, 4)
