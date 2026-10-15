# -*- coding: utf-8 -*-
# Copyright 2016-2025 The pyXem developers
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

"""template_match_disk / _ring / template_match with ``device=``: the batched
torch correlation must reproduce the scikit-image result."""

import dask.array as da
import numpy as np
import pytest
from skimage import morphology
from skimage.feature import match_template

from pyxem import TORCH_INSTALLED
from pyxem.signals import Diffraction2D, LazyDiffraction2D

pytestmark = pytest.mark.skipif(not TORCH_INSTALLED, reason="torch is required")

#: Agreement with scikit-image: the two compute the same sums in a different
#: order, so they agree to rounding of the floating-point type.
TOLERANCE = {np.dtype(np.float32): 1e-4, np.dtype(np.float64): 1e-10}


def devices():
    import torch

    found = ["cpu"]
    if torch.cuda.is_available():
        found.append("cuda")
    return found


def planted_disks(dtype=np.float32, navigation_shape=(3, 4), size=64, seed=0):
    rng = np.random.default_rng(seed)
    rows, columns = np.mgrid[:size, :size]
    data = np.empty(navigation_shape + (size, size), dtype)
    for position in np.ndindex(navigation_shape):
        pattern = np.full((size, size), 5.0)
        for row in range(12, size - 8, 16):
            for column in range(12, size - 8, 16):
                center = (row + 0.3 * position[0], column + 0.2 * position[-1])
                pattern += 300 * (
                    (rows - center[0]) ** 2 + (columns - center[1]) ** 2 <= 16
                )
        data[position] = rng.poisson(pattern)
    return Diffraction2D(data)


@pytest.mark.parametrize("device", devices())
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.uint16])
def test_match_template_stack_matches_skimage(device, dtype):
    from pyxem.utils._template_matching import float_type, match_template_stack

    stack = planted_disks(dtype).data.reshape(-1, 64, 64)
    template = morphology.disk(4, stack.dtype)
    for pad_input in (True, False):
        expected = np.stack(
            [
                match_template(pattern, template, pad_input=pad_input)
                for pattern in stack
            ]
        )
        found = match_template_stack(
            stack, template, pad_input=pad_input, device=device
        )
        assert found.dtype == expected.dtype == float_type(dtype)
        np.testing.assert_allclose(found, expected, atol=TOLERANCE[found.dtype], rtol=0)


@pytest.mark.parametrize("mode", ["reflect", "edge"])
def test_padding_modes(mode):
    from pyxem.utils._template_matching import match_template_stack

    stack = planted_disks(np.float64).data.reshape(-1, 64, 64)
    template = morphology.disk(3, np.float64)
    expected = np.stack(
        [match_template(p, template, pad_input=True, mode=mode) for p in stack]
    )
    found = match_template_stack(stack, template, mode=mode, device="cpu")
    np.testing.assert_allclose(found, expected, atol=1e-10, rtol=0)


class TestSignalMethods:
    @pytest.mark.parametrize("device", devices())
    @pytest.mark.parametrize(
        "method, arguments",
        [
            ("template_match_disk", {"disk_r": 4}),
            ("template_match_disk", {"disk_r": 4, "subtract_min": False}),
            ("template_match_ring", {"r_inner": 3, "r_outer": 5}),
            ("template_match", {"template": np.ones((5, 5), np.float32)}),
        ],
    )
    def test_matches_scikit_image(self, method, arguments, device):
        signal = planted_disks()
        expected = getattr(signal, method)(**arguments)
        found = getattr(signal, method)(device=device, **arguments)
        assert type(found) is type(expected)
        assert found.data.dtype == expected.data.dtype
        np.testing.assert_allclose(found.data, expected.data, atol=1e-4, rtol=0)
        assert found.axes_manager.shape == expected.axes_manager.shape

    def test_same_peaks_as_scikit_image(self):
        """Peaks found on the two correlation maps are identical."""
        signal = planted_disks(navigation_shape=(4, 4))
        settings = dict(threshold_abs=0.5, min_distance=5)
        expected = signal.template_match_disk(disk_r=4, subtract_min=False)
        found = signal.template_match_disk(disk_r=4, subtract_min=False, device="cpu")
        expected_vectors = expected.get_diffraction_vectors(**settings)
        found_vectors = found.get_diffraction_vectors(**settings)
        for position in np.ndindex(4, 4):
            np.testing.assert_array_equal(
                found_vectors.data[position][:, :2],
                expected_vectors.data[position][:, :2],
            )

    def test_lazy(self):
        signal = planted_disks()
        lazy = LazyDiffraction2D(da.from_array(signal.data, chunks=(2, 2, -1, -1)))
        found = lazy.template_match_disk(disk_r=4, device="cpu")
        assert isinstance(found, LazyDiffraction2D)
        found.compute()
        np.testing.assert_allclose(
            found.data, signal.template_match_disk(disk_r=4).data, atol=1e-4, rtol=0
        )
        computed = lazy.template_match_disk(disk_r=4, device="cpu", lazy_output=False)
        assert not computed._lazy

    def test_inplace(self):
        signal = planted_disks()
        expected = signal.template_match_disk(disk_r=4)
        assert signal.template_match_disk(disk_r=4, device="cpu", inplace=True) is None
        np.testing.assert_allclose(signal.data, expected.data, atol=1e-4, rtol=0)

    def test_metadata_is_kept(self):
        signal = planted_disks()
        signal.metadata.General.title = "scan"
        signal.axes_manager.signal_axes[0].scale = 0.1
        found = signal.template_match_disk(disk_r=4, device="cpu")
        assert found.metadata.General.title == "scan"
        assert found.axes_manager.signal_axes[0].scale == 0.1

    def test_unsupported_arguments(self):
        with pytest.raises(TypeError, match="circular_background"):
            planted_disks().template_match_disk(
                disk_r=4, device="cpu", circular_background=True
            )
