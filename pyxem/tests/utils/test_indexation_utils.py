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

from hyperspy.signals import Signal2D
import numpy as np
from orix.quaternion import Rotation
import pytest

from pyxem.utils.indexation_utils import match_vectors
from pyxem.utils.indexation_utils import (
    index_dataset_with_template_rotation,
    results_dict_to_crystal_map,
)


def test_match_vectors(vector_match_peaks, vector_library):
    # Wrap to test handling of ragged arrays
    peaks = np.empty(1, dtype="object")
    peaks[0] = vector_match_peaks
    matches, rhkls = match_vectors(
        peaks,
        vector_library,
        mag_tol=0.1,
        angle_tol=0.1,
        index_error_tol=0.3,
        n_peaks_to_index=2,
        n_best=1,
    )
    assert len(matches) == 1
    np.testing.assert_allclose(matches[0].match_rate, 1.0)
    np.testing.assert_allclose(matches[0].rotation_matrix, np.identity(3), atol=0.1)
    np.testing.assert_allclose(matches[0].total_error, 0.03, atol=0.01)

    np.testing.assert_allclose(rhkls[0][0], [1, 0, 0])
    np.testing.assert_allclose(rhkls[0][1], [0, 2, 0])
    np.testing.assert_allclose(rhkls[0][2], [1, 2, 3])


def test_match_vector_total_error_default(vector_match_peaks, vector_library):
    matches, rhkls = match_vectors(
        vector_match_peaks,
        vector_library,
        mag_tol=0.1,
        angle_tol=0.1,
        index_error_tol=0.0,
        n_peaks_to_index=2,
        n_best=5,
    )
    assert len(matches) == 5
    np.testing.assert_allclose(matches[0][2], 0.0)  # match rate
    np.testing.assert_allclose(matches[0][1], np.identity(3), atol=0.1)
    np.testing.assert_allclose(matches[0][4], 1.0)  # error mean

    assert len(rhkls) == 0


@pytest.mark.filterwarnings("ignore:Property 'correlation' was expected")
def test_results_dict_to_crystal_map(test_library_phases_multi, test_lib_gen):
    """Test getting a :class:`orix.crystal_map.CrystalMap` from returns
    from :func:`index_dataset_with_template_rotation`.
    """
    # Map and signal shapes
    nav_shape = (2, 3)
    sig_shape = (80, 80)
    test_set = np.zeros(nav_shape + sig_shape)

    # Diffraction conditions
    diff_lib = test_lib_gen.get_diffraction_library(
        test_library_phases_multi,
        calibration=0.015,
        reciprocal_radius=1.18,
        half_shape=tuple(np.array(sig_shape) // 2),
        with_direct_beam=False,
    )

    # Simulate multi-phase results
    phase_id = np.zeros(nav_shape, dtype=int)
    phase_id[:, [0, 2]] = 1
    phase_names = list(diff_lib.keys())

    # Simulate patterns
    # TODO: Remove version check after diffsims 0.5.0 is released
    from packaging.version import Version
    import diffsims
    from scipy.ndimage import rotate

    diffsims_version = Version(diffsims.__version__)
    if diffsims_version > Version("0.4.2"):
        sim_kwargs = dict(shape=sig_shape, sigma=4)
    else:  # pragma: no cover
        sim_kwargs = dict(size=sig_shape[0], sigma=4)
    for idx in np.ndindex(*nav_shape):
        i = phase_id[idx]
        j = int(idx[1] / 2)
        test_pattern = diff_lib[phase_names[i]]["simulations"][
            j
        ].get_diffraction_pattern(**sim_kwargs)
        test_pattern = np.flipud(test_pattern)
        test_set[idx] = test_pattern

    test_set[0, 0] = rotate(test_set[0, 0], 30, reshape=False)
    # Perform template matching
    n_best = 3
    results, phase_dict = index_dataset_with_template_rotation(
        Signal2D(test_set),
        diff_lib,
        phases=phase_names,
        n_best=n_best,
    )

    # Extract various results once
    phase_id = results["phase_index"].reshape((-1, n_best))
    ori = np.deg2rad(results["orientation"].reshape((-1, n_best, 3)))
    orientation_result = results["orientation"][0, 0, 0, 0]

    assert (
        np.remainder(abs(orientation_result), 15) < 1
        or np.remainder(abs(orientation_result), 30) < 1
    )

    # Property names in `results`
    prop_names = ["correlation", "mirrored_template", "template_index"]

    # Only get the bast match when multiple phases match best to some
    # patterns
    xmap = results_dict_to_crystal_map(
        results, phase_dict, diffraction_library=diff_lib
    )
    assert xmap.shape == nav_shape
    assert np.allclose(xmap.phase_id, phase_id[:, 0])
    assert xmap.rotations_per_point == 1
    assert xmap.phases.names == phase_names
    assert (
        xmap.phases.structures[0].lattice.abcABG()
        == test_library_phases_multi.structures[0].lattice.abcABG()
    )

    # Raise warning when a property is not present as expected
    del results[prop_names[0]]
    with pytest.warns(UserWarning, match=f"Property '{prop_names[0]}' was expected"):
        xmap2 = results_dict_to_crystal_map(
            results, phase_dict, diffraction_library=diff_lib
        )
    assert list(xmap2.prop.keys()) == prop_names[1:]

    # Raise error when trying to access a best match which isn't
    # available
    with pytest.raises(ValueError, match="`index` cannot be higher than 2"):
        _ = results_dict_to_crystal_map(results, phase_dict, index=3)
    # Get second best match
    i = 1
    xmap3 = results_dict_to_crystal_map(results, phase_dict, index=i)
    assert xmap3.rotations_per_point == 1
    assert np.allclose(xmap3.phase_id, phase_id[:, i])
    assert np.allclose(xmap3.rotations.data, Rotation.from_euler(ori[:, i]).data)
    assert np.allclose(
        xmap3.prop[prop_names[1]], results[prop_names[1]][:, :, i].flatten()
    )

    # Make map single-phase and get all matches per point
    results["phase_index"][..., 0] = 0
    xmap4 = results_dict_to_crystal_map(results, phase_dict)
    assert np.all(xmap4.phase_id == 0)
    assert xmap4.phases.names[0] == phase_names[0]
    assert xmap4.rotations_per_point == 3

    # Get only best match even though map is single-phase
    i = 0
    xmap5 = results_dict_to_crystal_map(results, phase_dict, index=i)
    assert xmap5.rotations_per_point == 1
    assert np.allclose(xmap5.rotations.data, Rotation.from_euler(ori[:, i]).data)
    assert np.allclose(
        xmap5.prop[prop_names[1]], results[prop_names[1]][:, :, i].flatten()
    )
