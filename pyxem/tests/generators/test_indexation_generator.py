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

from pyxem.generators.indexation_generator import ProfileIndexationGenerator
from pyxem.generators.indexation_generator import VectorIndexationGenerator
from pyxem.generators.indexation_generator import (
    get_fourier_transform,
    get_library_FT_dict,
)

from diffsims.libraries.vector_library import DiffractionVectorLibrary
from diffsims.libraries.diffraction_library import DiffractionLibrary
from diffsims.sims.diffraction_simulation import ProfileSimulation
from pyxem.signals.diffraction_vectors import DiffractionVectors

from pyxem.utils.indexation_utils import OrientationResult


@pytest.fixture
def profile_simulation():
    return ProfileSimulation(
        magnitudes=[
            0.31891931643691351,
            0.52079306292509475,
            0.6106839974876449,
            0.73651261277849378,
            0.80259601243613932,
            0.9020400452156796,
            0.95675794931074043,
            1.0415861258501895,
            1.0893168446141808,
            1.1645286909108374,
            1.2074090451670043,
            1.2756772657476541,
        ],
        intensities=np.array(
            [
                100.0,
                99.34619104,
                64.1846346,
                18.57137199,
                28.84307971,
                41.31084268,
                23.42104951,
                13.996264,
                24.87559364,
                20.85636003,
                9.46737774,
                5.43222307,
            ]
        ),
        hkls=[
            {(1, 1, 1): 8},
            {(2, 2, 0): 12},
            {(3, 1, 1): 24},
            {(4, 0, 0): 6},
            {(3, 3, 1): 24},
            {(4, 2, 2): 24},
            {(3, 3, 3): 8, (5, 1, 1): 24},
            {(4, 4, 0): 12},
            {(5, 3, 1): 48},
            {(6, 2, 0): 24},
            {(5, 3, 3): 24},
            {(4, 4, 4): 8},
        ],
    )


def test_profile_indexation_generator_init(profile_simulation):
    pig = ProfileIndexationGenerator(
        magnitudes=[
            0.31891931643691351,
            0.52079306292509475,
            0.6106839974876449,
            0.73651261277849378,
            0.80259601243613932,
            0.9020400452156796,
            0.95675794931074043,
            1.0415861258501895,
            1.0893168446141808,
            1.1645286909108374,
            1.2074090451670043,
            1.2756772657476541,
        ],
        simulation=profile_simulation,
    )
    assert isinstance(pig, ProfileIndexationGenerator)


def test_profile_indexation_generator_single_indexation(profile_simulation):
    pig = ProfileIndexationGenerator(
        magnitudes=[
            0.31891931643691351,
            0.52079306292509475,
            0.6106839974876449,
            0.73651261277849378,
            0.80259601243613932,
            0.9020400452156796,
            0.95675794931074043,
            1.0415861258501895,
            1.0893168446141808,
            1.1645286909108374,
            1.2074090451670043,
            1.2756772657476541,
        ],
        simulation=profile_simulation,
    )
    indexation = pig.index_peaks(tolerance=0.02)
    np.testing.assert_almost_equal(indexation[0][0], 0.3189193164369)


def test_get_fourier_transform():
    shape = (3, 3)
    fsize = (5, 5)
    normalization_constant = 0.9278426705718053  # Precomputed normalization. Formula full_frame(template, 1, template, 1)
    template_coordinates = np.asarray([[1, 1]])
    template_intensities = np.asarray([1])
    transform, norm = get_fourier_transform(
        template_coordinates, template_intensities, shape, fsize
    )
    test_value = np.real(transform[2, 1])  # Center value
    np.testing.assert_approx_equal(test_value, 1)
    np.testing.assert_approx_equal(norm, normalization_constant)


def test_get_library_FT_dict():
    new_template_library = DiffractionLibrary()
    new_template_library["GaSb"] = {
        "orientations": np.array([[0.0, 0.0, 0.0],]),
        "pixel_coords": np.array([np.asarray([[1, 1],])]),
        "intensities": np.array([np.array([1,])]),
    }
    shape = (3, 3)
    fsize = (5, 5)
    normalization_constant = 0.9278426705718053
    new_template_dict = get_library_FT_dict(new_template_library, shape, fsize)
    for phase_index, library_entry in enumerate(new_template_dict.values()):
        orientations = library_entry["orientations"]
        patterns = library_entry["patterns"]
        pattern_norms = library_entry["pattern_norms"]
    np.testing.assert_approx_equal(orientations[0][0], 0.0)
    np.testing.assert_approx_equal(np.real(patterns[0][2, 1]), 1)
    np.testing.assert_approx_equal(pattern_norms[0], normalization_constant)


def test_vector_indexation_generator_init():
    vectors = DiffractionVectors([[1], [2]])
    vectors.cartesian = [[1], [2]]
    vector_library = DiffractionVectorLibrary()
    vector_indexation_generator = VectorIndexationGenerator(vectors, vector_library)
    assert isinstance(vector_indexation_generator, VectorIndexationGenerator)
    assert vector_indexation_generator.vectors == vectors
    assert vector_indexation_generator.library == vector_library


def test_vector_indexation_generator_cartesian_check():
    vectors = DiffractionVectors([[1], [2]])
    vector_library = DiffractionVectorLibrary()

    with pytest.raises(
        ValueError,
        match="Cartesian coordinates are required in order to index diffraction vectors",
    ):
        vector_indexation_generator = VectorIndexationGenerator(vectors, vector_library)


def test_vector_indexation_generator_index_vectors(vector_match_peaks, vector_library):
    # vectors not used directly
    vectors = DiffractionVectors(np.array(vector_match_peaks[:, :2]))
    vectors.cartesian = DiffractionVectors(np.array(vector_match_peaks))
    gen = VectorIndexationGenerator(vectors, vector_library)
    indexation = gen.index_vectors(
        mag_tol=0.1, angle_tol=6, index_error_tol=0.3, n_peaks_to_index=2, n_best=5
    )

    # Values are tested directly on the match_vector in the util tests
    assert isinstance(indexation.vectors, DiffractionVectors)

    # (n_best=1, 5 result values from each)
    np.testing.assert_equal(indexation.data.shape, (5,))

    # n_best=1, 3 peaks with hkl)
    np.testing.assert_equal(indexation.hkls.shape, (1, 3, 3))

    refined1 = gen.refine_n_best_orientations(indexation, 1.0, 1.0, n_best=0)

    assert isinstance(refined1.vectors, DiffractionVectors)
    np.testing.assert_equal(refined1.data.shape, (5,))

    refined2 = gen.refine_best_orientation(indexation, 1.0, 1.0)

    assert isinstance(refined2.vectors, DiffractionVectors)
    np.testing.assert_equal(refined2.data.shape, (1,))
    assert isinstance(refined2.data[0], OrientationResult)

    assert refined2.data[0].phase_index == indexation.data[0].phase_index
    assert refined2.data[0].match_rate == indexation.data[0].match_rate

    # Must use a large tolerance here, because there are only 3 vectors
    np.testing.assert_almost_equal(
        np.diag(refined1.data[0].rotation_matrix),
        np.diag(indexation.data[0].rotation_matrix),
        1,
    )
    np.testing.assert_almost_equal(
        np.diag(refined2.data[0].rotation_matrix),
        np.diag(indexation.data[0].rotation_matrix),
        1,
    )
