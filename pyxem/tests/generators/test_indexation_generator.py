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
import dask.array as da

from hyperspy._signals.signal2d import Signal2D
from diffsims.libraries.vector_library import DiffractionVectorLibrary
from diffsims.sims.diffraction_simulation import ProfileSimulation
from diffsims.generators.diffraction_generator import DiffractionGenerator
from diffsims.generators.library_generator import DiffractionLibraryGenerator
from diffsims.libraries.structure_library import StructureLibrary

from pyxem.generators import (
    IndexationGenerator,
    TemplateIndexationGenerator,
    ProfileIndexationGenerator,
    VectorIndexationGenerator,
    AcceleratedIndexationGenerator,
)
from pyxem.signals import (
    ElectronDiffraction2D,
    TemplateMatchingResults,
    DiffractionVectors,
)
from pyxem.utils.indexation_utils import OrientationResult
from unittest.mock import Mock
import sys

def generate_library(good_library):
    """Here we're testing the __init__ so we focus on 0 being the first entry of the orientations"""
    mock_sim_1 = Mock()
    mock_sim_1.calibrated_coordinates = np.array(
            [
                [0, 1, 0],
                [1, 0, 0],
            ]
        )
    mock_sim_1.intensities = np.array([2, 3,])
    simlist = [mock_sim_1, mock_sim_1]

    lead_number = 0 if good_library else 1
    orientations = np.array(
            [
                [0, 2, 3],
                [lead_number, 4, 5],
            ]
        )
    library = {}
    library["dummyphase"] = {"simulations": simlist, "orientations": orientations}
    return library

def test_old_indexer_routine():
    with pytest.raises(ValueError):
        _ = IndexationGenerator("a", "b")

@pytest.mark.skipif(sys.platform=='darwin',reason="Fails on Mac OSX")
@pytest.mark.slow
@pytest.mark.parametrize("good_library",[True,False])
def test_AcceleratedIndexationGenerator(good_library):
    signal = ElectronDiffraction2D((np.ones((2,2,256,256)))).as_lazy()
    library = generate_library(good_library=good_library)

    if good_library:
        acgen = AcceleratedIndexationGenerator(signal,library)
        d = acgen.correlate(n_largest=2)

    elif not good_library:
        with pytest.raises(ValueError):
            acgen = AcceleratedIndexationGenerator(signal,library)

    return None

@pytest.mark.parametrize(
    "method", ["fast_correlation", "zero_mean_normalized_correlation"]
)
def test_TemplateIndexationGenerator(default_structure, method):
    identifiers = ["a", "b"]
    structures = [default_structure, default_structure]
    orientations = [
        [(0, 0, 0), (0, 1, 0), (1, 0, 0)],
        [(0, 0, 1), (0, 0, 2), (0, 0, 3)],
    ]
    structure_library = StructureLibrary(identifiers, structures, orientations)
    libgen = DiffractionLibraryGenerator(DiffractionGenerator(300))
    library = libgen.get_diffraction_library(
        structure_library, 0.017, 0.02, (100, 100), False
    )

    edp = ElectronDiffraction2D(np.random.rand(2, 2, 200, 200))
    indexer = TemplateIndexationGenerator(edp, library)

    mask_signal = hs.signals.Signal2D(np.array([[1, 0], [1, 1]])).T
    z = indexer.correlate(method=method, n_largest=2, mask=mask_signal)
    assert isinstance(z, TemplateMatchingResults)
    assert isinstance(z.data, Signal2D)
    assert z.data.data.shape[0:2] == edp.data.shape[0:2]
    assert z.data.data.shape[3] == 5


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
