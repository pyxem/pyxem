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

# Use agg backend to avoid displaying figure when running tests
import matplotlib

matplotlib.use("agg")

import pytest
import diffpy.structure
import numpy as np
from transforms3d.euler import euler2mat

from pyxem.signals.diffraction2d import Diffraction2D
from pyxem.signals.electron_diffraction2d import ElectronDiffraction2D
from pyxem.signals.electron_diffraction1d import ElectronDiffraction1D
from diffsims.libraries.vector_library import DiffractionVectorLibrary

from pyxem.utils.indexation_utils import OrientationResult


# a straight lift from 
# https://docs.pytest.org/en/latest/example/simple.html#control-skipping-of-tests-according-to-command-line-option--

def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)

@pytest.fixture
def default_structure():
    """An atomic structure represented using diffpy
    """
    latt = diffpy.structure.lattice.Lattice(3, 3, 5, 90, 90, 120)
    atom = diffpy.structure.atom.Atom(atype="Ni", xyz=[0, 0, 0], lattice=latt)
    hexagonal_structure = diffpy.structure.Structure(atoms=[atom], lattice=latt)
    return hexagonal_structure


@pytest.fixture
def z():
    return np.array(
        [
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 2.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 2.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 2.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
        ]
    ).reshape(2, 2, 8, 8)


@pytest.fixture
def diffraction_pattern(z):
    """A simple, multiuse diffraction pattern, with dimensions:
    ElectronDiffraction2D <2,2|8,8>
    """
    dp = ElectronDiffraction2D(z)
    dp.metadata.Signal.found_from = "conftest"  # dummy metadata
    return dp


@pytest.fixture(
    params=[
        np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
    ]
)
def dp_single(request):
    """
    1D (in navigation space) diffraction pattern <1|8,8>
    """
    return ElectronDiffraction2D(request.param)


@pytest.fixture
def electron_diffraction1d():
    """A simple, multiuse diffraction profile, with dimensions:
    ElectronDiffraction1D <2,2|12>
    """
    data = np.array(
        [
            [[1.0, 0.25, 0.0, 0.0, 0.0], [1.0, 0.25, 0.0, 0.0, 0.0]],
            [[1.0, 0.25, 0.0, 0.0, 0.16666667], [1.5, 0.5, 0.0, 0.0, 0.0]],
        ]
    )

    return ElectronDiffraction1D(data)


@pytest.fixture
def vector_match_peaks():
    return np.array([[1, 0.1, 0], [0, 2, 0], [1, 2, 3],])


@pytest.fixture
def vector_library():
    library = DiffractionVectorLibrary()
    library["A"] = {
        "indices": np.array(
            [[[0, 2, 0], [1, 0, 0]], [[1, 2, 3], [0, 2, 0]], [[1, 2, 3], [1, 0, 0]],]
        ),
        "measurements": np.array(
            [
                [2, 1, np.pi / 2],
                [np.sqrt(14), 2, 1.006853685],
                [np.sqrt(14), 1, 1.300246564],
            ]
        ),
    }
    lattice = diffpy.structure.Lattice(1, 1, 1, 90, 90, 90)
    library.structures = [diffpy.structure.Structure(lattice=lattice)]
    return library


@pytest.fixture
def sp_template_match_result():
    row_1 = np.array([0, np.array([2, 3, 4]), 0.7], dtype="object")
    row_2 = np.array([0, np.array([2, 3, 5]), 0.6], dtype="object")
    # note we require (correlation of row_1 > correlation row_2)
    return np.vstack((row_1, row_2))


@pytest.fixture
def dp_template_match_result():
    row_1 = np.array([0, np.array([2, 3, 4]), 0.7], dtype="object")
    row_2 = np.array([0, np.array([2, 3, 5]), 0.8], dtype="object")
    row_3 = np.array([1, np.array([2, 3, 4]), 0.5], dtype="object")
    row_4 = np.array([1, np.array([2, 3, 5]), 0.3], dtype="object")
    return np.vstack((row_1, row_2, row_3, row_4))


@pytest.fixture
def sp_vector_match_result():
    # We require (total_error of row_1 > correlation row_2)
    res = np.empty(2, dtype="object")
    res[0] = OrientationResult(
        0,
        euler2mat(*np.deg2rad([0, 0, 90]), "rzxz"),
        0.5,
        np.array([0.1, 0.05, 0.2]),
        0.1,
        1.0,
        0,
        0,
    )
    res[1] = OrientationResult(
        0,
        euler2mat(*np.deg2rad([0, 0, 90]), "rzxz"),
        0.6,
        np.array([0.1, 0.10, 0.2]),
        0.2,
        1.0,
        0,
        0,
    )
    return res


@pytest.fixture
def dp_vector_match_result():
    res = np.empty(4, dtype="object")
    res[0] = OrientationResult(
        0,
        euler2mat(*np.deg2rad([90, 0, 0]), "rzxz"),
        0.6,
        np.array([0.1, 0.10, 0.2]),
        0.3,
        1.0,
        0,
        0,
    )
    res[1] = OrientationResult(
        0,
        euler2mat(*np.deg2rad([0, 10, 20]), "rzxz"),
        0.5,
        np.array([0.1, 0.05, 0.2]),
        0.4,
        1.0,
        0,
        0,
    )
    res[2] = OrientationResult(
        1,
        euler2mat(*np.deg2rad([0, 45, 45]), "rzxz"),
        0.8,
        np.array([0.1, 0.30, 0.2]),
        0.1,
        1.0,
        0,
        0,
    )
    res[3] = OrientationResult(
        1,
        euler2mat(*np.deg2rad([0, 0, 90]), "rzxz"),
        0.7,
        np.array([0.1, 0.05, 0.1]),
        0.2,
        1.0,
        0,
        0,
    )
    return res
