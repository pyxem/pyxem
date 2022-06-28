# -*- coding: utf-8 -*-
# Copyright 2016-2022 The pyXem developers
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
import hyperspy.api as hs

from diffsims.libraries.vector_library import DiffractionVectorLibrary
from diffsims.generators.diffraction_generator import DiffractionGenerator
from diffsims.generators.library_generator import DiffractionLibraryGenerator
from diffsims.libraries.structure_library import StructureLibrary

from pyxem.signals import ElectronDiffraction1D, ElectronDiffraction2D

# a straight lift from
# https://docs.pytest.org/en/latest/example/simple.html#control-skipping-of-tests-according-to-command-line-option--
# This means we don't always run the slowest tests


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
    else:  # pragma: no cover
        skip_slow = pytest.mark.skip(reason="need --runslow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)


# End of the code lift, it's regular code from here on out


@pytest.fixture
def default_structure():
    """An atomic structure represented using diffpy"""
    latt = diffpy.structure.lattice.Lattice(3, 3, 5, 90, 90, 120)
    atom = diffpy.structure.atom.Atom(atype="Ni", xyz=[0, 0, 0], lattice=latt)
    hexagonal_structure = diffpy.structure.Structure(atoms=[atom], lattice=latt)
    return hexagonal_structure


@pytest.fixture
def z():
    z = np.array(
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
    return z


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
    return np.array(
        [
            [1, 0.1, 0],
            [0, 2, 0],
            [1, 2, 3],
        ]
    )


@pytest.fixture
def vector_library():
    library = DiffractionVectorLibrary()
    library["A"] = {
        "indices": np.array(
            [
                [[0, 2, 0], [1, 0, 0]],
                [[1, 2, 3], [0, 2, 0]],
                [[1, 2, 3], [1, 0, 0]],
            ]
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
def test_patterns():
    patterns = hs.signals.Signal2D(np.zeros((3, 1, 128, 128)))
    _001_indexs = (
        np.array([31, 31, 31, 64, 64, 97, 97, 97]),
        np.array([31, 64, 97, 31, 97, 31, 64, 97]),
    )
    _101_indexs = (
        np.array([17, 17, 17, 64, 64, 111, 111, 111]),
        np.array([31, 64, 97, 31, 97, 31, 64, 97]),
    )
    _111_indexs = (
        np.array([23, 23, 64, 64, 105, 105]),
        np.array([40, 88, 17, 111, 40, 88]),
    )
    for i in range(8):
        patterns.inav[0, 0].isig[_001_indexs[1][i], _001_indexs[0][i]] = 24.0
        patterns.inav[0, 1].isig[_101_indexs[1][i], _101_indexs[0][i]] = 24.0
    for i in range(6):
        patterns.inav[0, 2].isig[_111_indexs[1][i], _111_indexs[0][i]] = 24.0
    return patterns


@pytest.fixture
def test_lib_gen():
    diff_gen = DiffractionGenerator(
        accelerating_voltage=200,
        precession_angle=1,
        scattering_params="lobato",
        shape_factor_model="linear",
        minimum_intensity=0.05,
    )
    lib_gen = DiffractionLibraryGenerator(diff_gen)
    return lib_gen


@pytest.fixture
def test_library_phases():
    latt = diffpy.structure.lattice.Lattice(3, 3, 3, 90, 90, 90)
    atom = diffpy.structure.atom.Atom(atype="Ni", xyz=[0, 0, 0], lattice=latt)
    structure = diffpy.structure.Structure(atoms=[atom], lattice=latt)
    library_phases_test = StructureLibrary(
        ["Test"],
        [structure],
        [np.array([(0, 0, 90), (0, 44, 90), (0, 54.735, 45)])],
    )
    return library_phases_test


@pytest.fixture
def test_library_phases_multi():
    ni_structure = diffpy.structure.Structure(
        atoms=[diffpy.structure.atom.Atom(atype="Ni", xyz=[0, 0, 0])],
        lattice=diffpy.structure.lattice.Lattice(3, 3, 3, 90, 90, 90)
    )
    al_structure = diffpy.structure.Structure(
        atoms=[diffpy.structure.atom.Atom(atype="Al", xyz=[0, 0, 0])],
        lattice=diffpy.structure.lattice.Lattice(4, 4, 4, 90, 90, 90)
    )
    library_phases_test = StructureLibrary(
        ["Ni", "Al"],
        [ni_structure, al_structure],
        [
            np.array([(0, 0, 90), (0, 44, 90), (0, 54.735, 45)]),
            np.array([(45, 45, 90), (45, 90, 45), (90, 45, 0)]),
        ],
    )
    return library_phases_test

