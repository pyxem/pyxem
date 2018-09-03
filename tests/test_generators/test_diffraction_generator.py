# -*- coding: utf-8 -*-
# Copyright 2017-2018 The pyXem developers
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

import numpy as np
import pytest
from pyxem.signals.diffraction_simulation import DiffractionSimulation, ProfileSimulation
from pyxem.generators.diffraction_generator import DiffractionGenerator
import diffpy.structure


@pytest.fixture(params=[
    (300, 0.02, None),
])
def diffraction_calculator(request):
    return DiffractionGenerator(*request.param)

@pytest.fixture()
def structure(lattice_parameter=None):
    """
    We construct an Fd-3m silicon (with lattice parameter 5.431 as a default)
    """
    if lattice_parameter is not None:
        a = lattice_parameter
    else:
        a = 5.431
    latt = diffpy.structure.lattice.Lattice(a,a,a,90,90,90)
    #TODO - Make this construction with internal diffpy syntax
    atom_list = []
    for coords in [[0,0,0],[0.5,0,0.5],[0,0.5,0.5],[0.5,0.5,0]]:
        x,y,z = coords[0],coords[1],coords[2]
        atom_list.append(diffpy.structure.atom.Atom(atype='Si',xyz=[x,y,z],lattice=latt)) # Motif part A
        atom_list.append(diffpy.structure.atom.Atom(atype='Si',xyz=[x+0.25,y+0.25,z+0.25],lattice=latt)) # Motif part B
    return diffpy.structure.Structure(atoms=atom_list,lattice=latt)


@pytest.fixture(params=[{}])
def diffraction_simulation(request):
    return DiffractionSimulation(**request.param)


class TestDiffractionCalculator:

    def test_init(self, diffraction_calculator: DiffractionGenerator):
        assert diffraction_calculator.debye_waller_factors == {}

    def test_matching_results(self, diffraction_calculator, structure):
        diffraction = diffraction_calculator.calculate_ed_data(structure, reciprocal_radius=5.)
        assert len(diffraction.indices) == len(diffraction.coordinates)
        assert len(diffraction.coordinates) == len(diffraction.intensities)


    def test_appropriate_scaling(self, diffraction_calculator: DiffractionGenerator):
        """Tests that doubling the unit cell halves the pattern spacing."""
        silicon = structure(5)
        big_silicon = structure(10)
        diffraction = diffraction_calculator.calculate_ed_data(structure=silicon, reciprocal_radius=5.)
        big_diffraction = diffraction_calculator.calculate_ed_data(structure=big_silicon, reciprocal_radius=5.)
        indices = [tuple(i) for i in diffraction.indices]
        big_indices = [tuple(i) for i in big_diffraction.indices]
        assert (2, 2, 0) in indices
        assert (2, 2, 0) in big_indices
        coordinates = diffraction.coordinates[indices.index((2, 2, 0))]
        big_coordinates = big_diffraction.coordinates[big_indices.index((2, 2, 0))]
        assert np.allclose(coordinates, big_coordinates * 2)

    def test_appropriate_intensities(self, diffraction_calculator, structure):
        """Tests the central beam is strongest."""
        diffraction = diffraction_calculator.calculate_ed_data(structure=structure, reciprocal_radius=5.)
        indices = [tuple(i) for i in diffraction.indices]
        central_beam = indices.index((0, 0, 0))
        smaller = np.greater_equal(diffraction.intensities[central_beam], diffraction.intensities)
        assert np.all(smaller)

    @pytest.mark.skip(reason="This can't be done as yet with diffpy")
    @pytest.mark.parametrize('structure, expected', [
        ('Fd-3m', (2, 2, 0)),
        ('Im-3m', (1, 1, 0)),
        ('Pm-3m', (1, 0, 0)),
        ('Pm-3m', (1, 1, 0)),
        ('Pm-3m', (2, 1, 0)),
    ], indirect=['structure'])
    def test_correct_peaks(self, diffraction_calculator, structure, expected):
        "Tests appropriate reflections are produced for space groups."
        diffraction = diffraction_calculator.calculate_ed_data(structure=structure, reciprocal_radius=5.)
        indices = [tuple(i) for i in diffraction.indices]
        assert expected in indices

    @pytest.mark.skip(reason="This can't be done as yet with diffpy")
    @pytest.mark.parametrize('structure, expected_extinction', [
        ('Fd-3m', (2, 1, 0)),
        ('Im-3m', (2, 1, 0)),
        ('Fd-3m', (1, 1, 0)),
        ('Im-3m', (1, 0, 0))
    ], indirect=['structure'])
    def test_correct_extinction(self, diffraction_calculator, structure, expected_extinction):
        """Tests appropriate extinctions are produced for space groups."""
        diffraction = diffraction_calculator.calculate_ed_data(structure=structure, reciprocal_radius=5.)
        indices = [tuple(i) for i in diffraction.indices]
        assert expected_extinction not in indices

    def test_calculate_profile_class(self, structure, diffraction_calculator):
        # tests the non-hexagonal (cubic) case
        profile = diffraction_calculator.calculate_profile_data(structure=structure,
                                                                reciprocal_radius=1.)
        assert isinstance(profile, ProfileSimulation)

        latt = diffpy.structure.lattice.Lattice(3,3,5,90,90,120)
        atom = diffpy.structure.atom.Atom(atype='Ni',xyz=[0,0,0],lattice=latt)
        hexagonal_structure = diffpy.structure.Structure(atoms=[atom],lattice=latt)
        hexagonal_profile = diffraction_calculator.calculate_profile_data(structure=hexagonal_structure,
                                                                reciprocal_radius=1.)
        assert isinstance(hexagonal_profile, ProfileSimulation)

@pytest.mark.skip(reason="Not to do with the generation of the simulation")
class TestDiffractionSimulation:

    def test_init(self):
        diffraction_simulation = DiffractionSimulation()
        assert diffraction_simulation.coordinates is None
        assert diffraction_simulation.indices is None
        assert diffraction_simulation.intensities is None
        assert diffraction_simulation.calibration == (1., 1.)

    @pytest.mark.parametrize('calibration, expected', [
        (5., (5., 5.)),
        (2, (2., 2.)),
        pytest.param(0, (0, 0), marks=pytest.mark.xfail(raises=ValueError)),
        pytest.param((0, 0), (0, 0), marks=pytest.mark.xfail(raises=ValueError)),
        ((1.5, 1.5), (1.5, 1.5)),
        ((1.3, 1.5), (1.3, 1.5)),
    ])
    def test_calibration(
            self,
            diffraction_simulation: DiffractionSimulation,
            calibration, expected
    ):
        diffraction_simulation.calibration = calibration
        assert diffraction_simulation.calibration == expected

    @pytest.mark.parametrize('coordinates, with_direct_beam, expected', [
        (
            np.array([[-1, 0, 0], [0, 0, 0], [1, 0, 0]]),
            False,
            np.array([True, False, True])
        ),
        (
            np.array([[-1, 0, 0], [0, 0, 0], [1, 0, 0]]),
            True,
            np.array([True, True, True])
        ),
        (
            np.array([[-1, 0, 0], [1, 0, 0]]),
            False,
            np.array([True, True])
        ),
    ])
    def test_direct_beam_mask(
            self,
            diffraction_simulation: DiffractionSimulation,
            coordinates, with_direct_beam, expected
    ):
        diffraction_simulation.coordinates = coordinates
        diffraction_simulation.with_direct_beam = with_direct_beam
        mask = diffraction_simulation.direct_beam_mask
        assert np.all(mask == expected)

    @pytest.mark.parametrize('coordinates, calibration, offset, expected', [
        (
            np.array([[1., 0., 0.], [1., 2., 0.]]),
            1., (0., 0.),
            np.array([[1., 0., 0.], [1., 2., 0.]]),
        ),
        (
            np.array([[1., 0., 0.], [1., 2., 0.]]),
            1., (1., 1.),
            np.array([[2, 1, 0], [2, 3, 0]]),
        ),
        (
            np.array([[1., 0., 0.], [1., 2., 0.]]),
            0.25, (0., 0.),
            np.array([[4., 0., 0.], [4., 8., 0.]]),
        ),
        (
            np.array([[1., 0., 0.], [1., 2., 0.]]),
            (0.5, 0.25), (0., 0.),
            np.array([[2., 0., 0.], [2., 8., 0.]]),
        ),
        (
            np.array([[1., 0., 0.], [1., 2., 0.]]),
            0.5, (1., 0.),
            np.array([[4., 0., 0.], [4., 4., 0.]]),
        )
    ])
    def test_calibrated_coordinates(
            self,
            diffraction_simulation: DiffractionSimulation,
            coordinates, calibration, offset, expected
    ):
        diffraction_simulation.coordinates = coordinates
        diffraction_simulation.calibration = calibration
        diffraction_simulation.offset = offset
        assert np.allclose(diffraction_simulation.calibrated_coordinates, expected)
