import pytest
import diffpy.structure

@pytest.fixture(scope="session",autouse=True)
def default_structure():
    latt = diffpy.structure.lattice.Lattice(3,3,5,90,90,120)
    atom = diffpy.structure.atom.Atom(atype='Ni',xyz=[0,0,0],lattice=latt)
    hexagonal_structure = diffpy.structure.Structure(atoms=[atom],lattice=latt)
    return hexagonal_structure
