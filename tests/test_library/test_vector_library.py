import pytest
import pyxem as pxm
import os
import numpy as np
import diffpy.structure

from pyxem.libraries.vector_library import load_VectorLibrary
from pyxem.libraries.structure_library import StructureLibrary


@pytest.fixture
def get_library(default_structure):
    structure_library = StructureLibrary(['Phase'], [default_structure], [[(0, 0, 0), (0, 0.2, 0)]])
    vlg = pxm.VectorLibraryGenerator(structure_library)
    return vlg.get_vector_library(0.7)


def test_library_io(get_library):
    get_library.pickle_library('file_01.pickle')
    loaded_library = load_VectorLibrary('file_01.pickle', safety=True)
    os.remove('file_01.pickle')
    # we can't check that the entire libraries are the same as the memory
    # location of the 'Sim' changes
    assert np.allclose(get_library['Phase'][0][0], loaded_library['Phase'][0][0])


@pytest.mark.xfail(raises=RuntimeError)
def test_unsafe_loading(get_library):
    get_library.pickle_library('file_01.pickle')
    loaded_library = load_VectorLibrary('file_01.pickle')
