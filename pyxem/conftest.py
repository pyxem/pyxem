import os
from tempfile import TemporaryDirectory
import pytest
import numpy
import matplotlib.pyplot as plt
import hyperspy.api as hs
import pyxem as pxm


@pytest.fixture(autouse=True)
def doctest_setup_teardown(request):
    plt.ioff()
    tmp_dir = TemporaryDirectory()
    hs.preferences.General.show_progressbar = False
    org_dir = os.getcwd()
    os.chdir(tmp_dir.name)
    yield
    os.chdir(org_dir)
    tmp_dir.cleanup()
    plt.close("all")


@pytest.fixture(autouse=True)
def add_np_am(doctest_namespace):
    doctest_namespace["np"] = numpy
    doctest_namespace["hs"] = hs
    doctest_namespace["pxm"] = pxm
