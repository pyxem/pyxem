import os
from tempfile import TemporaryDirectory
import pytest
import numpy
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import pixstem.api as ps


@pytest.fixture(autouse=True)
def doctest_setup_teardown(request):
    plt.ioff()
    doctest_plugin = request.config.pluginmanager.getplugin("doctest")
    if isinstance(request.node, doctest_plugin.DoctestItem):
        tmp_dir = TemporaryDirectory()
        org_dir = os.getcwd()
        os.chdir(tmp_dir.name)
        yield
        os.chdir(org_dir)
        tmp_dir.cleanup()
    else:
        yield
    plt.close('all')


@pytest.fixture(autouse=True)
def add_np_ps(doctest_namespace):
    doctest_namespace['np'] = numpy
    doctest_namespace['ps'] = ps
