import os
import pytest
import numpy as np
from tempfile import TemporaryDirectory
import hyperspy.api as hs
import pixstem.api as ps
import pixstem.io_tools as it


my_path = os.path.dirname(__file__)


class TestDpcsignalIo:

    def setup_method(self):
        self.tmpdir = TemporaryDirectory()

    def teardown_method(self):
        self.tmpdir.cleanup()

    def test_load_basesignal(self):
        filename = os.path.join(
                my_path, "test_data", "dpcbasesignal_test.hdf5")
        ps.load_dpc_signal(filename)

    def test_load_signal1d(self):
        filename = os.path.join(
                my_path, "test_data", "dpcsignal1d_test.hdf5")
        ps.load_dpc_signal(filename)

    def test_load_signal2d(self):
        filename = os.path.join(
                my_path, "test_data", "dpcsignal2d_test.hdf5")
        ps.load_dpc_signal(filename)

    def test_load_signal2d_too_many_nav_dim(self):
        filename = os.path.join(
                my_path, "test_data", "dpcsignal2d_test_too_many_nav_dim.hdf5")
        with pytest.raises(Exception):
            ps.load_dpc_signal(filename)

    def test_load_basesignal_too_many_signal_dim(self):
        filename = os.path.join(
                my_path,
                "test_data",
                "dpcbasesignal_test_too_many_signal_dims.hdf5")
        with pytest.raises(NotImplementedError):
            ps.load_dpc_signal(filename)

    def test_retain_metadata(self):
        s = ps.DPCSignal2D(np.ones((2, 10, 5)))
        s.metadata.General.title = "test_data"
        filename = os.path.join(self.tmpdir.name, 'test_metadata.hspy')
        s.save(filename)
        s_load = ps.load_dpc_signal(filename)
        assert s_load.metadata.General.title == "test_data"

    def test_retain_axes_manager(self):
        s = ps.DPCSignal2D(np.ones((2, 10, 5)))
        s_sa0 = s.axes_manager.signal_axes[0]
        s_sa1 = s.axes_manager.signal_axes[1]
        s_sa0.offset, s_sa1.offset, s_sa0.scale, s_sa1.scale = 20, 10, 0.2, 0.3
        s_sa0.units, s_sa1.units, s_sa0.name, s_sa1.name = "a", "b", "e", "f"
        filename = os.path.join(self.tmpdir.name, 'test_axes_manager.hspy')
        s.save(filename)
        s_load = ps.load_dpc_signal(filename)
        assert s_load.axes_manager[1].offset == 20
        assert s_load.axes_manager[2].offset == 10
        assert s_load.axes_manager[1].scale == 0.2
        assert s_load.axes_manager[2].scale == 0.3
        assert s_load.axes_manager[1].units == "a"
        assert s_load.axes_manager[2].units == "b"
        assert s_load.axes_manager[1].name == "e"
        assert s_load.axes_manager[2].name == "f"


class TestPixelatedstemSignalIo:

    def test_load_hspy_signal(self):
        # Has shape (2, 5, 4, 3)
        filename = os.path.join(
                my_path, "test_data", "pixelated_stem_test.hdf5")
        s = ps.load_ps_signal(filename)
        assert s.axes_manager.shape == (2, 5, 4, 3)

    def test_load_hspy_signal_generated(self):
        shape = (7, 6, 3, 5)
        tmpdir = TemporaryDirectory()
        filename = os.path.join(
                tmpdir.name, "test.hdf5")
        s = ps.PixelatedSTEM(np.zeros(shape))
        s.save(filename)

        sl = ps.load_ps_signal(filename, lazy=False)
        assert sl.axes_manager.shape == (
                shape[1], shape[0], shape[3], shape[2])
        tmpdir.cleanup()

    def test_load_hspy_signal_generated_lazy(self):
        shape = (3, 5, 7, 9)
        tmpdir = TemporaryDirectory()
        filename = os.path.join(
                tmpdir.name, "test_lazy.hdf5")
        s = ps.PixelatedSTEM(np.zeros(shape))
        s.save(filename)

        sl = ps.load_ps_signal(filename, lazy=True)
        assert sl.axes_manager.shape == (
                shape[1], shape[0], shape[3], shape[2])
        tmpdir.cleanup()

    def test_load_ps_signal(self):
        # Dataset has known size (2, 2, 256, 256)
        filename = os.path.join(
                my_path, "test_data", "fpd_file_test.hdf5")
        s = ps.load_ps_signal(filename)
        assert s.axes_manager.shape == (2, 2, 256, 256)

        s = ps.load_ps_signal(filename, lazy=True)
        assert s.axes_manager.shape == (2, 2, 256, 256)

    def test_navigation_signal(self):
        # Dataset has known size (2, 2, 256, 256)
        filename = os.path.join(
                my_path, "test_data", "fpd_file_test.hdf5")
        s = ps.load_ps_signal(filename)
        assert s.axes_manager.shape == (2, 2, 256, 256)

        s_nav0 = hs.signals.Signal2D(np.zeros((2, 2)))
        s = ps.load_ps_signal(filename, lazy=True, navigation_signal=s_nav0)
        s.plot()

        s_nav1 = hs.signals.Signal2D(np.zeros((2, 4)))
        with pytest.raises(ValueError):
            s = ps.load_ps_signal(
                    filename, lazy=True, navigation_signal=s_nav1)


class TestSignalToPixelatedStem:

    def test_conserve_signal_axes_metadata(self):
        x_nav, y_nav, x_sig, y_sig = 9, 8, 5, 7
        x_nav_scale, y_nav_scale, x_sig_scale, y_sig_scale = 0.5, 0.2, 1.2, 3.2
        x_nav_off, y_nav_off, x_sig_off, y_sig_off = 30, 12, 76, 32
        x_nav_name, y_nav_name, x_sig_name, y_sig_name = "nX", "nY", "sX", "sY"
        x_nav_unit, y_nav_unit, x_sig_unit, y_sig_unit = "u1", "u2", "u3", "u4"
        title = "test_title"
        data = np.random.random((y_nav, x_nav, y_sig, x_sig))

        s = hs.signals.Signal2D(data)
        s.metadata.General.title = title
        am = s.axes_manager
        am[0].scale, am[0].offset = x_nav_scale, x_nav_off
        am[1].scale, am[1].offset = y_nav_scale, y_nav_off
        am[2].scale, am[2].offset = x_sig_scale, x_sig_off
        am[3].scale, am[3].offset = y_sig_scale, y_sig_off
        am[0].name, am[0].units = x_nav_name, x_nav_unit
        am[1].name, am[1].units = y_nav_name, y_nav_unit
        am[2].name, am[2].units = x_sig_name, x_sig_unit
        am[3].name, am[3].units = y_sig_name, y_sig_unit

        s1 = it.signal_to_pixelated_stem(s)

        assert (data == s1.data).all()
        assert s1.metadata.General.title == title
        assert s1.axes_manager.shape == (x_nav, y_nav, x_sig, y_sig)
        assert s1.axes_manager[0].scale == x_nav_scale
        assert s1.axes_manager[1].scale == y_nav_scale
        assert s1.axes_manager[2].scale == x_sig_scale
        assert s1.axes_manager[3].scale == y_sig_scale

        assert s1.axes_manager[0].offset == x_nav_off
        assert s1.axes_manager[1].offset == y_nav_off
        assert s1.axes_manager[2].offset == x_sig_off
        assert s1.axes_manager[3].offset == y_sig_off

        assert s1.axes_manager[0].name == x_nav_name
        assert s1.axes_manager[1].name == y_nav_name
        assert s1.axes_manager[2].name == x_sig_name
        assert s1.axes_manager[3].name == y_sig_name

        assert s1.axes_manager[0].units == x_nav_unit
        assert s1.axes_manager[1].units == y_nav_unit
        assert s1.axes_manager[2].units == x_sig_unit
        assert s1.axes_manager[3].units == y_sig_unit


class TestCopyAxesPsToDpc:

    def test_simple(self):
        s_ps = ps.PixelatedSTEM(np.zeros((10, 13, 5, 4)))
        s_dp = ps.DPCSignal2D(np.zeros((2, 10, 13)))
        it._copy_axes_ps_to_dpc(s_ps, s_dp)

    def test_copy_value(self):
        dpc_nav_name = 'q'
        dpc_nav_units = 'x'
        dpc_nav_offse = 32
        dpc_nav_scale = 0.2

        ps_sig0_scale, ps_sig1_scale = 1.3, 5.2
        ps_sig0_offse, ps_sig1_offse = -40, 52
        ps_sig0_units, ps_sig1_units = 'sa', 'sb'
        ps_sig0_name, ps_sig1_name = 'sc', 'sd'
        ps_nav0_scale, ps_nav1_scale = 0.2, 4
        ps_nav0_offse, ps_nav1_offse = 20, -12
        ps_nav0_units, ps_nav1_units = 'na', 'nb'
        ps_nav0_name, ps_nav1_name = 'nc', 'nd'

        s_ps = ps.PixelatedSTEM(np.zeros((10, 13, 5, 4)))
        s_ps.axes_manager.navigation_axes[0].scale = ps_nav0_scale
        s_ps.axes_manager.navigation_axes[1].scale = ps_nav1_scale
        s_ps.axes_manager.navigation_axes[0].offset = ps_nav0_offse
        s_ps.axes_manager.navigation_axes[1].offset = ps_nav1_offse
        s_ps.axes_manager.navigation_axes[0].units = ps_nav0_units
        s_ps.axes_manager.navigation_axes[1].units = ps_nav1_units
        s_ps.axes_manager.navigation_axes[0].name = ps_nav0_name
        s_ps.axes_manager.navigation_axes[1].name = ps_nav1_name
        s_ps.axes_manager.signal_axes[0].scale = ps_sig0_scale
        s_ps.axes_manager.signal_axes[1].scale = ps_sig1_scale
        s_ps.axes_manager.signal_axes[0].offset = ps_sig0_offse
        s_ps.axes_manager.signal_axes[1].offset = ps_sig1_offse
        s_ps.axes_manager.signal_axes[0].units = ps_sig0_units
        s_ps.axes_manager.signal_axes[1].units = ps_sig1_units
        s_ps.axes_manager.signal_axes[0].name = ps_sig0_name
        s_ps.axes_manager.signal_axes[1].name = ps_sig1_name

        s_dp = ps.DPCSignal2D(np.zeros((2, 10, 13)))
        s_dp.axes_manager.navigation_axes[0].scale = dpc_nav_scale
        s_dp.axes_manager.navigation_axes[0].offset = dpc_nav_offse
        s_dp.axes_manager.navigation_axes[0].name = dpc_nav_name
        s_dp.axes_manager.navigation_axes[0].units = dpc_nav_units
        it._copy_axes_ps_to_dpc(s_ps, s_dp)

        # Check everything is copied to the dpc signal
        assert s_dp.axes_manager.signal_axes[0].scale == ps_nav0_scale
        assert s_dp.axes_manager.signal_axes[1].scale == ps_nav1_scale
        assert s_dp.axes_manager.signal_axes[0].offset == ps_nav0_offse
        assert s_dp.axes_manager.signal_axes[1].offset == ps_nav1_offse
        assert s_dp.axes_manager.signal_axes[0].units == ps_nav0_units
        assert s_dp.axes_manager.signal_axes[1].units == ps_nav1_units
        assert s_dp.axes_manager.signal_axes[0].name == ps_nav0_name
        assert s_dp.axes_manager.signal_axes[1].name == ps_nav1_name

        # Check that s_ps is not changed
        assert s_ps.axes_manager.signal_axes[0].scale == ps_sig0_scale
        assert s_ps.axes_manager.signal_axes[1].scale == ps_sig1_scale
        assert s_ps.axes_manager.signal_axes[0].offset == ps_sig0_offse
        assert s_ps.axes_manager.signal_axes[1].offset == ps_sig1_offse
        assert s_ps.axes_manager.signal_axes[0].units == ps_sig0_units
        assert s_ps.axes_manager.signal_axes[1].units == ps_sig1_units
        assert s_ps.axes_manager.signal_axes[0].name == ps_sig0_name
        assert s_ps.axes_manager.signal_axes[1].name == ps_sig1_name
        assert s_ps.axes_manager.navigation_axes[0].scale == ps_nav0_scale
        assert s_ps.axes_manager.navigation_axes[1].scale == ps_nav1_scale
        assert s_ps.axes_manager.navigation_axes[0].offset == ps_nav0_offse
        assert s_ps.axes_manager.navigation_axes[1].offset == ps_nav1_offse
        assert s_ps.axes_manager.navigation_axes[0].units == ps_nav0_units
        assert s_ps.axes_manager.navigation_axes[1].units == ps_nav1_units
        assert s_ps.axes_manager.navigation_axes[0].name == ps_nav0_name
        assert s_ps.axes_manager.navigation_axes[1].name == ps_nav1_name

        # Check s_dp is not changed
        assert s_dp.axes_manager.navigation_axes[0].scale == dpc_nav_scale
        assert s_dp.axes_manager.navigation_axes[0].offset == dpc_nav_offse
        assert s_dp.axes_manager.navigation_axes[0].units == dpc_nav_units
        assert s_dp.axes_manager.navigation_axes[0].name == dpc_nav_name

    def test_1d_input(self):
        s_dp = ps.DPCSignal1D(np.zeros((2, 10)))
        s_ps = ps.PixelatedSTEM(np.zeros((10, 5, 4)))
        it._copy_axes_ps_to_dpc(s_ps, s_dp)

    def test_wrong_input(self):
        s_dp = ps.DPCSignal2D(np.zeros((2, 10, 13)))
        s_ps = ps.PixelatedSTEM(np.zeros((3, 10, 13, 5, 4)))
        with pytest.raises(ValueError):
            it._copy_axes_ps_to_dpc(s_ps, s_dp)

    def test_wrong_nav_shape(self):
        s_dp = ps.DPCSignal2D(np.zeros((2, 16, 13)))
        s_ps = ps.PixelatedSTEM(np.zeros((10, 13, 5, 4)))
        with pytest.raises(ValueError):
            it._copy_axes_ps_to_dpc(s_ps, s_dp)
