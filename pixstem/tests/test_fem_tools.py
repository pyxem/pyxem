import pixstem.fem_tools as femt
import pixstem.dummy_data as dd
from tempfile import TemporaryDirectory
from matplotlib.figure import Figure
import numpy as np


class TestFemResultIo:

    def test_femresults_io(self):
        tmpdir = TemporaryDirectory()

        s = dd.get_fem_signal()
        femresult = femt.fem_calc(s, centre_x=50, centre_y=50,
                                  show_progressbar=False)

        femt.save_fem(femresult, tmpdir.name)

        femresult1 = femt.load_fem(tmpdir.name)
        assert type(femresult1) is dict
        assert len(femresult1.keys()) == 7
        assert 'RadialInt' in femresult1.keys()
        assert 'V-Omegak' in femresult1.keys()
        assert 'RadialAvg' in femresult1.keys()
        assert 'Omega-Vi' in femresult1.keys()
        assert 'Omega-Vk' in femresult1.keys()
        assert 'Vrk' in femresult1.keys()
        assert 'Vrek' in femresult1.keys()
        tmpdir.cleanup()


class TestFemCalc:

    def test_full_calc(self):
        s = dd.get_fem_signal()
        femresult = femt.fem_calc(s, centre_x=50, centre_y=50,
                                  show_progressbar=False)
        assert type(femresult) is dict
        assert len(femresult.keys()) == 7
        assert 'RadialInt' in femresult.keys()
        assert 'V-Omegak' in femresult.keys()
        assert 'RadialAvg' in femresult.keys()
        assert 'Omega-Vi' in femresult.keys()
        assert 'Omega-Vk' in femresult.keys()
        assert 'Vrk' in femresult.keys()
        assert 'Vrek' in femresult.keys()

    def test_simple_calc(self):
        s = dd.get_simple_fem_signal()
        femresult = femt.fem_calc(s, centre_x=25, centre_y=25,
                                  show_progressbar=False)
        assert type(femresult) is dict
        assert len(femresult.keys()) == 7
        assert 'RadialInt' in femresult.keys()
        assert 'V-Omegak' in femresult.keys()
        assert 'RadialAvg' in femresult.keys()
        assert 'Omega-Vi' in femresult.keys()
        assert 'Omega-Vk' in femresult.keys()
        assert 'Vrk' in femresult.keys()
        assert 'Vrek' in femresult.keys()

    def test_nonsquare_navigation_calc(self):
        s = dd.get_generic_fem_signal(5, 6, 50, 50)
        femresult = femt.fem_calc(s, centre_x=25, centre_y=25,
                                  show_progressbar=False)
        assert type(femresult) is dict
        assert len(femresult.keys()) == 7
        assert 'RadialInt' in femresult.keys()
        assert 'V-Omegak' in femresult.keys()
        assert 'RadialAvg' in femresult.keys()
        assert 'Omega-Vi' in femresult.keys()
        assert 'Omega-Vk' in femresult.keys()
        assert 'Vrk' in femresult.keys()
        assert 'Vrek' in femresult.keys()

    def test_nonsquare_signal_calc(self):
        s = dd.get_generic_fem_signal(5, 5, 51, 50)
        femresult = femt.fem_calc(s, centre_x=25, centre_y=25,
                                  show_progressbar=False)
        assert type(femresult) is dict
        assert len(femresult.keys()) == 7
        assert 'RadialInt' in femresult.keys()
        assert 'V-Omegak' in femresult.keys()
        assert 'RadialAvg' in femresult.keys()
        assert 'Omega-Vi' in femresult.keys()
        assert 'Omega-Vk' in femresult.keys()
        assert 'Vrk' in femresult.keys()
        assert 'Vrek' in femresult.keys()


class TestFemPlot:

    def test_plot_square_fem(self):
        s = dd.get_simple_fem_signal()
        femresult = femt.fem_calc(s, centre_x=25, centre_y=25,
                                  show_progressbar=False)
        fig = femt.plot_fem(s, femresult, lowcutoff=10, highcutoff=120,
                            k_cal=None)
        assert type(fig) is Figure

    def test_plot_nonsquare_navigation_fem(self):
        s = dd.get_generic_fem_signal(5, 6, 50, 50)
        femresult = femt.fem_calc(s, centre_x=25, centre_y=25,
                                  show_progressbar=False)
        fig = femt.plot_fem(s, femresult, lowcutoff=10, highcutoff=120,
                            k_cal=None)
        assert type(fig) is Figure

    def test_plot_nonsquare_signal_fem(self):
        s = dd.get_generic_fem_signal(5, 5, 51, 50)
        femresult = femt.fem_calc(s, centre_x=25, centre_y=25,
                                  show_progressbar=False)
        fig = femt.plot_fem(s, femresult, lowcutoff=10, highcutoff=120,
                            k_cal=None)
        assert type(fig) is Figure

    def test_plot_parameters_kcal_none(self):
        s = dd.get_simple_fem_signal()

        femresult = femt.fem_calc(s, centre_x=25, centre_y=25,
                                  show_progressbar=False)

        lowcutoff, highcutoff, k_cal = 10, 20, None

        xaxis = np.arange(0, len(femresult['RadialAvg'].data))

        fig = femt.plot_fem(s, femresult, lowcutoff=lowcutoff,
                            highcutoff=highcutoff, k_cal=k_cal)

        assert fig.axes[1].xaxis.properties()['data_interval'][0] == \
            xaxis[lowcutoff]
        assert fig.axes[1].xaxis.properties()['data_interval'][1] == \
            xaxis[highcutoff - 1]
        assert xaxis[1] - xaxis[0] == 1

    def test_plot_parameters_kcal_finite(self):
        s = dd.get_simple_fem_signal()

        femresult = femt.fem_calc(s, centre_x=25, centre_y=25,
                                  show_progressbar=False)

        lowcutoff, highcutoff, k_cal = 10, 20, 0.25

        xaxis = 2 * np.pi * k_cal * \
            np.arange(0, len(femresult['RadialAvg'].data))

        fig = femt.plot_fem(s, femresult, lowcutoff=lowcutoff,
                            highcutoff=highcutoff, k_cal=k_cal)

        assert fig.axes[1].xaxis.properties()['data_interval'][0] == \
            xaxis[lowcutoff]
        assert fig.axes[1].xaxis.properties()['data_interval'][1] ==\
            xaxis[highcutoff - 1]
        assert xaxis[1] - xaxis[0] == 2 * np.pi * k_cal
