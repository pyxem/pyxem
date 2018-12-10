# import numpy as np
import pixstem.fem_tools as femt
import pixstem.dummy_data as dd

# s = dd.get_fem_signal()
# fem_results = s.fem_analysis(centre_x=128,centre_y=128,show_progressbar=False)
# fem_results['V-Omegak'].plot()


class TestFemCalc:

    def test(self):
        s = dd.get_fem_signal()
        femresult = femt.fem_calc(s, centre_x=128, centre_y=128, show_progressbar=False)
        assert type(femresult) is dict

    # def testkeys(self):
    #     keys = ['Omega-Vi',
    #             'Omega-Vk',
    #             'RadialAvg',
    #             'RadialInt',
    #             'V-Omegak',
    #             'Vrek',
    #             'Vrk']
    #
    # def testload(self):
    #     pass
    #
    # def testsave(self):
    #     pass
    #
    # def testplot(self):
    #     pass
