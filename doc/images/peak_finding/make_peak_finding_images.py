import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import pixstem.api as ps

# Finding peaks
s = ps.dummy_data.get_cbed_signal()
peak_array = s.find_peaks(lazy_result=False, show_progressbar=False)
s.add_peak_array_as_markers(peak_array, color='purple', size=18)
s.plot()
fig_peaks = s._plot.signal_plot.figure
fig_peaks.savefig("cbed_with_peaks.jpg")

# Closing all the matplotlib figures
matplotlib.pyplot.close("all")
