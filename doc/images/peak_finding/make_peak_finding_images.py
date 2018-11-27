import matplotlib
import matplotlib.pyplot as plt
import pixstem.api as ps
import pixstem.marker_tools as mt

# Finding peaks
s = ps.dummy_data.get_cbed_signal()
peak_array = s.find_peaks(lazy_result=False, show_progressbar=False)
mt.add_peak_array_to_signal_as_markers(s, peak_array)
s.plot()
fig_peaks = s._plot.signal_plot.figure
fig_peaks.savefig("cbed_with_peaks.jpg")

# Closing all the matplotlib figures
matplotlib.pyplot.close("all")
