import matplotlib.pyplot as plt
import pixstem.api as ps

s = ps.dummy_data.get_nanobeam_electron_diffraction_signal()

# First look at the dataset
s.plot()
fig_signal = s._plot.signal_plot.figure
fig_navigator = s._plot.navigator_plot.figure

fig_signal.savefig("s_signal.png")
fig_navigator.savefig("s_navigator.png")

# Finding the peaks for the raw data
s1 = s.inav[:1, :1]
peak_array1 = s1.find_peaks(lazy_result=False)
s1.add_peak_array_as_markers(peak_array1)
s1.plot()
s1._plot.signal_plot.figure.savefig("s_peak_finding.png")

# Template matching
s1t = s1.template_match_disk(disk_r=5, lazy_result=False)
s1t.plot()
s1t._plot.signal_plot.figure.savefig("s_template_matching.png")

# Template matching and peak finding

peak_array1 = s1t.find_peaks(lazy_result=False)
s1t.add_peak_array_as_markers(peak_array1)
s1t.plot()
s1t._plot.signal_plot.figure.savefig("s_template_matching_peak_array.png")
s1.add_peak_array_as_markers(peak_array1)
s1.plot()
s1._plot.signal_plot.figure.savefig("s_peak_array.png")

# Refining peak array
peak_array1_com = s1.peak_position_refinement_com(peak_array1, lazy_result=False)
s1.add_peak_array_as_markers(peak_array1_com, color='blue')
s1.plot()
s1._plot.signal_plot.figure.savefig("s_peak_array_with_refinement.png")

# Removing background
s1_rem = s1.subtract_diffraction_background(lazy_result=False)
s1_rem.plot()
s1_rem._plot.signal_plot.figure.savefig("s_remove_background.png")

# Refining positions
peak_array1_rem_com = s1_rem.peak_position_refinement_com(peak_array1, lazy_result=False)
s1_rem.add_peak_array_as_markers(peak_array1_rem_com)
s1_rem.plot()
s1_rem._plot.signal_plot.figure.savefig("s_remove_background_peak_array.png")
