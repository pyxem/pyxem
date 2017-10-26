import matplotlib
import matplotlib.pyplot as plt
import hyperspy.api as hs
import fpd_data_processing.api as fp
from fpd_data_processing.make_diffraction_test_data import MakeTestData

# Plotting STEM diffraction pattern
s = hs.load("stem_diffraction_pattern.hspy")
fig, ax = plt.subplots(figsize=(5, 5))
cax = ax.imshow(s.data, cmap='viridis')
cax.set_clim(0, 150)
fig.subplots_adjust(0, 0, 1, 1)
fig.savefig("stem_diffraction.jpg")

# DPC color image
s = fp.dummy_data.get_square_dpc_signal().get_color_signal()
s.metadata.General.title = "Magnetic DPC example data"
s.plot()
fig_sig_x = s._plot.signal_plot.figure
fig_sig_x.savefig("dpc_dummy_data.jpg")

# Closing all the matplotlib figures
matplotlib.pyplot.close("all")
