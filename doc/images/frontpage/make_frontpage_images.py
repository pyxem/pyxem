import matplotlib
import matplotlib.pyplot as plt
import hyperspy.api as hs
import pixstem.api as ps
import pixstem.make_diffraction_test_data as mdtd

# Plotting STEM diffraction pattern
s = hs.load("stem_diffraction_pattern.hspy")
fig, ax = plt.subplots(figsize=(5, 5))
cax = ax.imshow(s.data, cmap='viridis')
cax.set_clim(0, 150)
fig.subplots_adjust(0, 0, 1, 1)
fig.savefig("stem_diffraction.jpg")

# NBED example
di = mdtd.DiffractionTestImage(rotation=18)
di.add_disk(x=128, y=128, intensity=10.)
di.add_cubic_disks(vx=20, vy=20, intensity=9., n=5)
di.add_background_lorentz(intensity=50, width=30)
s = di.get_signal()
st = s.template_match_disk(disk_r=5, lazy_result=False)
peak_array = st.find_peaks(lazy_result=False)
peak_array = peak_array[()]

fig, ax = plt.subplots(figsize=(5, 5))
ax.imshow(s.data)
ax.scatter(peak_array[:, 1], peak_array[:, 0], s=5.9, color='red')
ax.set_axis_off()
fig.subplots_adjust(0, 0, 1, 1)
fig.savefig("nbed_example.jpg")

# DPC color image
s = ps.dummy_data.get_square_dpc_signal().get_color_signal()
s.metadata.General.title = "Magnetic DPC example data"
s.plot()
fig_sig_x = s._plot.signal_plot.figure
fig_sig_x.savefig("dpc_dummy_data.jpg")

# Closing all the matplotlib figures
matplotlib.pyplot.close("all")
