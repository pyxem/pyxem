import matplotlib
import matplotlib.pyplot as plt
import fpd_data_processing.api as fp

# Initial dummy DPC data
s = fp.dummy_data.get_square_dpc_signal(add_ramp=True)
s.plot()
fig_sig_x = s._plot.signal_plot.figure
fig_sig_x.savefig("dpc_x_raw.jpg")

s.axes_manager.indices = (1, )
fig_sig_y = s._plot.signal_plot.figure
fig_sig_y.savefig("dpc_y_raw.jpg")

# Correcting ramp
s1 = s.correct_ramp()
s1.plot()
fig_sig_x = s1._plot.signal_plot.figure
fig_sig_x.savefig("dpc_x_cor.jpg")

s1.axes_manager.indices = (1, )
fig_sig_y = s1._plot.signal_plot.figure
fig_sig_y.savefig("dpc_y_cor.jpg")

# Plot DPC color image
s_color = s1.get_color_signal()
s_color.plot()
fig_sig = s_color._plot.signal_plot.figure
fig_sig.savefig("dpc_color_image.jpg")

# Plot DPC phase image
s_phase = s1.get_phase_signal()
s_phase.plot()
fig_sig = s_phase._plot.signal_plot.figure
fig_sig.savefig("dpc_phase_image.jpg")

# Plot DPC mangitude image
s_magnitude = s1.get_magnitude_signal()
s_magnitude.plot()
fig_sig = s_magnitude._plot.signal_plot.figure
fig_sig.savefig("dpc_magnitude_image.jpg")

# Plot bivariate histogram
s_hist = s1.get_bivariate_histogram()
s_hist.plot(cmap='viridis')
fig_sig = s_hist._plot.signal_plot.figure
fig_sig.savefig("dpc_hist_image.jpg")

# Plot color image with indicator
fig_color_image = s1.get_color_image_with_indicator()
fig_color_image.savefig("dpc_color_image_indicator.jpg")

# Rotating the probe axes
s1_rot_probe = s1.rotate_data(10)
s1_rot_probe_color = s1_rot_probe.get_color_signal()
s1_rot_probe_color.plot()
fig_sig = s1_rot_probe_color._plot.signal_plot.figure
fig_sig.savefig("dpc_rotate_probe_color.jpg")

# Rotating the beam shifts
s1_rot_shifts = s1.rotate_beam_shifts(45)
s1_rot_shifts_color = s1_rot_shifts.get_color_signal()
s1_rot_shifts_color.plot()
fig_sig = s1_rot_shifts_color._plot.signal_plot.figure
fig_sig.savefig("dpc_rotate_shifts_color.jpg")

# Flipping the axes by 90 degrees, both probe and beam shifts
s2 = s1.deepcopy()
s2.data[0, 50:250, 145:155] += 5
s2_color = s2.get_color_signal()
s2_color.plot()
fig_sig = s2_color._plot.signal_plot.figure
fig_sig.savefig("dpc_rotate_flip_color1.jpg")

s2_rot_flip = s2.flip_axis_90_degrees()
s2_rot_flip_color = s2_rot_flip.get_color_signal()
s2_rot_flip_color.plot()
fig_sig = s2_rot_flip_color._plot.signal_plot.figure
fig_sig.savefig("dpc_rotate_flip_color2.jpg")

# Blurring the data
s = fp.dummy_data.get_square_dpc_signal()
s_color = s.get_color_signal()
s_color.plot()
fig_color = s_color._plot.signal_plot.figure
fig_color.savefig("dpc_gaussian_nonblur.jpg")

s_color_blur = s.gaussian_blur().get_color_signal()
s_color_blur.plot()
fig_color_blur = s_color_blur._plot.signal_plot.figure
fig_color_blur.savefig("dpc_gaussian_blur.jpg")

# Closing all the matplotlib figures
matplotlib.pyplot.close("all")
