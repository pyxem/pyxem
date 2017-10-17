import fpd_data_processing.api as fp

s = fp.dummy_data.get_holz_heterostructure_test_signal()

# First look at the dataset
s.plot()
fig_signal = s._plot.signal_plot.figure
fig_navigator = s._plot.navigator_plot.figure

fig_signal.savefig("testdata_signal.png")
fig_navigator.savefig("testdata_navigator.png")

# Changed contrast
s._plot.signal_plot.set_contrast(0, 4)
fig_signal.savefig("testdata_better_contrast_signal.png")

# Plotting center of mass
s_com = s.center_of_mass(threshold=2)
s_com.plot()
fig_signal = s_com._plot.signal_plot.figure
fig_navigator = s_com._plot.navigator_plot.figure

fig_signal.savefig("testdata_com_signal.png")
fig_navigator.savefig("testdata_com_navigator.png")

# Plotting radial integration signal
s_radial = s.radial_integration(centre_x=s_com.inav[0].data, centre_y=s_com.inav[1].data)
s_radial.plot()
fig_signal = s_radial._plot.signal_plot.figure
fig_navigator = s_radial._plot.navigator_plot.figure

fig_signal.savefig("testdata_radial_signal.png")
fig_navigator.savefig("testdata_radial_navigator.png")

# Plotting transposed radial signal
s_radial_T = s_radial.T
s_radial_T.axes_manager.indices = (32, )
s_radial_T.plot()
fig_signal = s_radial_T._plot.signal_plot.figure
fig_navigator = s_radial_T._plot.navigator_plot.figure

fig_signal.savefig("testdata_radial_T_signal.png")
fig_navigator.savefig("testdata_radial_T_navigator.png")

# Fitting offset to background model
s_radial_cropped = s_radial.isig[20:40]
m_r = s_radial_cropped.create_model()
from hyperspy.components1d import Offset

offset = Offset()
m_r.set_signal_range(20., 25.)
m_r.set_signal_range(37., 40.)
m_r.append(offset)
m_r.multifit()
m_r.reset_signal_range()
m_r.plot()

fig_signal = m_r._plot.signal_plot.figure
fig_navigator = m_r._plot.navigator_plot.figure

fig_signal.savefig("testdata_offset_model_signal.png")
fig_navigator.savefig("testdata_offset_model_navigator.png")

# Fitting offset to background model
from hyperspy.components1d import Gaussian
g = Gaussian()
m_r.append(g)
m_r.fit_component(g, signal_range=(25, 35), only_current=False)
m_r.multifit()
m_r.plot()

fig_signal = m_r._plot.signal_plot.figure
fig_navigator = m_r._plot.navigator_plot.figure

fig_signal.savefig("testdata_gaussian_model_signal.png")
fig_navigator.savefig("testdata_gaussian_model_navigator.png")

# Plotting Gaussian function amplitude
g_A = g.A.as_signal()
g_A.plot()

fig_signal = g_A._plot.signal_plot.figure

fig_signal.savefig("testdata_gaussian_amplitude.png")

# Plotting Gaussian function centre
g_centre = g.centre.as_signal()
g_centre.plot()

fig_signal = g_centre._plot.signal_plot.figure

fig_signal.savefig("testdata_gaussian_centre.png")

