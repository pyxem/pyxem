import fpd_data_processing.api as fp

s = fp.dummy_data.get_holz_heterostructure_test_signal()
s.plot()
fig_signal = s._plot.signal_plot.figure
fig_navigator = s._plot.navigator_plot.figure

fig_signal.savefig("testdata_signal.png")
fig_navigator.savefig("testdata_navigator.png")
