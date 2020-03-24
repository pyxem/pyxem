import matplotlib
import pixstem.api as ps
import pixstem.fem_tools as ft


s = ps.dummy_data.get_fem_signal()
fem_results = s.fem_analysis(centre_x=50, centre_y=50, show_progressbar=False)

# Plot V-Omegak
s_v_omegak = fem_results['V-Omegak']
s_v_omegak.plot()
fig_signal = s_v_omegak._plot.signal_plot.figure
fig_signal.savefig("fem_v_omegak.png")

# Visualize all the results
fig_full_results = ft.plot_fem(s, fem_results)
fig_full_results.savefig("fem_full_results.png")

# Closing all figures
matplotlib.pyplot.close("all")
