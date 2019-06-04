import matplotlib
import matplotlib.pyplot as plt
import pixstem.api as ps

# Initial dummy CBED data
s = ps.dummy_data.get_cbed_signal()
s.plot()
fig_cbed = s._plot.signal_plot.figure
fig_cbed.savefig("cbed_diff.jpg")

# Template matching
s_template = s.template_match_disk(disk_r=5, lazy_result=False)
s_template.plot()
fig_template = s_template._plot.signal_plot.figure
fig_template.savefig("cbed_template.jpg")

# Ring template matching
s_ring_template = s.template_match_ring(r_inner=3, r_outer=5, lazy_result=False)
s_ring_template.plot()
fig_ring_template = s_ring_template._plot.signal_plot.figure
fig_ring_template.savefig("cbed_ring_template.jpg")

# Closing all the matplotlib figures
matplotlib.pyplot.close("all")
