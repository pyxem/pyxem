import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pixstem.api as ps

# Making data
data = np.zeros((2, 2, 50, 50))
data[:, :, 23:27, 23:27] = 1
data[:, :, 13:17, 23:27] = 1
data[:, :, 33:37, 23:27] = 1
s = ps.PixelatedSTEM(data)
s.plot()
fig_square = s._plot.signal_plot.figure
fig_square.savefig("square_diff.jpg")

# Template matching
binary_image = np.zeros((8, 8))
binary_image[2:-2, 2:-2] = 1
s_template = s.template_match_with_binary_image(
        binary_image, show_progressbar=False, lazy_result=False)
s_template.plot()
fig_template = s_template._plot.signal_plot.figure
fig_template.savefig("square_diff_template.jpg")

# Closing all the matplotlib figures
matplotlib.pyplot.close("all")
