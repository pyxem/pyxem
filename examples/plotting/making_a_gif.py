"""
============
Making a GIF
============

In this example we will show how to use matplotlib to create a GIF from a 4D STEM
dataset.  This is useful for quickly showing a dataset in a presentation or
publication.
"""

import pyxem as pxm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
import numpy as np
from matplotlib.animation import FuncAnimation

fontprops = fm.FontProperties(size=18, family="serif")
family = "serif"

# Load the data
tilt = pxm.data.tilt_boundary_data()


# Define some functions for plotting
def plot_index(
    signal,
    index,
    navigator,
    ax1=None,
    ax2=None,
    fig=None,
    image1=None,
    image2=None,
    cmap="hot",
):
    """
    Plot the signal at some index along with a partial navigator.  This is
    useful when making GIFs showing a raster across some entire dataset.
    """
    if ax1 is None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    partial_nav = navigator.data.copy()
    partial_nav[index[0] + 1 :, :] = np.nan
    partial_nav[index[0], index[1] :] = np.nan
    if image1 is None:
        max_val = np.max(navigator.data)
        min_val = np.min(navigator.data)

        image1 = ax1.imshow(
            partial_nav,
            cmap="gray",
            extent=signal.axes_manager.navigation_extent,
            vmax=max_val,
            vmin=min_val,
        )
        ax1.axis("on")
        ax1.set_title("Virtual Image", size=20, family=family)
        ax1.set_ylabel("y axis (nm)", size=16, family=family)
        ax1.set_xlabel("x axis (nm)", size=16, family=family)
        ax1.set_xticks([])
        ax1.set_yticks([])
        scalebar = AnchoredSizeBar(
            ax1.transData,
            2,
            "2 nm",
            "lower left",
            pad=0.8,
            color="w",
            frameon=False,
            size_vertical=0.5,
            fontproperties=fontprops,
        )

        ax1.add_artist(scalebar)
    else:  # Update the image
        image1.set_data(partial_nav)

    if image2 is None:
        image2 = ax2.imshow(
            signal.inav[index[1], index[0]].data,
            cmap=cmap,
            extent=signal.axes_manager.signal_extent,
        )
        ax2.axis("on")
        ax2.set_title("Diffraction Pattern", size=20, family=family)
        ax2.set_ylabel("k$_y$ axis ($\AA^{-1}$)", size=16, family=family)
        ax2.set_xlabel("k$_x$ axis ($\AA^{-1}$)", size=16, family=family)
        ax2.set_xticks([])
        ax2.set_yticks([])
        scalebar2 = AnchoredSizeBar(
            ax2.transData,
            1,
            "1 $\AA^{-1}$",
            "lower left",
            pad=0.8,
            color="white",
            frameon=False,
            size_vertical=0.025,
            fontproperties=fontprops,
        )

        ax2.add_artist(scalebar2)
    else:
        image2.set_data(signal.inav[index[1], index[0]].data)
    return ax1, ax2, image1, image2, fig


# %%
# Plot the navigator and the diffraction pattern

navigator = tilt.sum(axis=(2, 3))
plot_index(tilt, index=(4, 4), navigator=navigator)

# %%
# Make this into a matplotlib animation:


def animate_4DSTEM(
    signal,
    navigator=None,
    step=1,
):
    if navigator is None:
        navigator = signal.sum(axis=(2, 3))

    ax1, ax2, image1, image2, fig = plot_index(
        signal, index=(0, 0), navigator=navigator
    )
    indexes = list(np.ndindex(signal.axes_manager.navigation_shape[::-1]))
    indexes = indexes[::step]

    def animate(i):
        plot_index(
            signal,
            index=(indexes[i][0], indexes[i][1]),
            ax1=ax1,
            ax2=ax2,
            image1=image1,
            image2=image2,
            fig=fig,
            navigator=navigator,
        )

    ani = FuncAnimation(fig, animate, frames=len(indexes), interval=10, repeat=False)

    ani.save(
        "4DSTEM.gif",
    )


animate_4DSTEM(tilt)

# %%
