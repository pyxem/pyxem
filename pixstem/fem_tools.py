import matplotlib.pylab as plt
import numpy as np
from dask import delayed
from tqdm import tqdm
import hyperspy.api as hs
import pixstem


def fem_calc(s, centre_x=None, centre_y=None, show_progressbar=True):
    """Perform analysis of fluctuation electron microscopy (FEM) data
    as outlined in:

    T. L. Daulton, et al., Ultramicroscopy 110 (2010) 1279-1289.
    doi:10.1016/j.ultramic.2010.05.010

    Parameters
    ----------
    s : PixelatedSTEM
        Signal on which FEM analysis was performed
    centre_x, centre_y : int, optional
        All the diffraction patterns assumed to have the same
        centre position.

    show_progressbar : bool
        Default True

    Returns
    -------
    results : Python dictionary
        Results of FEM data analysis, including the normalized variance
        of the annular mean (V-Omegak), mean of normalized variances of
        rings (V-rk), normalized variance of ring ensemble (Vrek),
        the normalized variance image (Omega-Vi), and annular mean of
        the variance image (Omega-Vk).

    Examples
    --------
    >>> import pixstem.dummy_data as dd
    >>> import pixstem.fem_tools as femt
    >>> s = dd.get_fem_signal()
    >>> fem_results = femt.fem_calc(
    ...     s,
    ...     centre_x=128,
    ...     centre_y=128,
    ...     show_progressbar=False)
    >>> fem_results['V-Omegak'].plot()

    """
    offset = False

    if centre_x is None:
        centre_x = np.int(s.axes_manager.signal_shape[0]/2)

    if centre_y is None:
        centre_y = np.int(s.axes_manager.signal_shape[1]/2)

    if s.data.min() == 0:
        s.data += 1  # To avoid division by 0
        offset = True
    results = dict()

    results['RadialInt'] = (
        s.radial_average(centre_x=centre_x, centre_y=centre_y,
                         normalize=False,
                         show_progressbar=show_progressbar))

    radialavgs = s.radial_average(centre_x=centre_x, centre_y=centre_y,
                                  normalize=True,
                                  show_progressbar=show_progressbar)
    if radialavgs.data.min() == 0:
        radialavgs.data += 1

    results['V-Omegak'] = ((radialavgs ** 2).mean() /
                           (radialavgs.mean()) ** 2) - 1
    results['RadialAvg'] = radialavgs.mean()

    if s._lazy:
        results['Omega-Vi'] = ((s ** 2).mean() / (s.mean()) ** 2) - 1
        results['Omega-Vi'].compute(progressbar=show_progressbar)
        results['Omega-Vi'] = pixstem.pixelated_stem_class.PixelatedSTEM(
                results['Omega-Vi'])

        results['Omega-Vk'] = results['Omega-Vi'].radial_average(
                centre_x=centre_x, centre_y=centre_y, normalize=True,
                show_progressbar=show_progressbar)

        oldshape = None
        if len(s.data.shape) == 4:
            oldshape = s.data.shape
            s.data = s.data.reshape(
                s.data.shape[0] * s.data.shape[1],
                s.data.shape[2], s.data.shape[3])
        y, x = np.indices(s.data.shape[-2:])
        r = np.sqrt((x - centre_x) ** 2 + (y - centre_y) ** 2)
        r = r.astype(np.int)

        nr = np.bincount(r.ravel())
        Vrklist = []
        Vreklist = []

        for k in tqdm(range(0, len(nr)), disable=(not show_progressbar)):
            locs = np.where(r == k)
            vals = s.data.vindex[:, locs[0], locs[1]].T
            Vrklist.append(np.mean((np.mean(vals ** 2, 1) /
                                    np.mean(vals, 1) ** 2) - 1))
            Vreklist.append(np.mean(vals.ravel() ** 2) /
                            np.mean(vals.ravel()) ** 2 - 1)

        Vrkdask = delayed(Vrklist)
        Vrekdask = delayed(Vreklist)

        results['Vrk'] = hs.signals.Signal1D(
                Vrkdask.compute(progressbar=show_progressbar))
        results['Vrek'] = hs.signals.Signal1D(
                Vrekdask.compute(progressbar=show_progressbar))
    else:
        results['Omega-Vi'] = ((s ** 2).mean() / (s.mean()) ** 2) - 1
        results['Omega-Vk'] = results['Omega-Vi'].radial_average(
            centre_x=centre_x, centre_y=centre_y, normalize=True,
            show_progressbar=show_progressbar)
        oldshape = None
        if len(s.data.shape) == 4:
            oldshape = s.data.shape
            s.data = s.data.reshape(
                s.data.shape[0] * s.data.shape[1],
                s.data.shape[2],
                s.data.shape[3])
        y, x = np.indices(s.data.shape[-2:])
        r = np.sqrt((x - centre_x) ** 2 + (y - centre_y) ** 2)
        r = r.astype(np.int)

        nr = np.bincount(r.ravel())
        results['Vrk'] = np.zeros(len(nr))
        results['Vrek'] = np.zeros(len(nr))

        for k in tqdm(range(0, len(nr)), disable=(not show_progressbar)):
            locs = np.where(r == k)
            vals = s.data[:, locs[0], locs[1]]
            results['Vrk'][k] = np.mean(
                (np.mean(vals ** 2, 1) / np.mean(vals, 1) ** 2) - 1)
            results['Vrek'][k] = np.mean(
                vals.ravel() ** 2) / np.mean(vals.ravel()) ** 2 - 1

        results['Vrk'] = hs.signals.Signal1D(results['Vrk'])
        results['Vrek'] = hs.signals.Signal1D(results['Vrek'])

    if oldshape:
        s.data = s.data.reshape(oldshape)
    if offset:
        s.data -= 1  # Undo previous addition of 1 to input data
    return results


def plot_fem(s, results, lowcutoff=10, highcutoff=120, k_cal=None):
    """Produce a summary plot of all calculated FEM parameters

    Parameters
    ----------
    s : PixelatedSTEM
        Signal on which FEM analysis was performed
    results : Dictionary
        Series of HyperSpy Signals containing results of FEM analysis performed
        on s
    lowcutoff : integer
        Position of low-q cutoff for plots
    highcutoff : integer
        Position of high-q cutoff for plots
    k_cal : float or None
        Reciprocal space unit of length per pixel in inverse Angstroms

    Returns
    -------
    fig : Matplotlib Figure

    Examples
    --------
    >>> s = ps.dummy_data.get_fem_signal()
    >>> import pixstem.fem_tools as femt
    >>> fem_results = s.fem_analysis(
    ...     centre_x=50,
    ...     centre_y=50,
    ...     show_progressbar=False)
    >>> fig = femt.plot_fem(s, fem_results, 10, 120)

    """
    if k_cal:
        xaxis = 2 * np.pi * k_cal * \
                np.arange(0, len(results['RadialAvg'].data))
    else:
        xaxis = np.arange(0, len(results['RadialAvg'].data))

    if highcutoff > len(results['RadialAvg'].data):
        highcutoff = len(results['RadialAvg'].data) - 1

    fig, axes = plt.subplots(3, 2, figsize=(9, 12))

    axes[0, 0].imshow(np.log(s.mean().data + 1), cmap='viridis')
    axes[0, 0].set_title('Mean Pattern', size=15)
    axes[0, 0].set_yticks([])
    axes[0, 0].set_xticks([])

    axes[0, 1].plot(xaxis[lowcutoff:highcutoff],
                    results['RadialAvg'].data[lowcutoff:highcutoff],
                    linestyle='', marker='o')
    axes[0, 1].set_title('Mean Radial Profile', size=15)
    axes[0, 1].set_ylabel('Integrated Intensity (counts)')
    axes[0, 1].set_xlabel(r'k ($\AA^{-1}$)')

    axes[1, 0].plot(xaxis[lowcutoff:highcutoff],
                    results['V-Omegak'].data[lowcutoff:highcutoff],
                    linestyle='', marker='o')
    axes[1, 0].set_ylabel(r'V$_\Omega$(k)', size=15)
    axes[1, 0].set_xlabel(r'k ($\AA^{-1}$)')

    axes[1, 1].plot(xaxis[lowcutoff:highcutoff],
                    results['Vrk'].data[lowcutoff:highcutoff],
                    linestyle='', marker='o', label=r'$\overline{V}_r$(k)')

    axes[1, 1].plot(xaxis[lowcutoff:highcutoff],
                    results['Vrek'].data[lowcutoff:highcutoff],
                    linestyle='', marker='o', label=r'V$_{re}$(k)')
    axes[1, 1].legend()
    axes[1, 1].set_ylabel(r'$\overline{V}_r$(k),V$_{re}$(k)', size=15)
    axes[1, 1].set_xlabel(r'k ($\AA^{-1}$)')

    axes[2, 0].imshow(results['Omega-Vi'], cmap='viridis')
    axes[2, 0].set_title('Variance Image', size=15)
    axes[2, 0].set_yticks([])
    axes[2, 0].set_xticks([])

    axes[2, 1].plot(xaxis[lowcutoff:highcutoff],
                    results['Omega-Vk'].data[lowcutoff:highcutoff],
                    linestyle='', marker='o')
    axes[2, 1].set_ylabel(r'$\Omega_V$(k)', size=15)
    axes[2, 1].set_xlabel(r'k ($\AA^{-1}$)')

    fig.tight_layout()
    return fig


def save_fem(results, rootname):
    """Save dictionary of FEM results to individual HDF5 files.

    Parameters
    ----------
    results : Dictionary
        Results of FEM analysis performed on s
    rootname : str
        Root name for output files. '_FEM_Parameter.hdf5' will be appended

    Examples
    --------
    >>> import pixstem.fem_tools as femt
    >>> s = ps.dummy_data.get_fem_signal()
    >>> fem_results = s.fem_analysis(
    ...     centre_x=50,
    ...     centre_y=50,
    ...     show_progressbar=False)
    >>> femt.save_fem(fem_results,'TestData')

    """
    for i in results:
        results[i].save(rootname + '_FEM_' + i + '.hdf5')


def load_fem(rootname):
    """Load individual HDF5 files containing previously saved FEM parameters to
     dictionary.

    Parameters
    ----------
    rootname : str
        Root name for output files. '_FEM_Parameter.hdf5' will be appended

    Returns
    ---------
    results : Dictionary
        Series of HyperSpy signals previously saved using saveFEM

    Examples
    --------
    >>> import pixstem.fem_tools as femt
    >>> s = ps.dummy_data.get_fem_signal()
    >>> fem_results = s.fem_analysis(
    ...     centre_x=50,
    ...     centre_y=50,
    ...     show_progressbar=False)
    >>> femt.save_fem(fem_results,'TestData')
    >>> new_results = femt.load_fem('TestData')

    """
    results = {}
    keys = ['Omega-Vi',
            'Omega-Vk',
            'RadialAvg',
            'RadialInt',
            'V-Omegak',
            'Vrek',
            'Vrk']
    for i in keys:
        results[i] = hs.load(rootname + '_FEM_' + i + '.hdf5')
    return results
