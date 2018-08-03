import matplotlib.pylab as plt
import numpy as np
import hyperspy.api as hspy

'''
Plotting function used to display output of FEM calculations
'''


def plotFEM(s, results, lowcutoff=10, highcutoff=120, k_cal=None):
    """Produce a summary plot of all calculated FEM parameters

    Parameters
    ----------
    s : PixelatedSTEM
        Signal on which FEM analysis was performed
    results : Dictionary
        Series of Hyperspy Signals containing results of FEM analysis performed on s
    lowcutoff : integer
        Position of low-q cutoff for plots
    highcutoff : integer
        Position of high-q cutoff for plots

    Returns
    -------
    fig : Matplotlib Figure

    Examples
    --------
    >>> import fpd_data_processing.fem_tools as femt
    >>> fig = femt(s,results,10,120)
    >>> fig.savefig('Output.png')

    """

    if k_cal:
        xaxis = 2 * np.pi * k_cal * np.arange(0, len(results['RadialAvg'].data))
    else:
        xaxis = np.arange(0, len(results['RadialAvg'].data))

    fig, axes = plt.subplots(3, 2, figsize=(9, 12))

    axes[0, 0].imshow(np.log(s.mean().data + 1), cmap='viridis')
    axes[0, 0].set_title('Mean Pattern', size=15)
    axes[0, 0].set_yticks([])
    axes[0, 0].set_xticks([])

    axes[0, 1].plot(xaxis[lowcutoff:highcutoff], results['RadialAvg'].data[lowcutoff:highcutoff], linestyle='',
                    marker='o')
    axes[0, 1].set_title('Mean Radial Profile', size=15)
    axes[0, 1].set_ylabel('Integrated Intensity (counts)')
    axes[0, 1].set_xlabel(r'k ($\AA^{-1}$)')

    axes[1, 0].plot(xaxis[lowcutoff:highcutoff], results['V-Omegak'].data[lowcutoff:highcutoff], linestyle='',
                    marker='o')
    axes[1, 0].set_ylabel(r'V$_\Omega$(k)', size=15)
    axes[1, 0].set_xlabel(r'k ($\AA^{-1}$)')

    axes[1, 1].plot(xaxis[lowcutoff:highcutoff], results['Vrk'].data[lowcutoff:highcutoff],
                    linestyle='', marker='o', label=r'$\overline{V}_r$(k)')
    axes[1, 1].plot(xaxis[lowcutoff:highcutoff], results['Vrek'].data[lowcutoff:highcutoff],
                    linestyle='', marker='o', label=r'V$_{re}$(k)')
    axes[1, 1].legend()
    axes[1, 1].set_ylabel(r'$\overline{V}_r$(k),V$_{re}$(k)', size=15)
    axes[1, 1].set_xlabel(r'k ($\AA^{-1}$)')

    axes[2, 0].imshow(results['Omega-Vi'], cmap='viridis')
    axes[2, 0].set_title('Variance Image', size=15)
    axes[2, 0].set_yticks([])
    axes[2, 0].set_xticks([])

    axes[2, 1].plot(xaxis[lowcutoff:highcutoff], results['Omega-Vk'].data[lowcutoff:highcutoff], linestyle='',
                    marker='o')

    plt.tight_layout()

    return (fig)


def saveFEM(results, rootname):
    """Save dictionary of FEM results to individual HDF5 files.

    Parameters
    ----------
    results : Dictionary
        Results of FEM analysis performed on s
    rootname : str
        Root name for output files. '_FEM_Parameter.hdf5' will be appended

    """

    for i in results:
        results[i].save(rootname + '_FEM_' + i + '.hdf5')
    return


def loadFEM(rootname):
    """Load individual HDF5 files containing previously saved FEM parameters to dictionary.

    Parameters
    ----------
    rootname : str
        Root name for output files. '_FEM_Parameter.hdf5' will be appended

    Returns
    ---------
    results : Dictionary
        Series of Hyperspy signals previously saved using saveFEM
    """

    results = {}
    keys = ['Omega-Vi', 'Omega-Vk', 'RadialAvg', 'RadialInt', 'V-Omegak', 'Vrek', 'Vrk']
    for i in keys:
        results[i] = hspy.load(rootname + '_FEM_' + i + '.hdf5')
    return results

