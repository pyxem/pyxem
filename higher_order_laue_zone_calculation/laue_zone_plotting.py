import hyperspy.api as hs
import matplotlib.pyplot as plt
from matplotlib import gridspec
import glob

def _make_subplot(signal, ax):
    ax.imshow(signal.data,
            interpolation='nearest',
            extent=[
                signal.axes_manager[0].high_value,
                signal.axes_manager[0].low_value,
                signal.axes_manager[1].high_value,
                signal.axes_manager[1].low_value])

def _make_line_subplot(signal, ax, hide_yticks=True):
    ax.plot(signal.data, signal.axes_manager[0].axis)
    ax.set_ylim(
            signal.axes_manager[0].high_value,
            signal.axes_manager[0].low_value)
    if hide_yticks == True:
        ax.set_yticklabels([])
    ax.grid()
    ax.locator_params(nbins=6, tight=True)

def _make_line_standalone_subplot(signal, ax, hide_yticks=True):
    ax.plot(signal.data, signal.axes_manager[0].axis)
    ax.set_ylim(
            signal.axes_manager[0].high_value,
            signal.axes_manager[0].low_value)
    if hide_yticks == True:
        ax.set_yticklabels([])
    ax.grid()
    ax.locator_params(axis='x', nbins=4, tight=True)

def plot_lfo_sto_laue_zone_report(
        m_lfo, 
        m_sto, 
        s_radial,
        adf_radial_slice_list=[
            [50.,90.],[75.,120.],[110.,120.]],
        figname="lfo_sto_laue_zone_report.jpg"):
    """Plot map and line profiles for modelled Laue zone 
    dataset. Plots ADF images from the radial dataset and
    the parameters of the Gaussian used to model the Laue 
    zone peaks. Also includes line profiles of the maps.

    Parameters:
    -----------
    m_lfo : HyperSpy model object
        Model of the LFO Laue zone peak, must contain
        a component called Gaussian.
    m_sto : HyperSpy model object
        Model of the STO Laue zone peak, must contain
        a component called Gaussian.
    s_radial : HyperSpy signal object
        Radially integrated STEM diffraction pattern
        dataset.
    adf_radial_slice_list : list of [float, float], optional
        List of the integration ranges used to generate
        ADF images from the s_radial signal.
    figname : string, optional
        Filename of the figure.
    """


    g_A_lfo = m_lfo['Gaussian'].A.as_signal() 
    g_centre_lfo = m_lfo['Gaussian'].centre.as_signal() 
    g_sigma_lfo = m_lfo['Gaussian'].sigma.as_signal() 

    g_A_lfo_line = g_A_lfo.mean(0)
    g_centre_lfo_line = g_centre_lfo.mean(0)
    g_sigma_lfo_line = g_sigma_lfo.mean(0)

    g_A_sto = m_sto['Gaussian'].A.as_signal() 
    g_centre_sto = m_sto['Gaussian'].centre.as_signal() 
    g_sigma_sto = m_sto['Gaussian'].sigma.as_signal() 

    g_A_sto_line = g_A_sto.mean(0)
    g_centre_sto_line = g_centre_sto.mean(0)
    g_sigma_sto_line = g_sigma_sto.mean(0)

    adf_signal_list = []
    for adf_radial_slice in adf_radial_slice_list:
        s_adf = s_radial.isig[
                adf_radial_slice[0]:adf_radial_slice[1]].mean(-1)
        adf_signal_list.append(s_adf)

    fig = plt.figure(figsize=(25,17))
    gs = gridspec.GridSpec(90,150)

    for index, adf_signal in zip(range(3), adf_signal_list):
        ax_adf = fig.add_subplot(gs[0:30,0+50*index:30+50*index])
        ax_adf_line = fig.add_subplot(gs[0:30,30+50*index:50+50*index])
        _make_subplot(adf_signal, ax_adf)
        _make_line_subplot(adf_signal.mean(0), ax_adf_line)
        ax_adf.set_title("mrad: " + str(adf_radial_slice_list[index]))
        if index == 0:
            ax_adf.set_ylabel("ADF")

    # STO Laue zones
    ax_A_sto = fig.add_subplot(gs[30:60,0:30])
    ax_A_sto_line = fig.add_subplot(gs[30:60,30:50])

    ax_centre_sto = fig.add_subplot(gs[30:60,50:80])
    ax_centre_sto_line = fig.add_subplot(gs[30:60,80:100])

    ax_sigma_sto = fig.add_subplot(gs[30:60,100:130])
    ax_sigma_sto_line = fig.add_subplot(gs[30:60,130:150])

    _make_subplot(g_A_sto, ax_A_sto)
    _make_line_subplot(g_A_sto_line, ax_A_sto_line)

    _make_subplot(g_centre_sto, ax_centre_sto)
    _make_line_subplot(g_centre_sto_line, ax_centre_sto_line)

    _make_subplot(g_sigma_sto, ax_sigma_sto)
    _make_line_subplot(g_sigma_sto_line, ax_sigma_sto_line)

    ax_A_sto.set_title("A")
    ax_centre_sto.set_title("Centre")
    ax_sigma_sto.set_title("Sigma")
    ax_A_sto.set_ylabel("STO")

    # LFO Laue zones
    ax_A_lfo = fig.add_subplot(gs[60:90,0:30])
    ax_A_lfo_line = fig.add_subplot(gs[60:90,30:50])

    ax_centre_lfo = fig.add_subplot(gs[60:90,50:80])
    ax_centre_lfo_line = fig.add_subplot(gs[60:90,80:100])

    ax_sigma_lfo = fig.add_subplot(gs[60:90,100:130])
    ax_sigma_lfo_line = fig.add_subplot(gs[60:90,130:150])

    _make_subplot(g_A_lfo, ax_A_lfo)
    _make_line_subplot(g_A_lfo_line, ax_A_lfo_line)

    _make_subplot(g_centre_lfo, ax_centre_lfo)
    _make_line_subplot(g_centre_lfo_line, ax_centre_lfo_line)

    _make_subplot(g_sigma_lfo, ax_sigma_lfo)
    _make_line_subplot(g_sigma_lfo_line, ax_sigma_lfo_line)

    ax_A_lfo.set_title("A")
    ax_centre_lfo.set_title("Centre")
    ax_sigma_lfo.set_title("Sigma")
    ax_A_lfo.set_ylabel("LFO")

    fig.tight_layout()
    fig.savefig("lfo_sto_laue_zone_report.jpg", dpi=300)

def plot_lfo_sto_laue_zone_line_profile_report(
        m_lfo, 
        m_sto, 
        s_radial,
        adf_radial_slice_list=[
            [50.,90.],[75.,120.],[110.,120.]],
        figname="lfo_sto_laue_zone_line_profile_report.jpg"):
    """Plot line profiles for modelled Laue zone 
    dataset. Plots ADF images from the radial dataset and
    the parameters of the Gaussian used to model the Laue 
    zone peaks. Useful for comparing different onsets.

    Parameters:
    -----------
    m_lfo : HyperSpy model object
        Model of the LFO Laue zone peak, must contain
        a component called Gaussian.
    m_sto : HyperSpy model object
        Model of the STO Laue zone peak, must contain
        a component called Gaussian.
    s_radial : HyperSpy signal object
        Radially integrated STEM diffraction pattern
        dataset.
    adf_radial_slice_list : list of [float, float], optional
        List of the integration ranges used to generate
        ADF images from the s_radial signal.
    figname : string, optional
        Filename of the figure.
    """

    g_A_lfo = m_lfo['Gaussian'].A.as_signal()
    g_centre_lfo = m_lfo['Gaussian'].centre.as_signal() 
    g_sigma_lfo = m_lfo['Gaussian'].sigma.as_signal() 

    g_A_lfo_line = g_A_lfo.mean(0)
    g_centre_lfo_line = g_centre_lfo.mean(0)
    g_sigma_lfo_line = g_sigma_lfo.mean(0)

    g_A_sto = m_sto['Gaussian'].A.as_signal() 
    g_centre_sto = m_sto['Gaussian'].centre.as_signal() 
    g_sigma_sto = m_sto['Gaussian'].sigma.as_signal() 

    g_A_sto_line = g_A_sto.mean(0)
    g_centre_sto_line = g_centre_sto.mean(0)
    g_sigma_sto_line = g_sigma_sto.mean(0)

    adf_signal_list = []
    for adf_radial_slice in adf_radial_slice_list:
        s_adf = s_radial.isig[
                adf_radial_slice[0]:adf_radial_slice[1]].mean(-1)
        adf_signal_list.append(s_adf)

    fig = plt.figure(figsize=(8,16))
    gs = gridspec.GridSpec(90,90)

    for index, adf_signal in zip(range(3), adf_signal_list):
        ax_adf_line = fig.add_subplot(gs[0:30,0+30*index:30+30*index])
        if index == 0:
            _make_line_standalone_subplot(adf_signal.mean(0), ax_adf_line, hide_yticks=False)
        else:
            _make_line_standalone_subplot(adf_signal.mean(0), ax_adf_line)
        ax_adf_line.set_title("mrad: " + str(adf_radial_slice_list[index]))
        if index == 0:
            ax_adf_line.set_ylabel("ADF")

    # STO Laue zones
    ax_A_sto_line = fig.add_subplot(gs[30:60,0:30])
    ax_centre_sto_line = fig.add_subplot(gs[30:60,30:60])
    ax_sigma_sto_line = fig.add_subplot(gs[30:60,60:90])

    _make_line_standalone_subplot(g_A_sto_line, ax_A_sto_line, hide_yticks=False)
    _make_line_standalone_subplot(g_centre_sto_line, ax_centre_sto_line)
    _make_line_standalone_subplot(g_sigma_sto_line, ax_sigma_sto_line)

    ax_A_sto_line.set_title("A")
    ax_centre_sto_line.set_title("Centre")
    ax_sigma_sto_line.set_title("Sigma")
    ax_A_sto_line.set_ylabel("STO")

    # LFO Laue zones
    ax_A_lfo_line = fig.add_subplot(gs[60:90,0:30])
    ax_centre_lfo_line = fig.add_subplot(gs[60:90,30:60])
    ax_sigma_lfo_line = fig.add_subplot(gs[60:90,60:90])

    _make_line_standalone_subplot(g_A_lfo_line, ax_A_lfo_line, hide_yticks=False)
    _make_line_standalone_subplot(g_centre_lfo_line, ax_centre_lfo_line)
    _make_line_standalone_subplot(g_sigma_lfo_line, ax_sigma_lfo_line)

    ax_A_lfo_line.set_title("A")
    ax_centre_lfo_line.set_title("Centre")
    ax_sigma_lfo_line.set_title("Sigma")
    ax_A_lfo_line.set_ylabel("LFO")
    
    for ax in fig.axes:
        ax.axhline(11, color='red')
    
    fig.tight_layout()
    fig.savefig("lfo_sto_laue_zone_line_profile_report.jpg", dpi=300)

def plot_sto_lfo_simulated_report_from_model_files(filename_list=None):
    fig = plt.figure(figsize=(8,16))
    gs = gridspec.GridSpec(90,90)

    ax_A_sto = fig.add_subplot(gs[30:60,0:30])
    ax_centre_sto = fig.add_subplot(gs[30:60,30:60])
    ax_sigma_sto = fig.add_subplot(gs[30:60,60:90])

    ax_A_lfo = fig.add_subplot(gs[60:90,0:30])
    ax_centre_lfo = fig.add_subplot(gs[60:90,30:60])
    ax_sigma_lfo = fig.add_subplot(gs[60:90,60:90])

    if filename_list == None:
        filename_list = glob.glob("*_model.hdf5")
        filename_list.sort()

    for index, filename in enumerate(filename_list):
        if "*_LFO_model.hdf5" in filename:
            s = hs.load(filename)
            m = s.models.a.restore()

            g_A = m['Gaussian'].A.as_signal()
            g_centre = m['Gaussian'].centre.as_signal() 
            g_sigma = m['Gaussian'].sigma.as_signal() 

            ax_A_lfo.plot(index, g_A.data)
            ax_centre_lfo.plot(index, g_A.data)
            ax_sigma_lfo.plot(index, g_A.data)
