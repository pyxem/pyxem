import higher_order_laue_zone_calculation.stem_diffraction_radial_integration as sdri
import higher_order_laue_zone_calculation.laue_zone_modelling as lzm
import higher_order_laue_zone_calculation.laue_zone_plotting as lzp
import hyperspy.api as hs

def run_full_process_on_fpd_dataset(
        filename,
        crop_dataset=None,
        mrad_per_pixel=None,
        detector_slice=None):
    """Fully process Laue zones diffraction circles from a 
    STEM diffraction 4-D dataset. Does radial integration
    of the diffraction pattern, modelling of the LFO and STO
    Laue zone peaks, and plots the results. This function
    assumes all the default processing values will work,
    which can lead to things going wrong so checking the
    end results looks reasonable is vital. Intermediate
    results are saved as HyperSpy signals.

    Parameters:
    -----------
    filename : string
        Name of the fpd-HDF5 datafile.
    crop_dataset : (float, float, float, float), optional
        If given, the dataset will be cropped in the navigation
        dimensions. If in floats, it will be cropped using
        physical dimensions (nm), if ints in pixels.
        Useful for removing the platinum protective layer.
    detector_slice : (int, int, int, int), optional
        If given, the diffraction images will be cropped
        when trying to find the center of the beam.
        Useful for datasets where the center beam occupies
        a small section of the diffraction pattern.
    """

    path = filename.replace(".hdf5","/")
    s_radial = sdri.get_fpd_dataset_as_radial_profile_signal(
            filename,
            crop_dataset=crop_dataset,
            mrad_per_pixel=mrad_per_pixel,
            detector_slice=detector_slice)
    radial_filename = filename.replace(".hdf5","_radial.hdf5")
    s_radial.save(path + radial_filename, overwrite=True)

    s_lfo = s_radial.isig[47.:78.]
    s_sto = s_radial.isig[77.:107.]

    m_lfo = lzm.model_lfo_with_one_gaussian(s_lfo)
    m_sto = lzm.model_sto(s_sto)

    lfo_model_filename = filename.replace(".hdf5","_lfo_model.hdf5")
    m_lfo.save(path + lfo_model_filename, overwrite=True)
    sto_model_filename = filename.replace(".hdf5","_sto_model.hdf5")
    m_sto.save(path + sto_model_filename, overwrite=True)

    lzp.plot_lfo_sto_laue_zone_report(m_lfo, m_sto, s_radial)
    lzp.plot_lfo_sto_laue_zone_line_profile_report(m_lfo, m_sto, s_radial)

def run_full_process_on_simulated_image(
        filename,
        crop=120.,
        radial_bins=None):
    """Fully process a simulated STEM diffraction
    image, using the same methods as the experimental
    ones. Assumes the image is calibrated to mrad.
    The x- and y-scale does not need to be the same.

    Parameters:
    -----------
    filename : string
        Name of the simulated STEM diffraction image.
        Can be in any format HyperSpy can load, but
        needs to properly calibrated in mrad. With
        the position of the centre of the disk being 
        defined as 0 in both x- and y-directions.
    crop : number, optional
        How much of the diffraction imageshould be
        used in the calculations. Default value is
        120 mrad, which will include the first 
        STO laue circle.
    radial_bins : int or list of scalars, optional
    """
    if radial_bins is None:
        radial_bins = 500
    s_diff = hs.load(filename)
    s_diff = s_diff.isig[
            float(-crop):float(crop),
            float(-crop):float(crop)]
    radial_data, xaxis = sdri._get_radial_profile_of_simulated_image(
            s_diff, bins=radial_bins)

    s_radial = hs.signals.Signal1D(radial_data)
    s_radial.axes_manager[0].scale = xaxis[1]-xaxis[0]
    s_radial.axes_manager[0].offset = xaxis[0]
    s_radial.axes_manager[0].units = s_diff.axes_manager[0].units

    s_lfo = s_radial.isig[47.:78.]
    s_sto = s_radial.isig[77.:107.]

    m_lfo = lzm.model_lfo_with_one_gaussian(
            s_lfo)
    m_sto = lzm.model_sto(s_sto)

    radial_signal_filename = filename.replace(".hdf5","_radial.hdf5")
    s_radial.save(radial_signal_filename, overwrite=True)
    lfo_model_filename = filename.replace(".hdf5","_lfo_model.hdf5")
    m_lfo.save(lfo_model_filename, overwrite=True)
    sto_model_filename = filename.replace(".hdf5","_sto_model.hdf5")
    m_sto.save(sto_model_filename, overwrite=True)
    return(m_lfo, m_sto)
