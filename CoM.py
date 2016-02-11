import hyperspy.api as hs
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import h5py
import copy
from numpy import unravel_index
from scipy.optimize import curve_fit

def loadh5py(i):
	fpdfile = h5py.File(i,'r') #find data file in a read only format
	data = fpdfile['fpd_expt']['fpd_data']['data'][:]
	im = hs.signals.Image(data[:,:,0,:,:])
	return im

def _dataset_dimensions(dataset):
    summed_signal = dataset.sum(-1).sum(-1).data
    dataset_shape = np.shape(summed_signal)
    return dataset_shape
    
#Centre of Mass calculation with thresholding from Magnus
def centre_of_disk_centre_of_mass(
        image,
        threshold=None):
    if threshold == None:
#            threshold = (image.max()-image.min())*0.5
        threshold = np.mean(image) * (45/2)
#       The mean of the data set being used to write this is roughly
#       1/45 the max value. This threshold is too low and means that
#       Lower intensity features are interfering with the creation
#       Of the boolean disk
        
    image[image<threshold] = 0
    image[image>threshold] = 1
    booleanArray = image.astype(bool)
    disk_centre = sp.ndimage.measurements.center_of_mass(booleanArray)

    return(disk_centre)

def centre_of_disk_centre_of_mass_full(
        image, data_file, size):
    disk_centre=[]
    for i in range(0,size[0]):
        for j in range(0,size[1]):
            imagedata = image[i,j].data.astype('float32')
            centre = sp.ndimage.measurements.center_of_mass(imagedata)
            centre2 = [centre[1],centre[0]]
            
            disk_centre.append(centre2)
    disk_centre_array = np.reshape(disk_centre, (size[0],size[1],2))
    np.save(data_file.replace(".hdf5", "_CoM_centre"), disk_centre_array)
    return(disk_centre_array)
          

#centre profile calculation from StackOverflow
def radial_profile(data,centre):
    y, x = np.indices((data.shape))
    r = np.sqrt((x - centre[0])**2 + (y - centre[1])**2)
    r = r.astype(int)
    r_max = np.max(r)
    
    tbin =  np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())   
    radialProfile = tbin / nr

    return radialProfile  
    	   
def diffraction_calibration(centre, size, im = None, scale = None):
#	This function calibrates the diffraction patters
#   It takes the raw data input & crops the data set to the last 14 rows
#	This assumes that bulk STO is all that exists in these 14 rows. It then sums
#	Over the 0th and 1st axes into a single	.tif image that we can put into ImageJ
#   To get a "scale". It then applies the calibration to the data and returns it
#   Unchanged other than the addition of the calibration
	
#   If no data is inputted, it asks the user for the name of the data file
	if im is None:	
		data = input("Please enter the name of data file as a string: ")
		im = loadh5py(data)

#   If no scale is inputted, the data is converted to a .tif image and saved
	if scale is None:
		rawSTO = im[:,50:]
		rawSumSTO = rawSTO.sum(0).sum(0)
		saveName = input("Please enter the name of the save file as a string: ")
		rawSumSTO32 = rawSumSTO.change_dtype('float32')
		rawSumSTO32.save(saveName + ".tif")
		print("Please use ImageJ to load the .tif file and find the radius of the STO Laue zone")
		print("The diameter of the STO Laue zone is 133.32 mrad")
		scale = input("Scale found from ImageJ: ")
	
# 	This takes the input of the number of pixels per "unit" and applies this calibration to
#	The dataset and centres the axes
    
	if type(centre) is tuple:
	    a2 = im.axes_manager[2]
	    a2.offset = -centre[0]
	    a2.scale = scale
	    a2.units = "mrad"
	    
	    a3 = im.axes_manager[3]
	    a3.offset = -centre[1]
	    a3.scale = scale
	    a3.units = "mrad"
	    
	else:
	    for i in range(0,size[0]):
	        for j in range (0,size[1]):
	            a2 = im[i,j].axes_manager[0]
	            a2.offset = - centre[i,j][0]
	            a2.scale = scale
	            a2.units = "mrad"
	            
	            a3 = im[i,j].axes_manager[1]
	            a3.offset = centre[i,j][1]
	            a3.scale = scale
	            a3.units = "mrad"

	return im 
   
def radial_profile_dataset(im, centre, save, size, flip = True):
#   Creates a hfd5 of the summed radial profiles for an input
#   hfd5 file
#   im is the dataset to be profiled
#   centre is the centre of each image in the dataset
#   save is the name that the profile will be saved under
#   flip is whether there is a single centre value or an array of centre values
#   rotation is whether the dataset is rotated or not

    
    rad_size = [] 
#   the length of rad varies and a numpy array needs a 
#   fixed row length for each entry

    for i in range(0,size[0]):
        for j in range (0,size[1]):
#   Iterates over each individual 256x256 image

            if flip == True:
                rad = radial_profile(im[i,j].data, centre[i,j])
            else: rad = radial_profile(im[i,j].data, centre)
            rad_length = len(rad)
            rad_size.append(rad_length)

    min_rad_size = min(rad_size)
    del rad_size
    blankFileToSave = np.zeros((size[0],size[1],min_rad_size))
    for i in range(0,size[0]):
        for j in range(0,size[1]):
            
            if flip == True:
                rad = radial_profile(im[i,j].data, centre[i,j])
            else: rad = radial_profile(im[i,j].data, centre)
            
            if len(rad) > min_rad_size:
                while len(rad) > min_rad_size:
                    rad = np.delete(rad, [len(rad)-1])

#           Shortens the array of rad to create a uniform 
#           Length          
          
            blankFileToSave[i,j] = rad

#           Passes the numpy array for each image alone with the
#           Average centre of mass calculated earlier to the 
#           Radial profiler in CoM and saves the profile data
#           In a multidimensional numpy array

    s = hs.signals.Image(blankFileToSave)
    s.save(save)        
    np.save(save, blankFileToSave)
#   Changes the numpy array with the profiles in it into a hyperspy
#   signal and saves it based on the chosen save name
    return blankFileToSave
    	
	       
def gauss(x, *p):
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))

def square(list): return[i**2 for i in list]
def cube(list): return[i**3 for i in list]
def fourth(list): return[i**4 for i in list]
def multiple(poly, list): return[poly*i for i in list]

def _powerlaw_fitter(data_file, profile_dataset, save_name, bounded_gaussian = True, threshold = True, save_file = True):
    powerlaw_fit_exist = input("Is there a powerlaw file already created? 1/0: ")
    
    im = hs.load(profile_dataset + data_file).as_image((0,2)) #load data
    
    im2 = im.to_spectrum() #change signal type
    
    central_beam_image = im2[50,50]
    central_beam_plot = central_beam_image.plot()
    plt.show()
    central_beam = input("Please look at the generated image and enter a rough end for the central beam: ")
    wait = 0
    while wait == False:
        wait = str(input("Enter 1 when all images closed: "))
    
    im3 = im2.isig[central_beam:] #crop out centre beam
    im3.sum(-1).plot()
    plt.show()
        
    platinum_range = input("Please look at the image and enter the row (y value) where the platinum roughly ends: ")
    wait = 0
    while wait == False:
        wait = input("Enter 1 when all images closed: ")
    
    im4 = im3.inav[:,platinum_range:] #remove platinum
    m = im4.create_model() #turns spectrum into a model
    
################################
    m.plot()
    plt.show()
    
    powerlaw = hs.model.components.PowerLaw() #creates a powerlaw
    m.append(powerlaw)
    
    start_powerlaw = central_beam
    end_powerlaw = float(input("Please look at the model and enter the end signal range for the power law background by navigating to an image with a peak: "))

    wait = 0
    while wait == False:
        wait = str(input("Enter 1 when all images closed: "))
    
    if powerlaw_fit_exist == False:
        m.set_signal_range(start_powerlaw,end_powerlaw)
        m.multifit() #fits powerlaw to all images 
        m.reset_signal_range() #reset signal range
        powerlaw.set_parameters_not_free() #fixes powerlaw shape

        powerlaw_fit = im4 - m.as_signal() #Removes the powerlaw as background
        powerlaw_fit.save(save_name + data_file)
    
    else:
        powerlaw_fit = hs.load(save_name + data_file)

#def _gaussian_fitter(powerlaw_fit, bounded_gaussian=True):
    
    gaussian_fit_exist = input("Has the Gaussian fit been performed already? 1/0: "
    
    if gaussian_fit_exist == False:
        gaussian = hs.model.components.Gaussian() #creates a gaussian component

################################
# Finds the probe position with the highest total intensity which 
# Will be used to set the initial conditions for the Gaussian fit
#    powerlaw_fit = hs.load(powerlaw_fit)
        peak_intensities = powerlaw_fit.sum(-1)
        maxSumIndex = unravel_index(peak_intensities.data.argmax(),peak_intensities.data.shape)
        maxSignalPosition = powerlaw_fit[maxSumIndex[1],maxSumIndex[0]]

        max_gauss_model = maxSignalPosition.create_model()
        max_gauss_model.append(gaussian)
        max_gauss_model.plot()
        plt.show()
    
        start_gaussian = float(input("Please look at the model and enter the start signal range for the Gaussian signal fit (don't close image yet): "))
        end_gaussian = float(input("Please look at the model and enter the end signal range for the Gaussian signal fit: "))
    
        max_gauss_model.set_signal_range(start_gaussian, end_gaussian)
        gaussian.centre.value = (start_gaussian + end_gaussian)/2.
        max_gauss_model.fit(fitter = "mpfit")
    
        print gaussian.centre.value
        print gaussian.sigma.value
        print gaussian.A.value
        current_centre = gaussian.centre.value
        current_A = gaussian.A.value
        current_sigma = gaussian.sigma.value
        del max_gauss_model
        del gaussian
        

################################
# Fit a Gaussian to the whole data set using the values from the 
# Strongest signal as initial conditions and apply bounds on the 
# Extremes
        gaussian = hs.model.components.Gaussian()
        gauss_model = powerlaw_fit.create_model() 
        gauss_model.append(gaussian) 
        gauss_model.set_signal_range(start_gaussian, end_gaussian)
    
        centreMin = start_gaussian
        centreMax = end_gaussian
        gaussian.centre.bmin = centreMin
        gaussian.centre.bmax = centreMax
        gaussian.centre.value = current_centre
        gaussian.centre.assign_current_value_to_all()
    
        gaussian.A.bmin = 0.
        gaussian.A.value = current_A
        gaussian.A.assign_current_value_to_all()
    
        gaussian.sigma.bmin = 0.1
        gaussian.sigma.bmax = 15.
        gaussian.sigma.value = current_sigma
        gaussian.sigma.assign_current_value_to_all()
    
        wait = 0
        while wait == False:
            wait = str(input("Enter 1 after closing all images: "))
    
        if bounded_gaussian == True:
            gauss_model.multifit(fitter="mpfit", bounded = True)
        else: gauss_model.multifit(fitter="mpfit", bounded = False) 
        gauss_model.reset_signal_range()
    
####    ###########################
    
        centreGaussian = gaussian.centre.as_signal() #create a signal of all centre values
        aGaussian = gaussian.A.as_signal() # create a signal of all A values
        sigmaGaussian = gaussian.sigma.as_signal() #create a signal of all sigma values
    
        
        if save_file == True:
            centreGaussian.save("centre_signal_" + data_file)        
            aGaussian.save("a_signal_" + data_file)
            sigmaGaussian.save("sigma_signal_" + data_file)
#            signals = centreGaussian, aGaussian, sigmaGaussian
#            h = h5py.File('Gaussian_signals_' + data_file, 'w')
#            h.create_dataset('Gaussian_signals_' + data_file.remove(".hdf5"), data = signals)

    else:
        centreGaussian  = hs.load("centre_signal_" + data_file)    
        aGaussian       = hs.load("a_signal_" + data_file)
        sigmaGaussian   = hs.load("sigma_signal_" + data_file)
        peak_intensities = powerlaw_fit.sum(-1)
        maxSumIndex = unravel_index(peak_intensities.data.argmax(),peak_intensities.data.shape)
        maxSignalPosition = powerlaw_fit[maxSumIndex[1],maxSumIndex[0]]

        max_gauss_model = maxSignalPosition.create_model()
        max_gauss_model.append(gaussian)
        max_gauss_model.plot()
        plt.show()
        end_gaussian = float(input("Please look at the model and enter the end signal range for the Gaussian signal fit: "))
        
    if threshold == True:
        sigma_data = copy.deepcopy(sigmaGaussian.data)
        centre_data = copy.deepcopy(centreGaussian.data)
        a_data = copy.deepcopy(aGaussian.data)
        sigma_data[sigma_data >= gaussian.sigma.bmax] = 0.
        centre_data[sigma_data == 0.] = 0.
        centre_data[sigma_data >= gaussian.sigma.bmax] = 0.
        centre_data[centre_data >= centreMax] = 0.
        centre_data[centre_data <= centreMin] = 0.
        centre_data[a_data <= gaussian.A.bmin] = 0.
            
        centreGaussian.data[centre_data == 0.] = 0. #apply the thresholded data map to the original signals
        aGaussian.data[centre_data == 0.] = 0. 
        sigmaGaussian.data[centre_data == 0.] = 0.
        del centre_data
        del sigma_data
        del a_data
#            signals = centreGaussian, aGaussian, sigmaGaussian
#            h = h5py.File('Gaussian_signals_thresh_' + data_file, 'w')
#            h.create_dataset('Gaussian_signals_thresh' + data_file.remove(".hdf5"), data = signals)
    
####    ##########################          
    
    
    return [aGaussian, centreGaussian, sigmaGaussian], [end_powerlaw, end_gaussian], peak_intensities, platinum_range

##############################
    
def _adf_step(im, step_ends):
    
#    adf_step = input("Please enter the start of the adf: ")
#    start_step = input("Please enter a value before the HOLZ: ")
#    wait = False
#    while wait == False:
#        wait = input("Please enter 1 when all images closed: ")
    adf_step = step_ends[1]
    start_step = step_ends[0]
    step_size = (adf_step - start_step) / 5
    step_array = [start_step]
    for i in range(1,6):
        next_step = start_step + (i*step_size)
        step_array.append(next_step)
        print step_array
    step_array.append(adf_step)
    return step_array 
    
    
    
def _clim_set(image_array):
    a_gauss     = image_array[6][0][0]
    centre_gauss = image_array[6][0][1]
    sigma_gauss = image_array[6][0][2]
    a_mean      = image_array[6][1]
    centre_mean = image_array[6][2]
    sigma_mean  = image_array[6][3]
    
    fig, axarr = plt.subplots(1, 3, figsize = (15,5))
    
    ax_centre_gauss = axarr[0]
    c_centre_gauss = ax_centre_gauss.imshow(centre_gauss.data)
    ax_centre_gauss.set_title("Gaussian Centre")
    fig.colorbar(c_centre_gauss, ax = ax_centre_gauss)
    
    ax_sigma_gauss  = axarr[1]
    c_sigma_gauss = ax_sigma_gauss.imshow(sigma_gauss.data)
    ax_sigma_gauss.set_title("Gaussian Sigma")
    fig.colorbar(c_sigma_gauss, ax = ax_sigma_gauss)

    ax_a_gauss      = axarr[2]
    c_a_gauss = ax_a_gauss.imshow(a_gauss.data)
    ax_a_gauss.set_title("Gaussian A")
    fig.colorbar(c_a_gauss, ax = ax_a_gauss)
        
    fig.show()
    
    centre_clim = input("Please enter values to view centre over as min,max: ")
    print centre_clim
    sigma_clim  = input("Please enter values to view sigma over as min,max: ")
    a_clim      = input("Please enter a minimum value to view A over: ")
    
    wait = 0
    while wait == False:
        wait = str(input("Enter 1 after closing all images: "))
    
    end = 0
    while end == False:
        fig, axarr = plt.subplots(1, 3, figsize = (15,5))
       
        ax_centre_gauss = axarr[0]
        c_centre_gauss = ax_centre_gauss.imshow(centre_gauss.data)
        c_centre_gauss.set_clim(centre_clim)
        ax_centre_gauss.set_title("Gaussian Centre")
        fig.colorbar(c_centre_gauss, ax = ax_centre_gauss)
        
        ax_sigma_gauss  = axarr[1]
        c_sigma_gauss = ax_sigma_gauss.imshow(sigma_gauss.data)
        c_sigma_gauss.set_clim(sigma_clim)
        ax_sigma_gauss.set_title("Gaussian Sigma")
        fig.colorbar(c_sigma_gauss, ax = ax_sigma_gauss)
        
        ax_a_gauss      = axarr[2]
        c_a_gauss = ax_a_gauss.imshow(a_gauss.data)
        c_a_gauss.set_clim(a_clim)
        ax_a_gauss.set_title("Gaussian A")
        fig.colorbar(c_a_gauss, ax = ax_a_gauss)        
        fig.show()
            
        end = input("Is this clim set ok? 0/1: ")
        if end == False:
            centre_clim = input("Please enter values to view centre over as min,max: ")
            sigma_clim  = input("Please enter values to view sigma over as min,max: ")
            a_clim      = input("Please enter a minimum value to view A over: ")
        
    return [centre_clim, sigma_clim, a_clim]

def _annular_dark_field_image(radial_dataset, size, start_point, end_point = None):
#   Returns a numpy array of the ADF that can be displayed as an image.
#   Input is the radially profiled dataset, not a HDF5 file.
    results = []
    for i in range (0,size[0]):
        for j in range (0,size[1]):
            radial_profile = radial_dataset[i,j]
            if end_point == None:
                 end_point = len(radial_profile)
            sum_range = sp.integrate.simps(radial_profile[start_point:end_point])
            results.append(sum_range)
    adf_array = np.array(results)
    adf_array = adf_array.reshape(size[0],size[1])
    del results
    return adf_array

def _gaussian_parameter_mean(gaussian_signals_rotated):
    a_gaussian      = gaussian_signals_rotated[0][0]
    centre_gaussian = gaussian_signals_rotated[0][1]
    sigma_gaussian  = gaussian_signals_rotated[0][2]  

    a_mean          = a_gaussian.data.mean(axis = 1)
    centre_mean     = centre_gaussian.data.mean(axis = 1) #find the means along the axis perp. to growth direction
    sigma_mean      = sigma_gaussian.data.mean(axis = 1)
    
    return [a_mean], [centre_mean], [sigma_mean] 
    # Fitting the data with Gaussians and polynomials, commented out for now.
    '''
#pInitial = [114., 15., 8.]
    pInitial = [50.,15.,5.]
    lengthOfFitGauss = list(range(len(a_mean)))
    coeffGauss, varmatrix = curve_fit(gauss, lengthOfFitGauss, a_mean, p0=pInitial)
    dataFitGauss = gauss(lengthOfFitGauss, *coeffGauss)

#plt.figure(0)
#A_mean_image = plt.plot(lengthOfFitGauss, a_mean, 'ro', label='Data')
#plt.plot(lengthOfFitGauss, dataFitGauss, 'bo', label='Fit')
#plt.show()
#plt.legend()

#print 'Most intense Laue zone at row ', coeffGauss[1]
#print 'Sigma = ', coeffGauss[2]


#############################
#Fit a polynomial to the thresholded centre_mean then translate that back to centre_mean

    lengthOfFitPoly = list(range(len(centre_mean_thresholded)))
    poly = np.polynomial.polynomial.polyfit(lengthOfFitPoly, centre_mean_thresholded, 4, rcond=None, full=False, w=None)

    polyFit = poly[0] +multiple(poly[1], lengthOfFitPoly) + multiple(poly[2],square(lengthOfFitPoly)) + multiple(poly[3], cube(lengthOfFitPoly)) + multiple(poly[4], fourth(lengthOfFitPoly))

#plt.figure(1)
#centre_mean_image = plt.plot(lengthOfFitPoly,centre_mean_thresholded,'ro', label = 'Data')
#plt.plot(lengthOfFitPoly, polyFit, 'bo', label = 'Fit')
#plt.show()
#plt.legend()



# Changes polyFit into a comparable dataset to centre_mean

    preLaueIndex = [n for n, i in enumerate(centre_mean) if i >= centreMin][0]
    postLaueIndex = preLaueIndex + len(centre_mean_thresholded)
    centre_meanPoly = [0.] * (preLaueIndex)
    listOfZeros = [0.] * (len(centre_mean) - postLaueIndex)
    centre_meanPoly.extend(polyFit)
    centre_meanPoly.extend(listOfZeros)
    '''
#plt.plot(lengthOfFitGauss, centre_meanPoly, 'bo', label = 'Data')
#plt.plot(lengthOfFitGauss, centre_mean, 'ro', label = 'Fit')
#plt.show()
#plt.legend() 

def _dataset_rotation(image_array):
    angle_of_rotation = input("What is the angle of rotation?: ")
    rotated_images = sp.ndimage,interpolation.rotate(image_array, angle_of_rotation)
    return rotated_images
    
def _figure_compare(image_array, centring_method, clim_set, bounded_gaussian = True):
#   Input an array of images to be displayed. Initally lets assume we are going to have 14 images.     
    
    adf         = image_array[0]
    adf_step_1  = image_array[1]
    adf_step_2  = image_array[2]
    adf_step_3  = image_array[3]
    adf_step_4  = image_array[4]
    adf_step_5  = image_array[5]
    a_gauss     = image_array[6][0][0]
    centre_gauss = image_array[6][0][1]
    sigma_gauss = image_array[6][0][2]
    a_mean      = image_array[6][1]
    centre_mean = image_array[6][2]
    sigma_mean  = image_array[6][3]
    peak_intensities = image_array[6][5]
    
     

    fig, axarr = plt.subplots(5,3, figsize = (20,20))
    ax_adf          = axarr[0][0]
    ax_adf.imshow(adf)
    ax_adf.set_title("ADF Image post HOLZ")
    ax_adf_step_1   = axarr[0][1]
    ax_adf_step_1.imshow(adf_step_1)
    ax_adf_step_1.set_title("ADF steps through HOLZ (1)")
    ax_adf_step_2   = axarr[0][2]
    ax_adf_step_2.imshow(adf_step_2)
    ax_adf_step_2.set_title("ADF steps through HOLZ (2)")
    ax_adf_step_3   = axarr[1][0]
    ax_adf_step_3.imshow(adf_step_3)
    ax_adf_step_3.set_title("ADF steps through HOLZ (3)")
    ax_adf_step_4   = axarr[1][1]
    ax_adf_step_4.imshow(adf_step_4)
    ax_adf_step_4.set_title("ADF steps through HOLZ (4)")
    ax_adf_step_5   = axarr[1][2]
    ax_adf_step_5.imshow(adf_step_5)
    ax_adf_step_5.set_title("ADF steps through HOLZ (5)")
    
    ax_a_gauss      = axarr[2][0]
    c_a_gauss = ax_a_gauss.imshow(a_gauss.data)
    c_a_gauss.set_clim(clim_set[2])
    ax_a_gauss.set_title("Gaussian A")
    fig.colorbar(c_a_gauss, ax = ax_a_gauss)
    
    ax_centre_gauss = axarr[2][1]
    c_centre_gauss = ax_centre_gauss.imshow(centre_gauss.data)
    c_centre_gauss.set_clim(clim_set[0][0], clim_set[0][1])
    ax_centre_gauss.set_title("Gaussian Centre")
    fig.colorbar(c_centre_gauss, ax = ax_centre_gauss)
    
    ax_sigma_gauss  = axarr[2][2]
    c_sigma_gauss = ax_sigma_gauss.imshow(sigma_gauss.data)
    c_sigma_gauss.set_clim(clim_set[1][0], clim_set[1][1])
    ax_sigma_gauss.set_title("Gaussian Sigma")
    fig.colorbar(c_sigma_gauss, ax = ax_sigma_gauss)
    
    ax_a_mean       = axarr[3][0]
    ax_a_mean.plot(a_mean[0], 'ro', label = "Mean Gaussian A")
    ax_a_mean.legend()
    
    ax_centre_mean  = axarr[3][1]
    ax_centre_mean.plot(centre_mean[0], 'ro', label = "Mean Gaussian Centre")
    ax_centre_mean.legend()
    
    ax_sigma_mean   = axarr[3][2]
    ax_sigma_mean.plot(sigma_mean[0], 'ro', label = "Mean Gaussian Sigma")
    ax_sigma_mean.legend()
    
    ax_peak_intensities = axarr[4][0]
    c_peak_intensities = ax_peak_intensities.imshow(peak_intensities.data)
    ax_peak_intensities.set_title("Summed intensities")

    fig.tight_layout()
    if bounded_gaussian == True:
        fig.savefig("bounded_compare_images_" + centring_method + ".jpg")
    else: 
        fig.savefig("unbounded_compare_images_" + centring_method + ".jpg")

def _flip_transpose (image_array):
#   Flips and transposes the image. For some reason, the profiler puts the platinum on the left hand side instead of the top
    image_flipped_transposed = np.transpose(image_array)
    return image_flipped_transposed        

def _dark_field_array(radial_profiled_dataset, step_array, size, platinum):
#   Integrates the profiled dataset in different steps throught the ADF and HOLZ
    #radial_profiled_dataset = radial_profiled_dataset[13:,:]
    adf_array  = _flip_transpose(_annular_dark_field_image(radial_profiled_dataset, size, step_array[5], end_point = None))[platinum:,:]
    adf_step_1 = _flip_transpose(_annular_dark_field_image(radial_profiled_dataset, size, step_array[0], step_array[1]))[platinum:,:]
    adf_step_2 = _flip_transpose(_annular_dark_field_image(radial_profiled_dataset, size, step_array[1], step_array[2]))[platinum:,:]
    adf_step_3 = _flip_transpose(_annular_dark_field_image(radial_profiled_dataset, size, step_array[2], step_array[3]))[platinum:,:]
    adf_step_4 = _flip_transpose(_annular_dark_field_image(radial_profiled_dataset, size, step_array[3], step_array[4]))[platinum:,:]
    adf_step_5 = _flip_transpose(_annular_dark_field_image(radial_profiled_dataset, size, step_array[4], step_array[5]))[platinum:,:]
    return [adf_array, adf_step_1, adf_step_2, adf_step_3, adf_step_4, adf_step_5]    
