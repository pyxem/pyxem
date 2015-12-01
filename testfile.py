import hyperspy.api as hs
import profiledata
import CoM

im = hs.load("test_hist.hdf5").as_image((0,2)) #load data
im2 = im.to_spectrum() #change signal type
im3 = im2.isig[70.:] #remove centre beam
im4 = im3.remove_background(signal_range = (70.,88.)) #removes background from start of signal to the rough start of the gaussian in the data set. This might need to be inputted by the user.
m = im3.create_model() #turns spectrum into a model

powerlaw= hs.model.components.PowerLaw() #creates a powerlaw
m.append(powerlaw)

m.set_signal_range(70,88) #fits powerlaw to channels 70-88
m.multifit() #fits powerlaw to all images   
powerlaw.set_parameters_not_free() #fixes powerlaw shape

gaussian = hs.model.components.Gaussian()
m.append(gaussian) #create and append gaussian to the model

#Set bounds and initial values for the A,sigma and centre of Gaussian
gaussian.centre.bmin = 95
gaussian.centre.bmax = 115
gaussian.centre.value = 105
gaussian.centre.assign_current_value_to_all()

gaussian.A.bmin = 
gaussian.A.bmax = 
gaussian.A.value=
gaussian.A.assign_current_value_to_all()

gaussian.sigma.bmin = 
gaussian.sigma.bmax = 
gaussian.sigma.value = 
gaussian.sigma.assign_current_value_to_all()

m.multifit(fitter="mpfit", bounded = True) #fit the Gaussian with the aforementioned bounds
gaussian.active = False

im4 = im3 - m.as_signal() 
im5 = im4.inav[:,13:]
