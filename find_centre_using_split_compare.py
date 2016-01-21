import CoM
import copy
import numpy as np
import matplotlib.pyplot as plt

def _plot_compare_around_debug(diff_compare_array):
    
    diff0 = diff_compare_array[0]
    diff1 = diff_compare_array[1]
    diff_original = diff_compare_array[2]

    fig, axarr = plt.subplots(3,1, figsize=(3,8))
    axarr[0].imshow(diff0) 
    axarr[1].imshow(diff1)
    axarr[2].imshow(diff_original)

    fig.tight_layout()
    fig.savefig("split_images.jpg")

def find_centre_using_split_top(image, centre, sub_image_size):
    
    image_original_centre = (centre[0], centre[1])
    image_0_centre = (centre[0]-1, centre[1])
    image_1_centre = (centre[0]+1, centre[1])
    
    image_top_original = image[round(centre[0])-sub_image_size:round(centre[0]),round(centre[1])-sub_image_size:round(centre[1])+sub_image_size].data
    image_top_0 = image[round(centre[0])-sub_image_size-1:round(centre[0])-1,round(centre[1])-sub_image_size:round(centre[1])+sub_image_size].data
    image_top_1 = image[round(centre[0])-sub_image_size+1:round(centre[0])+1,round(centre[1])-sub_image_size:round(centre[1])+sub_image_size].data

    image_bottom_original = image[round(centre[0]):round(centre[0])+sub_image_size,round(centre[1])-sub_image_size:round(centre[1])+sub_image_size].data
    image_bottom_0 = image[round(centre[0])-1:round(centre[0])+sub_image_size-1,round(centre[1])-sub_image_size:round(centre[1])+sub_image_size].data
    image_bottom_1 = image[round(centre[0])+1:round(centre[0])+sub_image_size+1,round(centre[1])-sub_image_size:round(centre[1])+sub_image_size].data

    image_diff_original = np.flipud(image_top_original) - image_bottom_original
    image_diff_0 = np.flipud(image_top_0) - image_bottom_0
    image_diff_1 = np.flipud(image_top_1) - image_bottom_1
    
    diff_sum_original = np.abs(image_diff_original).sum()
    diff_sum_0 = np.abs(image_diff_0).sum()
    diff_sum_1 = np.abs(image_diff_1).sum()
    
    diff_array = [diff_sum_original, diff_sum_0, diff_sum_1]
    
    print(diff_sum_original, diff_sum_0, diff_sum_1)   

    min_diff_array = np.min(diff_array)

    if min_diff_array == diff_sum_original:
        return image_original_centre, True
    
#   runs else statements to find which image currently has the best symmetry and return a new centre value 
    elif diff_sum_0 == min_diff_array:
        return image_0_centre, False
    else:
        return image_1_centre, False
  
def find_centre_using_split_left(image, centre, sub_image_size):
    
    image_original_centre = (centre[0], centre[1])
    image_0_centre = (centre[0], centre[1]-1)
    image_1_centre = (centre[0], centre[1]+1)
    
    image_left_original = image[round(centre[0])-sub_image_size:round(centre[0])+sub_image_size,round(centre[1]):round(centre[1])+sub_image_size].data
    image_left_0 = image[round(centre[0])-sub_image_size:round(centre[0])+sub_image_size,round(centre[1])-1:round(centre[1])+sub_image_size-1].data
    image_left_1 = image[round(centre[0])-sub_image_size:round(centre[0])+sub_image_size,round(centre[1])+1:round(centre[1])+sub_image_size+1].data
    
    image_right_original = image[round(centre[0])-sub_image_size:round(centre[0])+sub_image_size,round(centre[1])-sub_image_size:round(centre[1])].data
    image_right_0 = image[round(centre[0])-sub_image_size:round(centre[0])+sub_image_size,round(centre[1])-sub_image_size-1:round(centre[1])-1].data
    image_right_1 = image[round(centre[0])-sub_image_size:round(centre[0])+sub_image_size,round(centre[1])-sub_image_size+1:round(centre[1])+1].data
    
    image_diff_original = np.fliplr(image_left_original) - image_right_original
    image_diff_0 = np.fliplr(image_left_0) - image_right_0
    image_diff_1 = np.fliplr(image_left_1) - image_right_1
    
    diff_sum_original = np.abs(image_diff_original).sum()
    diff_sum_0 = np.abs(image_diff_0).sum()
    diff_sum_1 = np.abs(image_diff_1).sum()
    
    diff_image_array = [image_diff_0, image_diff_1, image_diff_original]
    
    diff_array = [diff_sum_original, diff_sum_0, diff_sum_1]

    print(diff_sum_original, diff_sum_0, diff_sum_1)   

    min_diff_array = np.min(diff_array)

    if min_diff_array == diff_sum_original:
        return image_original_centre, True
    
#   runs else statements to find which image currently has the best symmetry and return a new centre value 
    elif diff_sum_0 == min_diff_array:
        return image_0_centre, False
    else:
        return image_1_centre, False        
        
def _centre_finder_iterative_method(original_dataset):
    original_dataset.change_dtype('float64')
    array_of_centres = np.zeros(shape=(64,64)) #To store all centre values in
    sub_image_size = 50.
    array_of_centres = np.zeros((64,64,2))
    centre = CoM.centre_of_disk_centre_of_mass(copy.deepcopy(original_dataset.data))[2:]
    centre = list(centre) #issue with tuple so cast to a list
    image_spectrum = original_dataset.to_spectrum()
    
    for i in range(0,64):
        for j in range(0,64):
            
#          compare all the images until find the minimum
            check = False
            while check == False:
                centre,check = find_centre_using_split_top(image_spectrum,centre, sub_image_size)
        
            check = False    
            while check == False: 
                centre, check = find_centre_using_split_left(image_spectrum, centre, sub_image_size)
                
            print centre
            array_of_centres[i,j] = centre
            
    return array_of_centres

original_dataset = CoM.loadh5py("default1.hdf5")
array_of_centres = _centre_finder_iterative_method(original_dataset)
               
        

