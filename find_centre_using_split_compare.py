import CoM
import copy
import numpy as np
import matplotlib.pyplot as plt

def _plot_compare_around_debug(diff_compare_array):
    
    diff0 = diff_compare_array[0]
    diff1 = diff_compare_array[1]
#    diff2 = diff_compare_array[2]
#    diff3 = diff_compare_array[3]
    diff_original = diff_compare_array[2]

    fig, axarr = plt.subplots(3,1, figsize=(3,8))
    axarr[0].imshow(diff0) 

    axarr[1].imshow(diff1)

#    axarr[2].imshow(diff2) 

#    axarr[3].imshow(diff3) 

    axarr[2].imshow(diff_original)


    fig.tight_layout()
    fig.savefig("split_images.jpg")

def find_centre_using_split_top(image, centre, sub_image_size):
    
    image_original_centre = (centre[0], centre[1])
    image_0_centre = (centre[0]-1, centre[1])
    image_1_centre = (centre[0]+1, centre[1])
#    image_2_centre = (centre[0], centre[1]-1)
#    image_3_centre = (centre[0], centre[1]+1)
    
    image_top_original = image[round(centre[0])-sub_image_size:round(centre[0]),round(centre[1])-sub_image_size:round(centre[1])+sub_image_size].data
    image_top_0 = image[round(centre[0])-sub_image_size-1:round(centre[0])-1,round(centre[1])-sub_image_size:round(centre[1])+sub_image_size].data
    image_top_1 = image[round(centre[0])-sub_image_size+1:round(centre[0])+1,round(centre[1])-sub_image_size:round(centre[1])+sub_image_size].data
#    image_top_2 = image[round(centre[0])-sub_image_size:round(centre[0]),round(centre[1])-sub_image_size-1:round(centre[1])+sub_image_size-1].data
#    image_top_3 = image[round(centre[0])-sub_image_size:round(centre[0]),round(centre[1])-sub_image_size+1:round(centre[1])+sub_image_size+1].data
    
    image_bottom_original = image[round(centre[0]):round(centre[0])+sub_image_size,round(centre[1])-sub_image_size:round(centre[1])+sub_image_size].data
    image_bottom_0 = image[round(centre[0])-1:round(centre[0])+sub_image_size-1,round(centre[1])-sub_image_size:round(centre[1])+sub_image_size].data
    image_bottom_1 = image[round(centre[0])+1:round(centre[0])+sub_image_size+1,round(centre[1])-sub_image_size:round(centre[1])+sub_image_size].data
#    image_bottom_2 = image[round(centre[0]):round(centre[0])+sub_image_size,round(centre[1])-sub_image_size-1:round(centre[1])+sub_image_size-1].data
#    image_bottom_3 = image[round(centre[0]):round(centre[0])+sub_image_size,round(centre[1])-sub_image_size+1:round(centre[1])+sub_image_size+1].data
    
    image_diff_original = np.flipud(image_top_original) - image_bottom_original
    image_diff_0 = np.flipud(image_top_0) - image_bottom_0
    image_diff_1 = np.flipud(image_top_1) - image_bottom_1
#    image_diff_2 = image_left_2 - image_right_2
#    image_diff_3 = image_left_3 - image_right_3
    
    diff_sum_original = np.abs(image_diff_original).sum()
    diff_sum_0 = np.abs(image_diff_0).sum()
    diff_sum_1 = np.abs(image_diff_1).sum()
#    diff_sum_2 = np.abs(image_diff_2).sum()
#    diff_sum_3 = np.abs(image_diff_3).sum()
    
    diff_image_array = [image_diff_0, image_diff_1, image_diff_original]
    diff_array = [diff_sum_original, diff_sum_0, diff_sum_1]
    
#    _plot_compare_around_debug(diff_image_array)

    print(diff_sum_original, diff_sum_0, diff_sum_1)   

    min_diff_array = np.min(diff_array)

    if min_diff_array == diff_sum_original:
        return image_original_centre, True
    
#   runs else statements to find which image currently has the best symmetry and return a new centre value 
    elif diff_sum_0 == min_diff_array:
        return image_0_centre, False
#    elif diff_sum_1 == min_diff_array:
    else:
        return image_1_centre, False
#    elif diff_sum_2 == min_diff_array:
#        return image_2_centre, False
#    else:
#        return image_3_centre, False
        
        
def find_centre_using_split_left(image, centre, sub_image_size):
    
    image_original_centre = (centre[0], centre[1])
#    image_0_centre = (centre[0]-1, centre[1])
#    image_1_centre = (centre[0]+1, centre[1])
    image_2_centre = (centre[0], centre[1]-1)
    image_3_centre = (centre[0], centre[1]+1)
    
    image_left_original = image[round(centre[0])-sub_image_size:round(centre[0])+sub_image_size,round(centre[1]):round(centre[1])+sub_image_size].data
#    image_left_0 = image[round(centre[0])-sub_image_size-1:round(centre[0])+sub_image_size-1,round(centre[1]):round(centre[1])+sub_image_size].data
#    image_left_1 = image[round(centre[0])-sub_image_size+1:round(centre[0])+sub_image_size+1,round(centre[1]):round(centre[1])+sub_image_size].data
    image_left_2 = image[round(centre[0])-sub_image_size:round(centre[0])+sub_image_size,round(centre[1])-1:round(centre[1])+sub_image_size-1].data
    image_left_3 = image[round(centre[0])-sub_image_size:round(centre[0])+sub_image_size,round(centre[1])+1:round(centre[1])+sub_image_size+1].data
    
    image_right_original = image[round(centre[0])-sub_image_size:round(centre[0])+sub_image_size,round(centre[1])-sub_image_size:round(centre[1])].data
#    image_right_0 = image[round(centre[0])-sub_image_size-1:round(centre[0])+sub_image_size-1,round(centre[1])-sub_image_size:round(centre[1])].data
#    image_right_1 = image[round(centre[0])-sub_image_size+1:round(centre[0])+sub_image_size+1,round(centre[1])-sub_image_size:round(centre[1])].data
    image_right_2 = image[round(centre[0])-sub_image_size:round(centre[0])+sub_image_size,round(centre[1])-sub_image_size-1:round(centre[1])-1].data
    image_right_3 = image[round(centre[0])-sub_image_size:round(centre[0])+sub_image_size,round(centre[1])-sub_image_size+1:round(centre[1])+1].data
    
    image_diff_original = np.fliplr(image_left_original) - image_right_original
#    image_diff_0 = image_left_0 - image_right_0
#    image_diff_1 = image_left_1 - image_right_1
    image_diff_2 = np.fliplr(image_left_2) - image_right_2
    image_diff_3 = np.fliplr(image_left_3) - image_right_3
    
    diff_sum_original = np.abs(image_diff_original).sum()
#    diff_sum_0 = np.abs(image_diff_0).sum()
#    diff_sum_1 = np.abs(image_diff_1).sum()
    diff_sum_2 = np.abs(image_diff_2).sum()
    diff_sum_3 = np.abs(image_diff_3).sum()
    
    diff_image_array = [image_diff_2, image_diff_3, image_diff_original]
#    _plot_compare_around_debug(diff_image_array)
    
    diff_array = [diff_sum_original, diff_sum_2, diff_sum_3]

    print(diff_sum_original, diff_sum_2, diff_sum_3)   

    min_diff_array = np.min(diff_array)

    if min_diff_array == diff_sum_original:
        return image_original_centre, True
    
#   runs else statements to find which image currently has the best symmetry and return a new centre value 
#    elif diff_sum_0 == min_diff_array:
#        return image_0_centre, False
#    elif diff_sum_1 == min_diff_array:
#        return image_1_centre, False
    elif diff_sum_2 == min_diff_array:
        return image_2_centre, False
    else:
        return image_3_centre, False        
        
original_data_set = CoM.loadh5py("default1.hdf5")
array_of_centres = np.zeros(shape=(64,64)) #To store all centre values in

#   This section is test the code on a single probe position        
single_image = original_data_set.inav[34,50]
centre = CoM.centre_of_disk_centre_of_mass(copy.deepcopy(single_image.data))
centre = list(centre) #issue with tuple so cast to a list
centre = [int(i) for i in centre]
print centre
image_spectrum = single_image.to_spectrum()

#       compare all the images until find the minimum

sub_image_size = 50.
image_spectrum.change_dtype('float64')

check = False
while check == False:
    centre,check = find_centre_using_split_top(image_spectrum,centre, sub_image_size)
    print centre 

check = False    
while check == False: 
#   On each pass, the centre of the image that the minimum appears at is returned
    centre, check = find_centre_using_split_left(image_spectrum, centre, sub_image_size)
    print centre
    
  

print centre

               
        

