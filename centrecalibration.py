import CoM
import copy
import numpy as np
import matplotlib.pyplot as plt

def _plot_compare_around_debug(image_compare_array, diff_compare_array):
    c0 = image_compare_array[0]
    c1 = image_compare_array[1]
    c2 = image_compare_array[2]
    c3 = image_compare_array[3]

    diff0 = diff_compare_array[0]
    diff1 = diff_compare_array[1]
    diff2 = diff_compare_array[2]
    diff3 = diff_compare_array[3]

    fig, axarr = plt.subplots(4,2, figsize=(5,10))
    axarr[0][0].imshow(c0) 
    axarr[0][1].imshow(diff0) 
    axarr[1][0].imshow(c1)
    axarr[1][1].imshow(diff1) 
    axarr[2][0].imshow(c2) 
    axarr[2][1].imshow(diff2) 
    axarr[3][0].imshow(c3) 
    axarr[3][1].imshow(diff3) 

    fig.tight_layout()
    fig.savefig("compare_images.jpg")

def find_center_using_image_flip(image, centre, sub_image_size):
    image.change_dtype('float64')
    
    image_0_centre = (centre[0]+1, centre[1])
    image_1_centre = (centre[0]-1, centre[1])
    image_2_centre = (centre[0], centre[1]+1)
    image_3_centre = (centre[0], centre[1]-1)

    image_compare_original = image[round(centre[0])-sub_image_size:round(centre[0])+sub_image_size,round(centre[1])-sub_image_size:round(centre[1])+sub_image_size].data
    image_compare_0 = image[round(centre[0])-sub_image_size-1:round(centre[0])+sub_image_size-1,round(centre[1])-sub_image_size:round(centre[1])+sub_image_size].data
    image_compare_1 = image[round(centre[0])-sub_image_size+1:round(centre[0])+sub_image_size+1,round(centre[1])-sub_image_size:round(centre[1])+sub_image_size].data
    image_compare_2 = image[round(centre[0])-sub_image_size:round(centre[0])+sub_image_size,round(centre[1])-sub_image_size-1:round(centre[1])+sub_image_size-1].data
    image_compare_3 = image[round(centre[0])-sub_image_size:round(centre[0])+sub_image_size,round(centre[1])-sub_image_size+1:round(centre[1])+sub_image_size+1].data
    
    image_diff_original = np.fliplr(image_compare_original) - image_compare_original
    image_diff_0 = np.fliplr(image_compare_0) - image_compare_0
    image_diff_1 = np.fliplr(image_compare_1) - image_compare_1
    image_diff_2 = np.fliplr(image_compare_2) - image_compare_2
    image_diff_3 = np.fliplr(image_compare_3) - image_compare_3

    compare_image_array = [image_compare_0, image_compare_1, image_compare_2, image_compare_3]
    diff_image_array = [image_diff_0, image_diff_1, image_diff_2, image_diff_3]
    _plot_compare_around_debug(compare_image_array, diff_image_array)

    diff_sum_original = np.abs(image_diff_original).sum()
    diff_sum_0 = np.abs(image_diff_0).sum()
    diff_sum_1 = np.abs(image_diff_1).sum()
    diff_sum_2 = np.abs(image_diff_2).sum()
    diff_sum_3 = np.abs(image_diff_3).sum()

    diff_array = [diff_sum_0, diff_sum_1, diff_sum_2, diff_sum_3, diff_sum_original]

    print(diff_sum_original, diff_sum_0, diff_sum_1, diff_sum_2, diff_sum_3)   

    diff_array = [diff_1, diff_2, diff_3, diff_4]
    min_diff = np.min(diff_array) - imDiffSum
    if np.min(diff_array) > imDiffSum:
        return imOriginal, centre, True
    
#   runs else statements to find which image currently has the best symmetry and return that total sum along with a new centre value 
    elif (diff_1 - imDiffSum) == min_diff:
        (centre[0]) -= 1
        return diff_1,centre, False
    elif (diff_2 - imDiffSum) == min_diff:
        (centre[0]) += 1
        return diff_2,centre, False
    elif (diff_3 - imDiffSum) == min_diff:
        (centre[1]) -= 1
        return diff_3,centre, False
    elif (diff_4 - imDiffSum) == min_diff:
        (centre[1]) += 1
        return diff_4,centre, False

        
######################################################
im = CoM.loadh5py("default1.hdf5")
centres = np.zeros(shape=(64,64)) #To store all centre values in

#   This section is test the code on a single probe position        
im2 = im.inav[34,50]
centre = CoM.centre_of_disk_centre_of_mass(copy.deepcopy(im2.data))
centre = list(centre) #issue with tuple so cast to a list
centre = [int(i) for i in centre]
print centre
im3 = im2.to_spectrum()
spread = 40.
im4 = im3[centre[0]-spread:centre[0]+spread,centre[1]-spread:centre[1]+spread]
imOriginal = im4.data
imOriginal = np.array(imOriginal, dtype = np.int64)
imFlip = np.fliplr(imOriginal)
imDiffSum = flip_and_compare(imOriginal, imFlip)

#       compare all the images until find the minimum
check = False
while check == False: 
#   On each pass, imDiffSum is updated with the new lowest total and the centre of the image that it occurs at
    imDiffSum, centre, check = compare_around(im3, imOriginal, imFlip, imDiffSum, centre, spread)
    print centre   

print centre



'''for i in range (0,64):
    for j in range (0,64):
        im2 = im.inav[i,j]
        centre = CoM.centre_of_disk_centre_of_mass(copy.deepcopy(
        im2.data))
        im3 = im2.to_spectrum()
        spread = 50.
        im4 = im3[round(centre[0])-spread:round(centre[0])+spread,round(centre[1])-spread:round(centre[1])+spread]
        imOriginal = im4.data
        imFlip = np.fliplr(imOriginal)


#       compare all the images until find the minimum
        while False
            imDiffSum = flip_and_compare(imOriginal, imFlip)
            imOriginal, centre = compare_around(im3, imFlip, imDiffSum, centre, spread)
            return imOriginal

        centres[i,j] = centre'''
        
