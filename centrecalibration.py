import CoM
import copy
import numpy as np
import matplotlib.pyplot as plt

## This just subtracts the two images and sums the total intensity
## Hyperspy returns uint16 hence I manually set what would be negative
## numbers to be 0
def flip_and_compare(imOriginal, imFlip):
    imDiff = imFlip - imOriginal
    imDiff[imFlip>imOriginal]  == 0.
    imDiffSum = imDiff.sum(0).sum(0)
    return imDiffSum
    
## This function creates 4 new images to compare to the current "best guess"
## It then returns the new "best guess" along with the corresponding 
## centre value
def compare_around(im3, imOriginal, imFlip, imDiffSum, centre, spread):  
#   create 4 images each one shift 1 pixel in each direction from the centre which is update each iteration
    compare_1 = im3[round(centre[0])-spread-1:round(centre[0])+spread-1,round(centre[1])-spread:round(centre[1])+spread]
    compare_2 = im3[round(centre[0])-spread+1:round(centre[0])+spread+1,round(centre[1])-spread:round(centre[1])+spread]
    compare_3 = im3[round(centre[0])-spread:round(centre[0])+spread,round(centre[1])-spread-1:round(centre[1])+spread-1]
    compare_4 = im3[round(centre[0])-spread:round(centre[0])+spread,round(centre[1])-spread+1:round(centre[1])+spread+1]

#   compare each shifted image to the flipped image and find the total sum of intensites  
    diff_1 = flip_and_compare(compare_1.data, imFlip)
    diff_2 = flip_and_compare(compare_2.data, imFlip)
    diff_3 = flip_and_compare(compare_3.data, imFlip)
    diff_4 = flip_and_compare(compare_4.data, imFlip)
    
#   finds the minimum value for a quick check to see if the current image is infact a minimum
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
        
