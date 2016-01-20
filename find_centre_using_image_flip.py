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
