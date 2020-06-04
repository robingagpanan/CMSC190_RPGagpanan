import cv2
import numpy as np
import collections
from glob import glob
import json

#######################################################################################
#
#######################################################################################

#computing for the average of nine 5x5 pixel values in the image
def average(rows, cols, image):
	#location of starting pixels depending on where is the position of the current pixel
	pixel_location = [-7, -2, 3]
	average_value = np.zeros((3, 3), np.float32)
	total_sum = 0

	#3x3 array matrix where it will compute the average of nine 5x5 array matrix to be used for
	#computing the LBP
	for index_x, location_x in enumerate(pixel_location):
		count_x = rows + location_x
		for index_y, location_y in enumerate(pixel_location):
			count_y = cols + location_y
			
			#it will get the average by computing the sum of the 5x5 pixel matrix
			for x in range(count_x, count_x + 5): 
				for y in range(count_y, count_y + 5):
					total_sum = total_sum + image[x, y]

			average_value[index_x, index_y] = total_sum / 25
			total_sum = 0

	return(average_value)

#computing for the LBP value using the 3x3 matrix.
#the center pixel will be compared on its surrounding pixel staring from pixel (0,0) 
#in clockwise rotation. if center pixel is greater than the surrounding pixel,
#the value of the surrounding pixel will be 0, otherwise the value will be 1.
#then each pixel will be multiplied by the powers of two then compute for the sum
#the value will be a part of the histogram to be used for classifying images.
def lbp(aveList, center):
	summ_lbp = 0
	num = 0

	num = 1 if center <= aveList[0,0] else 0
	summ_lbp = summ_lbp + (num * 1)
	num = 1 if center <= aveList[0,1] else 0
	summ_lbp = summ_lbp + (num * 2)
	num = 1 if center <= aveList[0,2] else 0
	summ_lbp = summ_lbp + (num * 4)
	num = 1 if center <= aveList[1,2] else 0
	summ_lbp = summ_lbp + (num * 8)
	num = 1 if center <= aveList[2,2] else 0
	summ_lbp = summ_lbp + (num * 16)
	num = 1 if center <= aveList[2,1] else 0
	summ_lbp = summ_lbp + (num * 32)
	num = 1 if center <= aveList[2,0] else 0
	summ_lbp = summ_lbp + (num * 64)
	num = 1 if center <= aveList[1,0] else 0
	summ_lbp = summ_lbp + (num * 128)

	return(summ_lbp)

#image preprocessing
#bilateral filter to reduce noise and smoothens an image while preserving its edges
#then converts an image to grayscale to perform LBP
def img_preprocessing(img):
	smooth = cv2.bilateralFilter(img,30,70,70)
	gray = cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY)
	return gray

#######################################################################################
#
#######################################################################################

#stores the name of all image files from the informal folder in an array
#to be used when calling the file for getting the LBP histogram
informal_folder = 'training/informal/*'
informal_img_files = glob(informal_folder)

#where all the LBP histograms of informal images are stored
informal_histogram_list = []
count = 1
#getting the file names of the image from the array
for informal_images in informal_img_files:
	print('processing %s...' % informal_images,)
	#store the lbp values of the current image
	informal_mblbp_list = []
	#reads the image in the folder
	informal_img = cv2.imread(informal_images, 1)
	#applying pre-processing methods on the image
	informal_img = img_preprocessing(informal_img)
	#getting the dimention values of the current image
	informal_rows, informal_cols = informal_img.shape
	#pixel matrix for the average pixels
	informal_average_value = np.zeros((3,3), np.float32)

	#testing purposes. makes an empty image then stores the computed LBP values
	#lbp_img = np.zeros((informal_rows,informal_cols))

	#loops through the pixels on the image
	for rows in range(informal_rows):
		for cols in range(informal_cols):
			#skips the pixel to avoid values outside the boundary of the image
			#when performing LBP
			if rows < 7 or rows > informal_rows - 8:
				continue
			if cols < 7 or cols > informal_cols - 8:
				continue
			#getting the average of the nine 5x5 pixel matrix
			informal_average_value = average(rows, cols, informal_img)
			#getting the center value from the average pixel matrix to be compared
			informal_center_value = informal_average_value[1,1]
			#performing the LBP using the average pixel matrix
			informal_mblbp_value = lbp(informal_average_value, informal_center_value)
			#stores the value of the computed LBP in an array
			informal_mblbp_list.append(informal_mblbp_value)
			
			#putting the computed LBP pixel value on an image
			#lbp_img[rows, cols] = informal_mblbp_value
	#saving the LBP image
	# if count < 10:
	# 	cv2.imwrite('informal/000' + str(count) + ".jpg", lbp_img)
	# elif count >= 10:
	# 	cv2.imwrite('informal/00' + str(count) + ".jpg", lbp_img)
	# count = count + 1

	#using the LBP values to make a LBP histogram
	informal_histogram = collections.Counter(informal_mblbp_list)
	#stores the histogram along with other histograms
	informal_histogram_list.append(dict(informal_histogram))

#save the LBP histograms of all informal image into a file
with open('save_files/informal.txt', 'w') as fout:
    json.dump(informal_histogram_list, fout)