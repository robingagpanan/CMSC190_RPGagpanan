import cv2
import os
import numpy as np
import collections
from glob import glob
import json
from scipy.linalg import norm
import sys

#######################################################################################
#
#######################################################################################
#command line arguments for input and output file names
if len(sys.argv) != 3:
	print("Invalid Argument. Number of Arguments: 3")
	print("Argument List: filename name_of_input_file name_of_output_file")
	exit()
#input file must be an image file (jpg,png)
if sys.argv[1].endswith(('.jpg', '.png')) == False:
	print("invalid input file format. use .jpg or .png")
	exit()
#output file must be an image file (jpg,png)
if sys.argv[2].endswith(('.jpg', '.png')) == False:
	print("invalid output file format. use .jpg or .png")
	exit()

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
	smooth = cv2.bilateralFilter(img, 30, 70, 70)
	gray = cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY)
	return gray

#compute for the cosine similarity between two histograms
def cosine_similarity(hist_a, hist_b):
	#to avoid error, the histogram with the lower number of bins
	#are used for the loop to compute for the summation of AxB
    if len(hist_b) < len(hist_a):
        hist_a, hist_b = hist_b, hist_a
    #computing for the summation of AxB
    #get the key from the dictionary to be used to get the
    #values of the two histogram
    summ_AxB = 0
    for a_key, a_value in hist_a.items():
        summ_AxB += a_value * hist_b.get(a_key, 0)
    if summ_AxB == 0:
        return 0
    #divides the summation of AxB to the frobenius norm of A times the frobenius norm of B
    try:
        summ_AxB = summ_AxB / norm(list(hist_a.values())) / norm(list(hist_b.values()))
    except ZeroDivisionError:
        summ_AxB = 0
    return summ_AxB

#######################################################################################
#
#######################################################################################

#loads the histograms computed from formal images
formal_load_file = []
formal_histogram_list = []
with open(r"save_files/formal.txt" , 'r') as f:
    formal_load_file = json.load(f)

for i in formal_load_file:
	i = {int(k):int(v) for k,v in i.items()}
	formal_histogram_list.append(i)

#loads the histograms computed from informal images
informal_load_file = []
informal_histogram_list = []
with open(r"save_files/informal.txt" , 'r') as f:
    informal_load_file = json.load(f)

for i in informal_load_file:
	i = {int(k):int(v) for k,v in i.items()}
	informal_histogram_list.append(i)

#######################################################################################
#
#######################################################################################

#makes a new testing folder that contains sliced images from the input image to be
#classified
testing_directory = 'Testing'
try:
	os.mkdir(testing_directory)
	print("Directory " , testing_directory ,  " Created ") 
except FileExistsError:
	print("Directory " , testing_directory ,  " already exists")
#reads the input image to be classified
img = cv2.imread(sys.argv[1], 1)

#slices the input image by 36 320 x 180 images
#using each of the images to classify whether it is formal or informal settlement
count = 1
for i in range(6):
	for j in range(6):
		cropped = img[i * 180: (i+1) * 180, j * 320: (j+1) * 320]
		
		if count < 10:
			cv2.imwrite('Testing/000' + str(count) + ".jpg", cropped)
		elif count >= 10:
			cv2.imwrite('Testing/00' + str(count) + ".jpg", cropped)
		
		count = count + 1

#create folder for the classified images
result_directory = 'Result'
try:
	os.mkdir(result_directory)
	print("Directory " , result_directory ,  " Created ") 
except FileExistsError:
	print("Directory " , result_directory ,  " already exists")

#######################################################################################
#
#######################################################################################

#stores the name of all image files from the testing folder in an array
#to be used when calling the file for getting the LBP histogram
testing_folder = 'Testing/*'
testing_img_files = glob(testing_folder)
#color used for overlaying the original image
#red for informal settlements
#green for iformal settlements
red = cv2.imread('color/red.jpg',1)
green = cv2.imread('color/green.jpg',1)
count = 1
#getting the file names of the image from the array
for testing_images in testing_img_files:
	print('processing %s...' % testing_images,)
	#store the lbp values of the current image
	testing_mblbp_list = []
	#reads the image in the folder
	testing_img = cv2.imread(testing_images, 1)
	#reads the colored image in the folder
	colored_img = cv2.imread(testing_images, 1)
	#applying pre-processing methods on the image
	testing_img = img_preprocessing(testing_img)
	#getting the dimention values of the current image
	testing_rows, testing_cols = testing_img.shape
	#pixel matrix for the average pixels
	testing_average_value = np.zeros((3,3), np.float32)

	#stored the value of similarity of two histograms between
	#formal/informal histogram and testing histogram
	total_histogram_lists = []
	histogram_count = {}
	#loops through the pixels on the image
	for rows in range(testing_rows):
		for cols in range(testing_cols):
			#skips the pixel to avoid values outside the boundary of the image
			#when performing LBP
			if rows < 7 or rows > testing_rows - 8:
				continue
			if cols < 7 or cols > testing_cols - 8:
				continue
			#getting the average of the nine 5x5 pixel matrix
			testing_average_value = average(rows, cols, testing_img)
			#getting the center value from the average pixel matrix to be compared
			testing_center_value = testing_average_value[1,1]
			#performing the LBP using the average pixel matrix
			testing_mblbp_value = lbp(testing_average_value, testing_center_value)
			#stores the value of the computed LBP in an array
			testing_mblbp_list.append(testing_mblbp_value)

	#using the LBP values to make a LBP histogram
	testing_histogram = collections.Counter(testing_mblbp_list)

	#computes the cosine similarity between the formal LBP histograms and the testing LBP histogram
	for i in range(len(formal_histogram_list)):
		formal_correlation = cosine_similarity(formal_histogram_list[i], testing_histogram)
		total_histogram_lists.append(["formal", formal_correlation])

	#computes the cosine similarity between the informal LBP histograms and the testing LBP histogram
	for i in range(len(informal_histogram_list)):
		informal_correlation = cosine_similarity(informal_histogram_list[i], testing_histogram)
		total_histogram_lists.append(["informal", informal_correlation])

	#sorts the similarity values between formal/informal to the testing image in descending order
	total_histogram_lists_sorted = sorted(total_histogram_lists , key=lambda x: x[1], reverse=True)
	#print(total_histogram_lists_sorted)
	#only the top 50 values will be included
	total_histogram_lists_sliced = total_histogram_lists_sorted[:50]
	total_histogram_lists_result = [histogram.pop(0) for histogram in total_histogram_lists_sliced]
	#how many classified image on the top 50 images is formal or informal
	histogram_count = collections.Counter(total_histogram_lists_result)
	#print(histogram_count)
	#get the highest count of the classified image
	result = max(histogram_count, key=histogram_count.get)
	#print(result)
	#if the highest count is formal, sliced image will be highlighted as green, otherwise red
	if result == "formal":
		final_img = cv2.addWeighted(colored_img, .8, green, .2,0)
	elif result == "informal":
		final_img = cv2.addWeighted(colored_img, .8, red, .2,0)
	#saves the sliced image
	if count < 10:
		cv2.imwrite('Result/000'+str(count)+".jpg", final_img)
	elif count >= 10:
		cv2.imwrite('Result/00'+str(count)+".jpg", final_img)
	count = count + 1

#######################################################################################
#
#######################################################################################

#empty image
background = cv2.imread('color/bg.jpg',1)
#get the filenames from the result folder where the sliced classified images are stored
result_folder = 'Result/*'
result_img_files = glob(result_folder)

result_image_count = 0
#combines the images from the result folder to a one complete image
for i in range(6):
	for j in range(6):
		result_img = cv2.imread(result_img_files[result_image_count], 1)
		background[i*180: (i*180) + result_img.shape[0], j*320: (j*320) + result_img.shape[1]] = result_img
		result_image_count = result_image_count + 1
#saves the image
cv2.imwrite(sys.argv[2], background)

#######################################################################################
#
#######################################################################################

#REFERENCE
#Cosine Similarity Function - https://stackoverflow.com/questions/22381939/python-calculate-cosine-similarity-of-two-dicts-faster