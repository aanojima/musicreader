import cv2
import numpy as np
from numpy.linalg import norm
from PIL import Image

from object_segment3 import *


def preprocess_hog(x):
    samples = []
    numrows = x.shape[0]
    numcols = x.shape[1]
    for row in range(0,numrows):
    	for col in range(0,numcols):
    		img = x[row][col]
    		# print "hog"
    		# print row, col
    		# print img
    		
    		# print type(img)
    		# img.convertTo(img, cv2.CV_32F)
	        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
	        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
	        mag, ang = cv2.cartToPolar(gx, gy)
	        bin_n = 16
	        bin = np.int32(bin_n*ang/(2*np.pi))
	        bin_cells = bin[:10,:10], bin[10:,:10], bin[:10,10:], bin[10:,10:]
	        mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
	        hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
	        hist = np.hstack(hists)

	        # transform to Hellinger kernel
	        eps = 1e-7
	        hist /= hist.sum() + eps
	        hist = np.sqrt(hist)
	        hist /= norm(hist) + eps

        	samples.append(hist)
    # print "len samples", len(samples)
    return np.float32(samples)

def center_data(x):
	samples = []
	print "x.shape", x.shape
	numrows = x.shape[0]
	numcols = x.shape[1]
	for row in range(0,numrows):
		for col in range(0,numcols):
			img = x[row][col]
			print findObjects(img)
			samples.append(hist)
	return np.float32(samples)


# classify all given notes using the given knn
# for testing purposes!!!! (includes pregiven testing labels)
def classifyDebug(testingData, testingLabels, knn):
	# print testingData.shape
	# test = testingData[:,:].reshape(-1, 400).astype(np.float32)
	# print test.dtype
	print "CLASSIFY DEBUG"
	# testing_data = center_data(testingData)
	testing_data = preprocess_hog(testing_data)
	ret,result,neighbours,dist = knn.findNearest(testing_data,k=3)
	print "result:\n", result
	print "expected: \n", testingLabels

	matches = result==testingLabels
	correct = np.count_nonzero(matches)
	accuracy = correct*100.0/result.size
	print "accuracy:", accuracy

	return result


# classify all given notes using the given knn
# not given testing labels to compare
def classify(testingData, knn):

	# test = testingData[:,:].reshape(-1, 400).astype(np.float32)
	testing_data = preprocess_hog(testingData)
	ret,result,neighbours,dist = knn.findNearest(testing_data,k=3)
	ret,result,neighbours,dist = knn.findNearest(test,k=3)
	print "result:\n", result

	return result


# converts image to grayscaled inverted image to be used in knn
# input: name of image file
def getTrainingData(imageFileName):
	i1 = Image.open(imageFileName).convert('RGB') # each image 20*20= 400
	cv_1 = np.array(i1)
	cv_1 = cv_1[:, :, ::-1].copy()
	cv_1 =  cv2.cvtColor(cv_1, cv2.COLOR_BGR2GRAY)
	t1 = cv2.adaptiveThreshold(cv_1,255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 3)	
	# need to center all data....
	# cv2.imshow('thresholded', t1)
	# cv2.waitKey(0)
	return t1


# add training image to the cells array and makes np array
# input: list of image filenames -- for testing purposes!!!
def makeTestingData(imageNameList, width, height):
	cells = np.empty((len(imageNameList), 1, width,height), np.float32)
	# print cells.shape
	for index in range(0, len(imageNameList)):
		imageName = imageNameList[index]
		i1 = Image.open(imageName).convert('RGB') # each image 25 x 50 = 1250
		cv_1 = np.array(i1)
		cv_1 = cv_1[:, :, ::-1].copy()
		cv_1 = cv2.resize(cv_1, (width, height))
		cv_1 =  cv2.cvtColor(cv_1, cv2.COLOR_BGR2GRAY)
		t1 = cv2.adaptiveThreshold(cv_1,255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 5)
		cells[index][0] = t1
		# cv2.imshow('thresholded', t1)
		# cv2.waitKey(0)
		# print"hello"
		
		# print "cells shape",  cells.shape
		# print cells
	# cells = np.float64(cells)
	# print "cells shape",  cells.shape		
	cells = np.array(cells)
	return cells


# makes expected testing labels in column format
# for testing purposes!!
def makeTestingLabels(test_labels_list):
	test_labels =  np.array(test_labels_list)
	test_labels = np.tile(test_labels, 1)[:, np.newaxis]
	return test_labels # to make column vector

# maps output results to the corresponding labels for the model
def getLabels(result, dictionary):
	# print result
	result = result.astype(int)
	# print result
 	labels = []
	for i in range(len(result)):
		index = result.item(i)
		labels.append(dictionary[index])
	print labels
	return labels
