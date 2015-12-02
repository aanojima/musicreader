#receive pitch, need to identify the type
#second step: find accidentals
import cv2
import numpy as np

# External Dependencies
from PIL import Image

# from enum import Enum 

# class NoteLength(Enum):
# 	whole = 1
# 	half = 2
# 	quarter = 4
# 	eighth = 8

WHOLE = 1;
HALF = 2;
QUARTER = 4;
EIGHTH = 8;


# Given: Image and bounding box; assume note pitch has already been calculated
# assume given a note without any lines--using just plain handwritten notes for now
def getNoteLength(rawImage):
	# threshold
    ret,thresh2 = cv2.threshold(rawImage,127,255,cv2.THRESH_BINARY_INV)
    cv2.imshow('thresholded', thresh2)
    cv2.waitKey(0)

def trainData():
	#input data
	i1 = Image.open("../../data/notes1.png").convert('RGB') # each image 20*20= 400
	cv_1 = np.array(i1)
	cv_1 = cv_1[:, :, ::-1].copy()
	cv_1 =  cv2.cvtColor(cv_1, cv2.COLOR_BGR2GRAY)
	t1 = cv2.adaptiveThreshold(cv_1,255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 3)

	i2 = Image.open("../../data/img2.png").convert('RGB') # each image 20*20= 400
	cv_2 = np.array(i2)
	cv_2 = cv_2[:, :, ::-1].copy()
	cv_2 =  cv2.cvtColor(cv_2, cv2.COLOR_BGR2GRAY)
	t2 = cv2.adaptiveThreshold(cv_2, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 3)

	i3 = Image.open("../../data/img3.jpg").convert('RGB') # each image 20*20= 400
	cv_3 = np.array(i3)
	cv_3 = cv_3[:, :, ::-1].copy()
	cv_3 =  cv2.cvtColor(cv_3, cv2.COLOR_BGR2GRAY)
	t3 = cv2.adaptiveThreshold(cv_3, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 3)

	# put data in array of size 27 x 4
	cells1 = [np.hsplit(row, 4) for row in np.vsplit(t1, 9)]
	cells2 = [np.hsplit(row, 4) for row in np.vsplit(t2, 9)]
	cells3 = [np.hsplit(row, 4) for row in np.vsplit(t3, 9)]

	cells = np.vstack((cells1, cells2, cells3))
	x = np.array(cells);
	# print x.shape

	# first 22 samples for training, last 5 samples for testing
	train = x[:, :].reshape(-1, 400).astype(np.float32)
	# test = x[22:27,:].reshape(-1, 400).astype(np.float32)
	# can do something like this when we want to enter in multiple notes all at once and classify all of them

	# print train.shape
	# print test.shape

	# Make labels for train and test data
	labels = np.array([WHOLE, HALF, QUARTER, EIGHTH])
	train_labels = np.tile(labels, 27)[:, np.newaxis]
	# test_labels = np.tile(labels, 5)[:, np.newaxis]



	# Initiate kNN, train the data, then test it with test data for k=1
	knn = cv2.ml.KNearest_create()
	knn.train(train,cv2.ml.ROW_SAMPLE, train_labels)
	# ret,result,neighbours,dist = knn.findNearest(test,k=3)

	# Now we check the accuracy of classification
	# For that, compare the result with test_labels and check which are wrong
	# matches = result==test_labels
	# correct = np.count_nonzero(matches)
	# accuracy = correct*100.0/result.size
	# print "accuracy:", accuracy

	# cv2.imshow("Original Image", t3)
	# cv2.waitKey(0)
	return knn

# classify just one note
#knownNoteValue used purely to compare if its right
def classifyNotes(testingData, testingLabels, knn):
	# image = cv2.resize(image, (20, 20))
	# image =  cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# t1 = cv2.adaptiveThreshold(image,255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 5)

	# x = np.array(t1);
	# print x.shape
	test = testingData[:,:].reshape(-1, 400).astype(np.float32)
	ret,result,neighbours,dist = knn.findNearest(test,k=3)
	print "result:\n", result
	print "expected: \n", testingLabels

	matches = result==testingLabels
	correct = np.count_nonzero(matches)
	accuracy = correct*100.0/result.size
	print "accuracy:", accuracy

	# cv2.imshow("Original Image", t1)
	# cv2.waitKey(0)




#add training image to the cells array and makes np array
def makeTestingData(imageNameList):
	cells = np.empty((0,20), int)
	for imageName in imageNameList:
		i1 = Image.open(imageName).convert('RGB') # each image 25 x 50 = 1250
		cv_1 = np.array(i1)
		cv_1 = cv_1[:, :, ::-1].copy()
		cv_1 = cv2.resize(cv_1, (20, 20))
		cv_1 =  cv2.cvtColor(cv_1, cv2.COLOR_BGR2GRAY)
		t1 = cv2.adaptiveThreshold(cv_1,255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 5)
		cells = np.append(cells, t1, axis=0)
	return cells

# makes expected testing labels in column format
def makeTestingLabels(test_labels_list):
	test_labels =  np.array(test_labels_list)
	test_labels = np.tile(test_labels, 1)[:, np.newaxis]
	return test_labels # to make column vector


if __name__ == '__main__':
    # cv2.destroyAllWindows()
	# image = Image.open("../../data/quarter-note.png").convert('RGB')
	# cv_image = np.array(image)
	# cv_image = cv_image[:, :, ::-1].copy()
	# width = cv_image.shape[1]
	# height = cv_image.shape[0]
	# cv_image = cv2.resize(cv_image, (width, height)) # TODO: Play around with size
	# cv2.imshow("Original Image", cv_image)
	# cv2.waitKey(0)
	test_name_list = ["../../data/whole-note.png",
					   "../../data/half-note.png",
					   "../../data/quarter-note.png",
					   "../../data/eighth-note.png",
					   "../../data/test-whole.png",
					   "../../data/test-half.png",
					   "../../data/test-quarter.png",
					   "../../data/test-eighth.png"]

	test_labels_list = [WHOLE, HALF, QUARTER, EIGHTH, WHOLE, HALF, QUARTER, EIGHTH]

	knn = trainData()
	testing_data = makeTestingData(test_name_list)
	testing_labels = makeTestingLabels(test_labels_list)
	classifyNotes(testing_data, testing_labels, knn)
	# cv_image_note = getNoteLength(cv_image)

