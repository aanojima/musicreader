#receive pitch, need to identify the type
#second step: find accidentals
import cv2
import numpy as np
import classifyhelper

from common import *
# External Dependencies
from PIL import Image

TREBLE = 1;
BASS = 2;

clef_dictionary = {1: ClefLabel.TREBLE,
 	 			  2: ClefLabel.BASS}

# Given: Image and list of bounding boxes
# assume note pitch has already been calculated
# assume given a note without any lines--using just plain handwritten notes for now
def getClef(rawImage, clefBoundingBox):
	# if adding the white pixel buffers, notesList may even be a list of matrices representing
	# each bounding box--in this case, just run our threshold on that data
	knn = classifyhelper.trainData()

	t1 = cv2.adaptiveThreshold(note,255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 5)
	cells = np.append(cells, t1, axis=0)

	result = classifyhelper.classifyClef(cells, testing_labels, knn)

	# new_result = result
		
	# change it so that matches up with Note.Types actual types
	# for i in xraresult:
	# 	new_result[]
	# 	note = new Note()
 #   		note.set_type(options[])
 	# {1: Type.Whole,
 	#  2: Type.Half,
 	#  4: Type.Quarter,
 	#  8: Type.EIGHTH}
 	return result

# trains data using all given data samples
def trainData():
	#input data
	t1 = classifyhelper.getTrainingData("../../data/clef_train/clef1.jpg")

	# put data in array of size 27 x 4
	cells1 = [np.hsplit(row, 6) for row in np.vsplit(t1, 10)]

	x = np.array(cells1);

	# hogg features to classify
	train = classifyhelper.preprocess_hog(x)

	# Make labels for train data
	labels = np.array([TREBLE, BASS, TREBLE, BASS, TREBLE, BASS])
	train_labels = np.tile(labels, 10)[:, np.newaxis]
	# print "train labels", train_labels.size

	# Initiate kNN, train the data, then test it with test data for k=1
	knn = cv2.ml.KNearest_create()
	knn.train(train,cv2.ml.ROW_SAMPLE, train_labels)

	return knn




if __name__ == '__main__':
	#later, take in a bunch of bounding boxes, (which should be sized 20x20 and find out what they are)
	test_name_list = ["../../data/cleftest1.jpg",
					   "../../data/cleftest2.jpg",
					   "../../data/cleftest3.jpg",
					   "../../data/cleftest4.jpg",]

	test_labels_list = [TREBLE, BASS, TREBLE, BASS]

	# TODO: must convert so that can get image matrx if given a bunch of bounding boxes
	# just need to look at original image and get sub section of it as a matrix

	knn = trainData() # is 20 x 20
	testing_data = classifyhelper.makeTestingData(test_name_list, 20, 20)
	print "testing_data", testing_data.shape
	testing_labels = classifyhelper.makeTestingLabels(test_labels_list)
	result = classifyhelper.classifyDebug(testing_data, testing_labels, knn)
	getLabels(result, clef_dictionary) # gets corresponding labels for the model
	# cv_image_note = getNoteLength(cv_image)

