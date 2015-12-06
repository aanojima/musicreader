#receive pitch, need to identify the type
#second step: find accidentals
import cv2
import numpy as np
import classifyhelper

# External Dependencies
from PIL import Image

WHOLE = 1;
HALF = 2;
QUARTER = 4;
EIGHTH = 8;
SHARP = 9;
FLAT = 10;
NATURAL = 11;

note_dictionary = {1: ClefLabel.TREBLE,
 	 			  2: ClefLabel.BASS}

# Given: Image and list of bounding boxes
# assume note pitch has already been calculated
# assume given a note without any lines--using just plain handwritten notes for now
def getNoteLengths(rawImage, notesList):
	# if adding the white pixel buffers, notesList may even be a list of matrices representing
	# each bounding box--in this case, just run our threshold on that data
	knn = trainData()

	for note in notesList:
		t1 = cv2.adaptiveThreshold(note,255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 5)
		cells = np.append(cells, t1, axis=0)

	result = classifyhelper.classifyNotes(cells, knn)

	# new_result = result
		
	# change it so that matches up with Note.Types actual types
	# Whole, Half, Quarter, Eighth, Rest
	# for i in xraresult:
	# 	new_result[]
	# 	note = new Note()
 #   		note.set_type(options[])
 	# {1: ,
 	#  2: }
 	return result

# trains note type and accidentals using all given data samples
def trainData():
	#input data
	t1 = classifyhelper.getTrainingData("../../data/note_train/notes1.png")
	t2 = classifyhelper.getTrainingData("../../data/note_train/img2.png")
	t3 = classifyhelper.getTrainingData("../../data/note_train/img3.jpg")
	
	#SHARP, FLAT, NATURAL DATA THROW IT IN
	# t4 = getTrainingData("../../data/accidental_train/img1.jpg")

	# put data in array of size 27 x 4
	cells1 = [np.hsplit(row, 4) for row in np.vsplit(t1, 9)]
	cells2 = [np.hsplit(row, 4) for row in np.vsplit(t2, 9)]
	cells3 = [np.hsplit(row, 4) for row in np.vsplit(t3, 9)]

	# FOR SHARPS AND FLATS
	# cells4 = [np.hsplit(row, 3) for row in np.vsplit(t1, 9)]

	cells = np.vstack((cells1, cells2, cells3))
	x = np.array(cells);

	# FOR SHARPS AND FLATS
	# cells_accidental = np.vstack(cells4);
	# x2 = np.array(cells_accidental)

	# train = x[:, :].reshape(-1, 400).astype(np.float32)
	train = classifyhelper.preprocess_hog(x)
	# train2 = x2[:,:].reshape(-1, 400).astype(np.float32)

	# Make labels for train data
	labels = np.array([WHOLE, HALF, QUARTER, EIGHTH])
	train_labels = np.tile(labels, 27)[:, np.newaxis]

	# labels_acc = np.array([SHARP, FLAT, NATURAL])
	# train_labels_acc = np.tile(labels, 27)[:, np.newaxis]

	# Initiate kNN, train the data, then test it with test data for k=1
	knn = cv2.ml.KNearest_create()
	knn.train(train,cv2.ml.ROW_SAMPLE, train_labels)
	# knn.train(train2, cv2.ml.ROW_SAMPLE, train_labels_acc)

	return knn


# # classify all given notes using the given knn
# # for testing purposes!!!! (includes pregiven testing labels)
# def classifyNotesDebug(testingData, testingLabels, knn):

# 	test = testingData[:,:].reshape(-1, 400).astype(np.float32)
# 	ret,result,neighbours,dist = knn.findNearest(test,k=3)
# 	print "result:\n", result
# 	print "expected: \n", testingLabels

# 	matches = result==testingLabels
# 	correct = np.count_nonzero(matches)
# 	accuracy = correct*100.0/result.size
# 	print "accuracy:", accuracy

# 	return result


# # classify all given notes using the given knn
# # not given testing labels to compare
# def classifyNotes(testingData, knn):

# 	test = testingData[:,:].reshape(-1, 400).astype(np.float32)
# 	ret,result,neighbours,dist = knn.findNearest(test,k=3)
# 	print "result:\n", result

# 	return result


# # converts image to grayscaled inverted image to be used in knn
# # input: name of image file
# def getTrainingData(imageFileName):
# 	i1 = Image.open(imageFileName).convert('RGB') # each image 20*20= 400
# 	cv_1 = np.array(i1)
# 	cv_1 = cv_1[:, :, ::-1].copy()
# 	cv_1 =  cv2.cvtColor(cv_1, cv2.COLOR_BGR2GRAY)
# 	t1 = cv2.adaptiveThreshold(cv_1,255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 3)	
# 	return t1


# # add training image to the cells array and makes np array
# # input: list of image filenames -- for testing purposes!!!
# def makeTestingData(imageNameList):
# 	cells = np.empty((0,20), int)
# 	for imageName in imageNameList:
# 		i1 = Image.open(imageName).convert('RGB') # each image 25 x 50 = 1250
# 		cv_1 = np.array(i1)
# 		cv_1 = cv_1[:, :, ::-1].copy()
# 		cv_1 = cv2.resize(cv_1, (20, 20))
# 		cv_1 =  cv2.cvtColor(cv_1, cv2.COLOR_BGR2GRAY)
# 		t1 = cv2.adaptiveThreshold(cv_1,255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 5)
# 		cells = np.append(cells, t1, axis=0)
# 	return cells


# # makes expected testing labels in column format
# # for testing purposes!!
# def makeTestingLabels(test_labels_list):
# 	test_labels =  np.array(test_labels_list)
# 	test_labels = np.tile(test_labels, 1)[:, np.newaxis]
# 	return test_labels # to make column vector


if __name__ == '__main__':

	#later, take in a bunch of bounding boxes, (which should be sized 20x20 and find out what they are)
	test_name_list = [ "../../data/whole.jpg",
					   "../../data/half.jpg",
					   "../../data/quarter.jpg",
					   "../../data/eighth.jpg"]

	test_labels_list = [WHOLE, HALF, QUARTER, EIGHTH]

	# TODO: must convert so that can get image matrx if given a bunch of bounding boxes
	# just need to look at original image and get sub section of it as a matrix

	knn = trainData()
	testing_data = classifyhelper.makeTestingData(test_name_list, 20, 20)
	testing_labels = classifyhelper.makeTestingLabels(test_labels_list)
	classifyhelper.classifyDebug(testing_data, testing_labels, knn)
	getLabels(result, clef_dictionary)
	# cv_image_note = getNoteLength(cv_image)

