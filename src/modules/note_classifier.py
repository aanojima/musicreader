#receive pitch, need to identify the type
#second step: find accidentals
import cv2
import numpy as np
import classifyhelper
from common import *

# External Dependencies
from PIL import Image

WHOLE_RESULT = 1;
HALF_RESULT = 2;
QUARTER_RESULT = 4;
EIGHTH_RESULT = 8;
SHARP_RESULT = 9;
FLAT_RESULT = 10;
NATURAL_RESULT = 11;

note_dictionary = {1: NoteLabel.WHOLE,
 	 			   2: NoteLabel.HALF,
 	 			   4: NoteLabel.QUARTER,
 	 			   8: NoteLabel.EIGHTH,
 	 			   9: AccidentalLabel.SHARP, 
 	 			   10: AccidentalLabel.FLAT,
 	 			   11: AccidentalLabel.NATURAL}

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
	t1 = classifyhelper.getTrainingData("../data/note_train/notes1.png")
	t2 = classifyhelper.getTrainingData("../data/note_train/img2.png")
	t3 = classifyhelper.getTrainingData("../data/note_train/img3.jpg")
	#SHARP, FLAT, NATURAL DATA THROW IT IN
	t4 = classifyhelper.getTrainingData("../data/accidental_train/acc1.jpg")
	t5 = classifyhelper.getTrainingData("../data/accidental_train/acc2.jpg")

	# put data in array of size 27 x 4
	cells1 = [np.hsplit(row, 4) for row in np.vsplit(t1, 9)]
	cells2 = [np.hsplit(row, 4) for row in np.vsplit(t2, 9)]
	cells3 = [np.hsplit(row, 4) for row in np.vsplit(t3, 9)]
	# FOR SHARPS AND FLATS
	cells4 = [np.hsplit(row, 6) for row in np.vsplit(t4, 9)]
	cells5 = [np.hsplit(row, 7) for row in np.vsplit(t5, 9)]

	cells = np.vstack((cells1, cells2, cells3))
	x = np.array(cells);
	# FOR SHARPS AND FLATS
	x2 = np.array(cells4)
	x3 = np.array(cells5)
	
	# preprocess testing data
	testing_data = classifyhelper.center_data(x)
	testing_data2 = classifyhelper.center_data(x2)
	testing_data3 = classifyhelper.center_data(x3)

	train = classifyhelper.preprocess_hog(testing_data)
	train2 = classifyhelper.preprocess_hog(testing_data2)
	train3 = classifyhelper.preprocess_hog(testing_data3)


	# Make labels for train data
	labels = np.array([WHOLE_RESULT, HALF_RESULT, QUARTER_RESULT, EIGHTH_RESULT])
	train_labels = np.tile(labels, 27)[:, np.newaxis]

	labels_acc = np.array([SHARP_RESULT, FLAT_RESULT, SHARP_RESULT, FLAT_RESULT, SHARP_RESULT, FLAT_RESULT])
	train_labels_acc = np.tile(labels_acc, 9)[:, np.newaxis]

	labels_acc2 = np.array([NATURAL_RESULT, NATURAL_RESULT, NATURAL_RESULT, QUARTER_RESULT, QUARTER_RESULT, HALF_RESULT, HALF_RESULT])
	train_labels_acc2 = np.tile(labels_acc2, 9)[:, np.newaxis]

	
	# combine training data into one array
	# combine training labels into one array
	train_length = train.shape[0] + train2.shape[0] + train3.shape[0]
	train_final = np.empty((train_length, train.shape[1]), np.float32)
	train_labels_final = np.empty((train_length, 1), np.float32)
	

	len1 = train.shape[0]
	len2 = train2.shape[0]
	train_final[:len1,:]= train
	train_final[len1: len1+ len2, :] = train2
	train_final[len1 + len2:, :] = train3
	
	train_labels_final[:len1,:]= train_labels
	train_labels_final[len1: len1+ len2, :] = train_labels_acc
	train_labels_final[len1 + len2:, :] = train_labels_acc2


	# Initiate kNN, train the data, then test it with test data for k=1
	knn = cv2.ml.KNearest_create()
	knn.train(train_final,cv2.ml.ROW_SAMPLE, train_labels_final)
	return knn


if __name__ == '__main__':

	#later, take in a bunch of bounding boxes, (which should be sized 20x20 and find out what they are)
	test_name_list = [ "../../data/whole.jpg",
					   "../../data/half.jpg",
					   "../../data/quarter.jpg",
					   "../../data/eighth.jpg",
					   "../../data/whole1.jpg",
					   "../../data/half1.jpg",
					   "../../data/quarter1.jpg",
					   "../../data/eighth1.jpg",
					   "../../data/sharp.jpg",
					   "../../data/flat.jpg",
					   "../../data/natural.jpg",
					   "../../data/sharp1.jpg",
					   "../../data/flat1.jpg",
					   "../../data/natural1.jpg",
					   ]

	test_labels_list = [WHOLE_RESULT, HALF_RESULT, QUARTER_RESULT, EIGHTH_RESULT, 
						WHOLE_RESULT, HALF_RESULT, QUARTER_RESULT, EIGHTH_RESULT, 
						SHARP_RESULT, FLAT_RESULT, NATURAL_RESULT, 
						SHARP_RESULT, FLAT_RESULT, NATURAL_RESULT]

	# TODO: must convert so that can get image matrx if given a bunch of bounding boxes
	# just need to look at original image and get sub section of it as a matrix

	knn = trainData()
	print "make testng data now"
	testing_data = classifyhelper.makeTestingData(test_name_list, 20, 20)
	testing_labels = classifyhelper.makeTestingLabels(test_labels_list)
	result = classifyhelper.classifyDebug(testing_data, testing_labels, knn)
	classifyhelper.getLabels(result, note_dictionary)

