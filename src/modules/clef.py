#receive pitch, need to identify the type
#second step: find accidentals
import cv2
import numpy as np
from numpy.linalg import norm


# External Dependencies
from PIL import Image

TREBLE = 1;
BASS = 2;


# Given: Image and list of bounding boxes
# assume note pitch has already been calculated
# assume given a note without any lines--using just plain handwritten notes for now
def getClef(rawImage, clefBoundingBox):
	# if adding the white pixel buffers, notesList may even be a list of matrices representing
	# each bounding box--in this case, just run our threshold on that data
	knn = trainData()

	t1 = cv2.adaptiveThreshold(note,255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 5)
	cells = np.append(cells, t1, axis=0)

	result = classifyClef(cells, testing_labels, knn)

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
	t1 = getTrainingData("../../data/clef_train/clef1.jpg")
	# t2 = getTrainingData("../../data/clef_train/img2.png")
	# t3 = getTrainingData("../../data/clef_train/img3.jpg")

	# put data in array of size 27 x 4
	cells1 = [np.hsplit(row, 6) for row in np.vsplit(t1, 10)]
	# cells2 = [np.hsplit(row, 2) for row in np.vsplit(t2, 9)]
	# cells3 = [np.hsplit(row, 2) for row in np.vsplit(t3, 9)]

	
	# cells1_hogg = preprocess_hog(cells1)

	x = np.array(cells1);
	# hogg features to classify
	x1 = preprocess_hog(x)
	print x1.shape

	# first 22 samples for training, last 5 samples for testing
	train = x1[:, :].reshape(-1, 400).astype(np.float32)

	# Make labels for train data
	labels = np.array([TREBLE, BASS, TREBLE, BASS, TREBLE, BASS])
	train_labels = np.tile(labels, 10)[:, np.newaxis]

	# Initiate kNN, train the data, then test it with test data for k=1
	knn = cv2.ml.KNearest_create()
	knn.train(train,cv2.ml.ROW_SAMPLE, train_labels)

	return knn

def preprocess_hog(digits):
    samples = []
    for img in digits:
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
    return np.float32(samples)

# classify all given notes using the given knn
# for testing purposes!!!! (includes pregiven testing labels)
def classifyClefDebug(testingData, testingLabels, knn):

	test = testingData[:,:].reshape(-1, 400).astype(np.float32)
	ret,result,neighbours,dist = knn.findNearest(test,k=3)
	print "result:\n", result
	print "expected: \n", testingLabels

	matches = result==testingLabels
	correct = np.count_nonzero(matches)
	accuracy = correct*100.0/result.size
	print "accuracy:", accuracy

	return result


# classify all given notes using the given knn
# not given testing labels to compare
def classifyClef(testingData, knn):

	test = testingData[:,:].reshape(-1, 400).astype(np.float32)
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
	return t1


# add training image to the cells array and makes np array
# input: list of image filenames -- for testing purposes!!!
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
# for testing purposes!!
def makeTestingLabels(test_labels_list):
	test_labels =  np.array(test_labels_list)
	test_labels = np.tile(test_labels, 1)[:, np.newaxis]
	return test_labels # to make column vector


if __name__ == '__main__':
	#later, take in a bunch of bounding boxes, (which should be sized 20x20 and find out what they are)
	test_name_list = ["../../data/cleftest1.jpg",
					   "../../data/cleftest2.jpg",
					   "../../data/cleftest3.jpg",
					   "../../data/cleftest4.jpg",]

	test_labels_list = [TREBLE, BASS, TREBLE, BASS]

	# TODO: must convert so that can get image matrx if given a bunch of bounding boxes
	# just need to look at original image and get sub section of it as a matrix

	knn = trainData()
	testing_data = makeTestingData(test_name_list)
	testing_data_hogg = preprocess_hog(testing_data)
	testing_labels = makeTestingLabels(test_labels_list)
	classifyClefDebug(testing_data, testing_labels, knn)
	# cv_image_note = getNoteLength(cv_image)

