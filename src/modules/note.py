#receive pitch, need to identify the type
#second step: find accidentals
import cv2
import numpy as np

# External Dependencies
from PIL import Image

# Given: Image and bounding box; assume note pitch has already been calculated
# assume given a note without any lines--using just plain handwritten notes for now
def getNoteLength(rawImage):
	# threshold
    ret,thresh2 = cv2.threshold(rawImage,127,255,cv2.THRESH_BINARY_INV)
    cv2.imshow('thresholded', thresh2)
    cv2.waitKey(0)

def trainData():
	# training data image name list
	train_name_list = ["../../data/test-whole.png",
					   "../../data/test-half.png",
					   "../../data/test-quarter.png",
					   "../../data/test-eighth.png"]

	# testing data image name list
	test_name_list = ["../../data/whole-note.png",
					   "../../data/half-note.png",
					   "../../data/quarter-note.png",
					   "../../data/eighth-note.png"]


	# make arrays of data
	train_cells = addData(train_name_list) # size ()
	test_cells = addData(test_name_list)

	# make Numpy arrays of size (1, 50, 25)	
	x1 = np.array(train_cells)
	x2 = np.array(test_cells)

	print x1.shape
	print x2.shape

	# Prepare training and testing data by flattening into single row of 
	# pixels per image --> create a feature set
	train = x1[:].reshape(-1, 1250).astype(np.float32) # 1 x 1250 pixels
	test = x2[:].reshape(-1, 1250).astype(np.float32)
	print train.shape
	print test.shape

	# Make labels for train and test data
	train_labels = np.array(['whole', 'half', 'quarter', 'eighth']);
	test_labels = train_labels.copy()

	# Initiate kNN, train the data, then test it with test data for k=1
	knn = cv2.ml.KNearest()
	knn.train(train,train_labels)
	ret,result,neighbours,dist = knn.find_nearest(test,k=5)
	# DOESNT WORK WHY

	# Now we check the accuracy of classification
	# For that, compare the result with test_labels and check which are wrong
	matches = result==test_labels
	correct = np.count_nonzero(matches)
	accuracy = correct*100.0/result.size
	print accuracy



#add training image to the cells array
def addData(imageNameList):
	cells = []
	for imageName in imageNameList:
		i1 = Image.open(imageName).convert('RGB') # each image 25 x 50 = 1250
		cv_1 = np.array(i1)
		cv_1 = cv_1[:, :, ::-1].copy()
		ret,t1 = cv2.threshold(cv_1,127,255,cv2.THRESH_BINARY_INV)
		t1 =  cv2.cvtColor(t1, cv2.COLOR_BGR2GRAY)
		cells.append(t1);
	return cells



if __name__ == '__main__':
    # cv2.destroyAllWindows()
	image = Image.open("../../data/test-quarter.png").convert('RGB')
	cv_image = np.array(image)
	cv_image = cv_image[:, :, ::-1].copy()
	width = cv_image.shape[1]
	height = cv_image.shape[0]
	cv_image = cv2.resize(cv_image, (width, height)) # TODO: Play around with size
	cv2.imshow("Original Image", cv_image)
	# cv2.waitKey(0)
	trainData()
	# cv_image_note = getNoteLength(cv_image)

