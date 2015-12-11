import cv2
import numpy as np
from numpy.linalg import norm
from PIL import Image

from util import * 

# expect input to just be (length, 20, 20) array
def preprocess_hog(x):

    samples = []
    numrows = x.shape[0]
    for row in range(0,numrows):
    		img = x[row]
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

# centers given image and produces suare image
def center_data(x):
	samples = []
	numrows = x.shape[0]
	numcols = x.shape[1]
	for row in range(0,numrows):
		for col in range(0,numcols):
			img = x[row][col]
			img = img.astype(np.uint8)

			bb_array, note_head_image = findBoundingBox(img)
			crop_img = get_image_from_bb(bb_array[0], img)
			pad_image = pad_to_square(crop_img)
			resize_image = cv2.resize(pad_image, (20,20)) # 20 ARE CONSTANTS

			samples.append(resize_image)
	return np.float32(samples)


# classify all given notes using the given knn
# for testing purposes!!!! (includes pregiven testing labels)
def classifyDebug(testingData, testingLabels, knn):
	testing_data = center_data(testingData)
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
	testing_data = center_data(testingData)
	testing_data = preprocess_hog(testing_data)
	ret,result,neighbours,dist = knn.findNearest(testing_data,k=3)
	# ret,result,neighbours,dist = knn.findNearest(test,k=3)
	# print "result:\n", result

	return result


# gets image and converts image to grayscale
# input: name of image file
def getTrainingData(imageFileName):
	i1 = Image.open(imageFileName).convert('RGB') # each image 20*20= 400
	cv_1 = np.array(i1)
	cv_1 = cv_1[:, :, ::-1].copy()
	cv_1 =  cv2.cvtColor(cv_1, cv2.COLOR_BGR2GRAY)
	return cv_1


# add training image to the cells array and makes np array
# input: list of image filenames, and desired width and height of testing images
def makeTestingData(imageNameList, width, height):
	cells = np.empty((len(imageNameList), 1, width,height), np.float32)

	for index in range(0, len(imageNameList)):
		imageName = imageNameList[index]
		i1 = Image.open(imageName).convert('RGB') # each image 25 x 50 = 1250
		cv_1 = np.array(i1)
		cv_1 = cv_1[:, :, ::-1].copy()
		cv_1 = cv2.resize(cv_1, (width, height))
		cv_1 =  cv2.cvtColor(cv_1, cv2.COLOR_BGR2GRAY)
		t1 = cv_1
		cells[index][0] = t1
	
	cells = np.array(cells)

	return cells

def makeInputData(image, height, width):
	# cv2.destroyAllWindows()
	# cv2.imshow('input data', image)
	# cv2.waitKey(0)
	cells = np.empty((1,1,width,height), np.float32)
	# image =  cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	cells[0][0] = image
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
	result = result.astype(int)
 	labels = []
	for i in range(len(result)):
		index = result.item(i)
		labels.append(dictionary[index])
	return labels

# derived from object_segment3 code to get bounding box coordinates
def findBoundingBox(rawImage):
    thresh2 = cv2.adaptiveThreshold(rawImage,255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 5)

    image, contours, hierarchy = cv2.findContours(thresh2,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    
    out = rawImage.copy()
    
    cv2.drawContours(out, contours, -1, (0,0,0), 2)

    # fit bounding boxes to polygons
    out2 = rawImage.copy()
    bb_rand_array = []

    for i in range(len(contours)):
        cnt = contours[i]
        x,y,w,h = cv2.boundingRect(cnt)

        if len(bb_rand_array) == 0:
        	bb_rand_array.append((x,y,w,h))
        	cv2.rectangle(out2,(x,y),(x+w,y+h),(0,255,0),2)  
        else:
        	if w > bb_rand_array[0][2] and h > bb_rand_array[0][3]:
        		bb_rand_array[0] = ((x,y,w,h))
        		cv2.rectangle(out2,(x,y),(x+w,y+h),(0,255,0),2)  

    # re-order bounding box array from left to right
    bb_array = sorted(bb_rand_array,key=lambda x: x[0])
        
    return bb_array, out
