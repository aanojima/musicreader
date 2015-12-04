from models.Accidental import Accidental
from models.Clef import Clef
from models.Note import Note
from models.TimeSignature import TimeSignature
from modules.staves import *
from modules.object_segment3 import *
import numpy as np
import cv2

# External Dependencies
from PIL import Image

def main():
	cv2.destroyAllWindows()
	print "Hello World"
	# a = Accidental()
	# c = Clef()
	# n = Note()
	# ts = TimeSignature()
	image = Image.open("../data/handwritten-test.png").convert('RGB')
	# image = Image.open("../data/sheet_music2.png").convert('RGB')
	cv_image = np.array(image)
	cv_image = cv_image[:, :, ::-1].copy()
	width = cv_image.shape[1]
	height = cv_image.shape[0]
	cv_image = cv2.resize(cv_image, (width, height)) # TODO: Play around with size
	cv_image_no_staves = remove_stave_lines(cv_image)
	cv2.imshow("No Staves Result", cv_image_no_staves)
	cv2.waitKey(0)
	bb_array = findObjects(cv_image_no_staves)

if __name__ == '__main__':
	main()