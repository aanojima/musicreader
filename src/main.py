from models.SheetMusic import *
from modules.staves import *
from modules.object_segment3 import *
from modules.note import *

import numpy as np
import cv2

# External Dependencies
from PIL import Image

def main():
	# image = Image.open("../data/handwritten-test.png").convert('RGB')
	image = Image.open("../data/sheet_music2.png").convert('RGB')
	cv_image = np.array(image)
	cv_image = cv_image[:, :, ::-1].copy()
	width = cv_image.shape[1]
	height = cv_image.shape[0]
	cv_image = cv2.resize(cv_image, (width, height)) # TODO: Play around with size
	sheet = create_sheet(cv_image)
	cv2.imshow("No Staves Result", sheet.vertical)
	cv2.waitKey(0)
	bb_array = findObjects(sheet.vertical)
	# TODO: Pass bb_array to CLASSIFIER to get object type (and any necessary data for object_type)
	# [ { 'type' : TYPE, 'box' : BBOX, 'label' : LABEL, 'data' : DATA }, ... ]
	symbol_array = [ { 'type' : CLEF, 'bbox' : bb_array[0], 'label' : BASS 'data' : None } ] # EXAMPLE
	sheet_music = SheetMusic(symbol_array)

	# TODO: What do we do with the sheet (up to here is the MVP)

if __name__ == '__main__':
	main()