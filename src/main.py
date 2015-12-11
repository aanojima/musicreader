from models.SheetMusic import *
from modules.staves import *
from modules.object_segment3 import *
from modules.note import *

from modules.classifier import *
from modules.common import *

import numpy as np
import cv2

# External Dependencies 64
from PIL import Image

def main():
	image = Image.open("../data/blank sheet music.png").convert('RGB')
	# image = Image.open("../data/sheet_music2.png").convert('RGB')
	cv_image = np.array(image)
	cv_image = cv_image[:, :, ::-1].copy()
	width = cv_image.shape[1]
	height = cv_image.shape[0]
	cv_image = cv2.resize(cv_image, (width, height)) # TODO: Play around with size
	cv_image = cv2.imread('../data/handwritten-test.png',0)
	sheet_images = create_sheet_images(cv_image)
	cv2.imshow("No Staves Result", sheet_images.vertical)
	cv2.waitKey(0)
	bb_array, note_head_image = findObjects(sheet_images.vertical)
	sheet_images.set_note_head(note_head_image)
	symbol_array = classifier(bb_array)
	sheet_music = SheetMusic(sheet_images, symbol_array)

	# TODO: What do we do with the sheet_images (up to here is the MVP)
	cv2.waitKey(0)

if __name__ == '__main__':
	main() 