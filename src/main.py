from models.SheetMusic import SheetMusic
from models.Accidental import Accidental
from models.Clef import Clef
from models.Note import Note
from models.TimeSignature import TimeSignature
from modules.staves import *
from modules.object_segment3 import *
from modules.note import *
from model_constructor import *
import numpy as np
import cv2

# External Dependencies
from PIL import Image

def main():
	cv2.destroyAllWindows()
	# a = Accidental()
	# c = Clef()
	# n = Note()
	# ts = TimeSignature()
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
	# [ { 'type' : TYPE, 'box' : BBOX, 'data' : DATA }, ... ]
	symbol_array = [ { 'type' : EIGHTH, 'bbox' : bb_array[0], 'data' : None } ]
	sheet_music = SheetMusic()
	for key in symbol_array:
		symbol = symbol_array[key]
		if symbol['type'] in [BASS_CLEF, TREBLE_CLEF]: # TODO: Create Enums/Constants for these
			# Ignore Getting Line ==> straight to data model construction
			clef = create_model(symbol['type'])
			sheet.set_clef(clef)
			continue
		elif symbol['type'] == TIME_SIGNATURE_COUNT:
			# TODO: CLASSIFIER should return COUNT or TYPE
			ts_count = create_model(symbol['type'], symbol['data'])
			sheet.set_time_sinature_count(ts_count)
			continue
		elif symbol['type'] == TIME_SIGNATURE_TYPE:
			ts_type = create_model(symbol['type'], symbol['data'])
			sheet.set_time_signature_type(ts_type)
		else:
			line_index = sheet.get_line(symbol['type'], symbol['box'])
			sheet.add_note(symbol['type'], line_index)

	# TODO: What do we do with the sheet

if __name__ == '__main__':
	main()