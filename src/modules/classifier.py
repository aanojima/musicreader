'''
Classifier:
Sort the list of bounding boxes into their appropriate categories
'''

import cv2
import object_segment3
from note_classifier import *
from classifyhelper import *
from time_sig_ocr import *

from common import *

def general_classify(knn, ocr, bounding_box_array, sheet_images):
    # Symbol Format: { 'type' : TYPE, 'box' : BOX, 'label' : LABEL, 'data' : DATA }
    symbol = {
        'type' : None,
        'box' : None,
        'label' : None,
        'data' : None
    }

    # Clef
    i = 0
    clef = symbol.copy()
    clef['type'] = Symbol.CLEF
    clef['box'] = bounding_box_array[i]
    clef['label'] = ClefLabel.TREBLE # TODO

    i+=1

    # Key Signature
    # TODO: no pat ternizable way to identify time signature
    num_of_key = 3
    key_signature = symbol.copy()
    key_signature['type'] = Symbol.KEY_SIGNATURE
    key_signature['box'] = []
    key_signature['label'] = [] 
    key_signature['data'] = []
    for k in range(1,i + num_of_key):
        (x,y,w,h) = bounding_box_array[k]
        key_signature['box'].append(bounding_box_array[k])
        img = sheet_images.vertical[y:y+h,x:x+w]
        data = makeInputData(img, w, h)
        ks_result = classify(data, knn)
        labels = getLabels(ks_result, note_dictionary)
        key_signature['label'].append(labels[0])
        i+=1

    # Time Signature
    # check y - coordinates in order to sort top and bottom
    time_signature_count = symbol.copy()
    time_signature_count['type'] = Symbol.TIME_SIGNATURE
    time_signature_count['label'] = TimeSignatureLabel.COUNT
    time_signature_type = symbol.copy()
    time_signature_type['type'] = Symbol.TIME_SIGNATURE
    time_signature_type['label'] = TimeSignatureLabel.TYPE
    if bounding_box_array[i][1] > bounding_box_array[i+1][1]:
        time_signature_count['box'] = bounding_box_array[num_of_key + 1]
        (x,y,w,h) = bounding_box_array[num_of_key + 1]
        img = sheet_images.vertical[y:y+h,x:x+w]
        data = get_digit_guess(ocr, img)[0]
        time_signature_count['data'] = data

        time_signature_type['box'] = bounding_box_array[num_of_key + 2]
        (x,y,w,h) = bounding_box_array[num_of_key + 2]
        img = sheet_images.vertical[y:y+h,x:x+w]
        data = get_digit_guess(ocr, img)[0]
        time_signature_type['data'] = data
    else:
        time_signature_count['box'] = bounding_box_array[num_of_key + 2]
        (x,y,w,h) = bounding_box_array[num_of_key + 2]
        img = sheet_images.vertical[y:y+h,x:x+w]
        data = get_digit_guess(ocr, img)[0]
        time_signature_count['data'] = data

        time_signature_type['box'] = bounding_box_array[num_of_key + 1]
        (x,y,w,h) = bounding_box_array[num_of_key + 1]
        img = sheet_images.vertical[y:y+h,x:x+w]
        data = get_digit_guess(ocr, img)[0]
        time_signature_type['data'] = data
    i+=2

    notes = []
    for j in range(i,len(bounding_box_array)):
        # NOTES OR ACCIDENTALS
        note = symbol.copy()
        note['type'] = Symbol.NOTE
        note['box'] = bounding_box_array[j]
        # TODO: Note label (quarter, eight, etc)

        (x,y,w,h) = bounding_box_array[j]
        img = sheet_images.vertical[y:y+h,x:x+w]
        data = makeInputData(img, w, h)
        ks_result = classify(data, knn)
        label = getLabels(ks_result, note_dictionary)[0]
        note['label'] = label
        notes.append(note)

    return [clef, key_signature, time_signature_type, time_signature_count] + notes

if __name__ == '__main__':
    img = cv2.imread('../../data/sample_line.jpg',0)
    #small_img = cv2.resize(img, (0,0), fx=0.1, fy=0.1)
    cv2.imshow('original', img)
    cv2.waitKey(0)
    bb_list = object_segment3.findObjects(img)
    clef, key, time_s_1, time_s_2, note = general_classify(bb_list)
    print clef
    print key
    print time_s_1
    print time_s_2
    print note
    cv2.destroyAllWindows()
