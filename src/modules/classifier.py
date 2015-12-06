'''
Classifier:
Sort the list of bounding boxes into their appropriate categories
'''

import cv2
import object_segment3

from modules.common import *

def classifier(bounding_box_array):
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
    # no patternizable way to identify time signature
    num_of_key = 3
    key_signature = symbol.copy()
    key_signature['type'] = Symbol.KEY_SIGNATURE
    key_signature['box'] = []
    key_signature['label'] = []
    key_signature['data'] = []
    # key_signature['type'] = 
    for k in range(1,i + num_of_key):
        key_signature['box'].append(bounding_box_array[k])
        # TODO: Add label (sharp, flat, etc)
        i+=1


    # Time Signature
    # check y - coordinates in order to sort top and bottom
    time_signature_count = symbol.copy()
    time_signature_count['type'] = Symbol.TimeSignature
    time_signature_count['label'] = TimeSignatureLabel.COUNT
    time_signature_type = symbol.copy()
    time_signature_type['type'] = Symbol.TimeSignature
    time_signature_type['label'] = TimeSignatureLabel.TYPE
    if bounding_box_array[i][1] > bounding_box_array[i+1][1]:
        time_signature_count['box'] = bounding_box_array[num_of_key + 1]
        time_signature_type['box'] = bounding_box_array[num_of_key + 2]
    else:
        time_signature_count['box'] = bounding_box_array[num_of_key + 2]
        time_signature_type['box'] = bounding_box_array[num_of_key + 1]
    # TODO: Need data for time signature data (count or type)
        
    i+=1
    notes = []
    for j in range(i,len(bounding_box_array)):
        note = symbol.copy()
        note['type'] = Symbol.NOTE
        note['box'] = bounding_box_array[j]
        # TODO: Note label (quarter, eight, etc)
        notes.append(note)

    return [clef, key_signature, time_signature_count, time_signature_type] + notes

if __name__ == '__main__':
    img = cv2.imread('../../data/sample_line.jpg',0)
    #small_img = cv2.resize(img, (0,0), fx=0.1, fy=0.1)
    cv2.imshow('original', img)
    cv2.waitKey(0)
    bb_list = object_segment3.findObjects(img)
    clef, key, time_s_1, time_s_2, note = classifier(bb_list)
    print clef
    print key
    print time_s_1
    print time_s_2
    print note
    cv2.destroyAllWindows()
