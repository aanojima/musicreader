'''
Classifier:
Sort the list of bounding boxes into their appropriate categories
'''

import cv2
import object_segment3

def classifier(bounding_box_array):
    i = 0
    clef = bounding_box_array[i]
    i+=1
    # no patternizable way to identify time signature
    num_of_key = 3
    key_signature = []
    for k in range(1,i + num_of_key):
        key_signature.append(bounding_box_array[k])
        i+=1
    # check y - coordinates in order to sort top and bottom
    if bounding_box_array[i][1] > bounding_box_array[i+1][1]:
        time_signature_top = bounding_box_array[num_of_key + 1]
        time_signature_bottom = bounding_box_array[num_of_key + 2]
    else:
        time_signature_top = bounding_box_array[num_of_key + 2]
        time_signature_bottom = bounding_box_array[num_of_key + 1]
        
    i+=1
    note = []
    for j in range(i,len(bounding_box_array)):
        note.append(bounding_box_array[j])

    return (clef, key_signature, time_signature_top, time_signature_bottom, note)

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
