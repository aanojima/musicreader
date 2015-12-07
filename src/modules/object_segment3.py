'''
Object segmentation for music notes
1) Threshold
2) Find contours
3) Fit contours to polygon
4) Return bounding boxes
'''
import numpy as np
import cv2

def findObjects(rawImage):
    # threshold
    ret,thresh2 = cv2.threshold(rawImage,127,255,cv2.THRESH_BINARY_INV)
    
    cv2.imshow('thresholded', thresh2)
    cv2.waitKey(0)
    # find contours based on thresholding
    image, contours, hierarchy = cv2.findContours(thresh2,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    out = rawImage.copy()
    
    cv2.drawContours(out, contours, -1, (255,0,0), 2)
    cv2.imshow('contours', out)
    cv2.waitKey(0)

    # fit bounding boxes to polygons
    out2 = rawImage.copy()
    bb_rand_array = []
    for i in range(len(contours)):
        cnt = contours[i]
        x,y,w,h = cv2.boundingRect(cnt)
        # filter out artifacts by size
        # TODO: Add additional clean up on bounding boxes
        if w > 10 and h > 10:
            bb_rand_array.append((x,y,w,h))
            cv2.rectangle(out2,(x,y),(x+w,y+h),(0,255,0),2)  

    # re-order bounding box array from left to right
    bb_array = sorted(bb_rand_array,key=lambda x: x[0])
        
    cv2.imshow('All Contours', out2)
    cv2.waitKey(0)

    return bb_array, out

if __name__ == '__main__':
    img = cv2.imread('../../data/sample_line.jpg',0)
    #small_img = cv2.resize(img, (0,0), fx=0.1, fy=0.1)
    cv2.imshow('original', img)
    cv2.waitKey(0)
    print findObjects(img)
    cv2.destroyAllWindows()
