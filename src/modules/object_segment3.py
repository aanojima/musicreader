'''
Object segmentation for music notes
1) Threshold
2) Find contours
3) Fit contours to polygon
4) Return bounding boxes

In main: Call 'findObjects(rawImage)' (image that has already been read by imread)
Returns: list of bounding box tuples in format --> (x,y,w,h)
'''
import numpy as np
import cv2

def findSmallerRect(bb1, bb2, index1, index2):
    x1, y1, w1, h1 = bb1
    x2, y2, w2, h2 = bb2
    if (w1*h1) > (w2*h2):
        return index2
    else:
        return index1

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
            #cv2.rectangle(out2,(x,y),(x+w,y+h),(0,255,0),2)
            
    to_delete = []
    for i in range(len(bb_rand_array)-1):
        for j in range(i+1, len(bb_rand_array)):
            RectA_X1 = bb_rand_array[i][0]
            RectA_X2 = bb_rand_array[i][0] + bb_rand_array[i][2]
            RectB_X2 = bb_rand_array[j][0] + bb_rand_array[j][2]
            RectA_Y1 = bb_rand_array[i][1]
            RectA_Y2 = bb_rand_array[i][1] + bb_rand_array[i][3]
            RectB_X1 = bb_rand_array[j][0]
            RectB_Y1 = bb_rand_array[j][1]
            RectB_Y2 = bb_rand_array[j][1] + bb_rand_array[j][3]
            
            if (RectA_X1 < RectB_X2 and RectA_X2 > RectB_X1 and
            RectA_Y1 < RectB_Y2 and RectA_Y2 > RectB_Y1):
                print "Overlap!"
                # TODO: Intelligent Merging?
                # if one is completely a subset of the other
                to_delete.append(findSmallerRect(bb_rand_array[i],bb_rand_array[j],i,j))
                
    to_delete = list(set(to_delete))
    to_delete.sort()
    to_delete.reverse()
    print to_delete
    for ind in to_delete:
        print ind
        bb_rand_array.pop(ind)
        
    # re-order bounding box array from left to right
    bb_array = sorted(bb_rand_array,key=lambda x: x[0])

    # display boxes
    for box in bb_array:
        x,y,w,h = box
        cv2.rectangle(out2,(x,y),(x+w,y+h),(0,255,0),2)
        
    cv2.imshow('All Contours', out2)
    cv2.waitKey(0)

    print bb_array
    return bb_array

if __name__ == '__main__':
    img = cv2.imread('../../data/sample_line.jpg',0)
    #small_img = cv2.resize(img, (0,0), fx=0.1, fy=0.1)
    cv2.imshow('original', img)
    cv2.waitKey(0)
    print findObjects(img)
    cv2.destroyAllWindows()
