'''
Utilities for:
1) retrieving image based on bounding box (might be a duplicate to Aaron?)
2) padding image with white if not square
'''
import cv2
import object_segment3

def get_image_from_bb(bb_tuple, original_image):
    (x,y,w,h) = bb_tuple
    crop_img = img[y:y+h, x:x+w] # Crop from x, y, w, h
    # NOTE: it is img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]
    cv2.imshow("cropped", crop_img)
    cv2.waitKey(0)
    return crop_img

def pad_to_square(img):
    (y, x) = img.shape
    if x < y:
        x_border_left = (y - x) / 2
        x_border_right = y - (x + x_border_left)
        img2 = cv2.copyMakeBorder(img,0,0,x_border_left,x_border_right,cv2.BORDER_CONSTANT)
    else:
        y_border_top = (x - y) / 2
        y_border_bottom = x - (y + y_border_top)
        img2 = cv2.copyMakeBorder(img,y_border_top,y_border_bottom,0,0,cv2.BORDER_CONSTANT)
    return img2

if __name__ == '__main__':
    img = cv2.imread("../../data/sample_line.jpg", 0)
    bb_array = object_segment3.findObjects(img)
    crop_img = get_image_from_bb(bb_array[7], img)
    print crop_img.shape
    pad_image = pad_to_square(crop_img)
    print pad_image.shape
    cv2.imshow('padded', pad_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
