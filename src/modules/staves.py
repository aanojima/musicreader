from PIL import Image
import cv2
import numpy as np
import math
from modules.note import WHOLE, HALF, QUARTER, EIGHTH
# TODO: Same import for Sharp, Natural, Flat

class Sheet():
    
    horizontal = None
    vertical = None

    def __init__(self, horizontal, vertical):
        self.horizontal = horizontal
        self.vertical = vertical
        # Create Index Mapping

    def get_line(self, object_type, box):
        # For Sharp, Natural, Flat, Eight/Quarter/Half Notes, Whole Note
        (x0, y0, w, h) = box

        # Default to Middle
        x = x0 + 0.5*w
        y = y0 + 0.5*w
        
        if object_type == EIGHTH:
            # Near Bottom, Near Left
            x = x0 + 0.25*w
            y = y0 + 0.8*h
        elif object_type in [QUARTER, HALF]:
            # Near Bottom
            x = x0 + 0.5*w
            y = y0 + 0.8*h
        # TODO: Sharp, Natural, Flat
            

        # At column x, find all white spots
        column = self.horizontal[:,x]
        indices = []
        for i in range(len(column)):
            if column[i] == 255:
                indices.append(i)

        for j in range(len(indices) - 1):
            # TODO: What if inbetween different groups of lines?
            a = indices[j]
            b = indices[j+1]
            mid = float(a+b) / 2.0
            indices.append(mid)

        indices = sorted(indices)

        pixel = -1
        distance = float('infinity')
        for k in range(len(indices)):
            diff = abs(indices[k]-y)
            if diff <= distance: #TODO (Could be just less than)
                # Still decreasing
                distance = diff
                pixel = indices[k]
            elif diff > distance:
                # Stop (will only increase more)
                break

        # TODO: return line index instead of pixel
        return pixel

def create_sheet(image):
    # Check if image is loaded fine
    if not image.data:
        print "Problem loading image"
        return
    
    # Show source image
    cv2.imshow("Sheet Music 1", image)
    cv2.waitKey(0)

    # Transform source image to gray if it is not
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Show gray image
    cv2.imshow("Gray Sheet Music 1", gray)
    cv2.waitKey(0)

    # Apply adaptiveThreshold at the bitwise_not of gray, notice the ~ symbol
    ret, bw = cv2.threshold(cv2.bitwise_not(gray), 127, 255, cv2.THRESH_BINARY)

    # Show the binary image
    cv2.imshow("Binary", bw)
    cv2.waitKey(0)

    # Create the images that will use to extract the horizontal and vertical lines
    horizontal = bw.copy()
    vertical = bw.copy()

    # Specify size on horizontal axis
    horizontalsize = horizontal.shape[1] / 30

    # Create structure element for extracting horizontal lines through morphology operations
    horizontal_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontalsize,1))

    # Apply morphology operations
    horizontal = cv2.erode(horizontal, horizontal_structure, (-1, -1))
    # horizontal = cv2.dilate(horizontal, horizontal_structure, (-1, -1))

    # Show extracted horizontal lines
    cv2.imshow("Horizontal", horizontal)
    cv2.waitKey(0)

    # Specify size on vertical axis
    verticalsize = vertical.shape[0] / 30

    # Create structure element for extracting vertical lines through morphology operations
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1,verticalsize));
    
    # Apply morphology operations
    vertical = cv2.erode(vertical, verticalStructure, (-1, -1));
    vertical = cv2.dilate(vertical, verticalStructure, (-1, -1));
    
    # Show extracted vertical lines
    cv2.imshow("vertical", vertical);
    cv2.waitKey(0)
    
    # Inverse vertical image
    vertical = cv2.bitwise_not(vertical);
    cv2.imshow("vertical_bit", vertical);
    cv2.waitKey(0)
    
    # Extract edges and smooth image according to the logic
    # 1. extract edges
    # 2. dilate(edges)
    # 3. src.copyTo(smooth)
    # 4. blur smooth img
    # 5. smooth.copyTo(src, edges)
    
    # Step 1
    edges = cv2.adaptiveThreshold(vertical, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, -2);
    cv2.imshow("edges", edges);
    cv2.waitKey(0)
    
    # Step 2
    kernel = np.ones( (2, 2), np.uint8); # TODO
    edges = cv2.dilate(edges, kernel);
    cv2.imshow("dilate", edges);
    cv2.waitKey(0)
    
    # Step 3
    smooth = vertical.copy(); 
    
    # Step 4
    smooth = cv2.blur(smooth, (2, 2));
    
    # Step 5
    smooth = cv2.bitwise_or(smooth, smooth, mask=edges)
    
    # Show final result
    cv2.imshow("smooth", smooth);
    cv2.waitKey(0)

    sheet = Sheet(horizontal, vertical)

    return sheet

def identify_line(box, center):
    print "TODO: Implement identify_line"