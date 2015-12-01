from PIL import Image
import cv2
import numpy as np

class Stave():
    def __init__(self):
        pass

def find_stave_lines(image):
    pass

def remove_stave_lines(image):
    # Check if image is loaded fine
    if not image.data:
        print "Problem loading image"
        return
    
    # Show source image
    # cv2.imshow("Sheet Music 1", image)

    # Transform source image to gray if it is not
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Show gray image
    # cv2.imshow("Gray Sheet Music 1", gray)

    # Apply adaptiveThreshold at the bitwise_not of gray, notice the ~ symbol
    bw = cv2.adaptiveThreshold(cv2.bitwise_not(gray), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)

    # Show the binary image
    # cv2.imshow("Binary", bw)

    # Create the images that will use to extract the horizontal and vertical lines
    horizontal = bw.copy()
    vertical = bw.copy()

    # Specify size on horizontal axis
    horizontalsize = horizontal.shape[1] / 30

    # Create structure element for extracting horizontal lines through morphology operations
    horizontal_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontalsize,1))

    # Apply morphology operations
    horizontal = cv2.erode(horizontal, horizontal_structure, (-1, -1))
    horizontal = cv2.dilate(horizontal, horizontal_structure, (-1, -1))

    # Show extracted horizontal lines
    # cv2.imshow("Horizontal", horizontal)

    # Specify size on vertical axis
    verticalsize = vertical.shape[0] / 30

    # Create structure element for extracting vertical lines through morphology operations
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1,verticalsize));
    
    # Apply morphology operations
    vertical = cv2.erode(vertical, verticalStructure, (-1, -1));
    vertical = cv2.dilate(vertical, verticalStructure, (-1, -1));
    
    # Show extracted vertical lines
    # cv2.imshow("vertical", vertical);
    
    # Inverse vertical image
    vertical = cv2.bitwise_not(vertical);
    # cv2.imshow("vertical_bit", vertical);

    return vertical
    
    # Extract edges and smooth image according to the logic
    # 1. extract edges
    # 2. dilate(edges)
    # 3. src.copyTo(smooth)
    # 4. blur smooth img
    # 5. smooth.copyTo(src, edges)
    
    # Step 1
    edges = cv2.adaptiveThreshold(vertical, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, -2);
    # cv2.imshow("edges", edges);
    
    # Step 2
    kernel = np.ones( (2, 2), np.uint8); # TODO
    edges = cv2.dilate(edges, kernel);
    # cv2.imshow("dilate", edges);
    
    # Step 3
    smooth = vertical.copy(); 
    
    # Step 4
    smooth = cv2.blur(smooth, (2, 2));
    
    # Step 5
    vertical = cv2.bitwise_and(smooth, smooth, mask=edges)
    
    # Show final result
    # cv2.imshow("smooth", vertical);

    return vertical

def identify_line(box):
    print "TODO: Implement identify_line"