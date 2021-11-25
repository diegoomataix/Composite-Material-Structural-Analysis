#%% Data loading

import pytesseract
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
sample = r'/content/gdrive/My Drive/sample.png'
read_image = cv2.imread(sample, 0)

#%% Detecting the cells
convert_bin, grey_scale = cv2.threshold(
    read_image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
grey_scale = 255-grey_scale

grey_graph=plt.imshow(grey_scale, cmap = 'gray')

plt.show()

# For the purpose of converting an image to excel we need to first detect the cells that are the horizontal and 
# vertical lines that form the cells. To do this, we need to first convert the image to binary and turn them
# into grayscale with OpenCV.

# Here, we have converted the image into a binary format. Now let us define two kernels to extract the 
# horizontal and vertical lines from these cells.

# For the horizontal lines, we will do the following. We will first get the entire image dimensions and 
# then using the OpenCV structural element function we will get the horizontal lines.

# Now, using the erode and dilate function we will apply it to our image and detect and extract the horizontal lines.

horizontal_detect = cv2.erode(grey_scale, horizontal_kernel, iterations=3)
hor_line = cv2.dilate(horizontal_detect, horizontal_kernel, iterations=3)
plotting = plt.imshow(horizontal_detect, cmap='gray')
plt.show()

# In the same way, we will repeat these steps to detect the vertical lines by building another kernel.

vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, length))
vertical_detect = cv2.erode(grey_scale, vertical_kernel, iterations=3)
ver_lines = cv2.dilate(vertical_detect, vertical_kernel, iterations=3)
show = plt.imshow(vertical_detect, cmap='gray')
plt.show()

# Once we have these ready, all we need to do is combine them and make them into a grid-like structure and get 
# a clear tabular representation without the content inside. To do this, we will first combine the vertical 
# and horizontal lines and create a final rectangular kernel. Then we will convert the image back 
# from greyscale to white.

final = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
combine = cv2.addWeighted(ver_lines, 0.5, hor_line, 0.5, 0.0)
combine = cv2.erode(~combine, final, iterations=2)
thresh, combine = cv2.threshold(
    combine, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
convert_xor = cv2.bitwise_xor(read_image, combine)
inverse = cv2.bitwise_not(convert_xor)
output = plt.imshow(inverse, cmap='gray')
plt.show()

# Once we have these ready, all we need to do is combine them and make them into a grid-like structure and get 
# a clear tabular representation without the content inside. To do this, we will first combine the vertical 
# and horizontal lines and create a final rectangular kernel. Then we will convert the image back from greyscale to white.

final = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
combine = cv2.addWeighted(ver_lines, 0.5, hor_line, 0.5, 0.0)
combine = cv2.erode(~combine, final, iterations=2)
thresh, combine = cv2.threshold(
    combine, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
convert_xor = cv2.bitwise_xor(read_image, combine)
inverse = cv2.bitwise_not(convert_xor)
output = plt.imshow(inverse, cmap='gray')
plt.show()

# This shows the final combined table where the cells are present. 

#%% Retrieving the cell positions

# Now that we have our empty table ready, we need to find the right location to add the text. That is the 
# column and the row where the text needs to be inserted. To do this, we need to get bounding boxes 
# around each cell. Contours are the best way to highlight the cell lines and determine the bounding boxes. 
# Let us now write a function to get the contours and the bounding box. It is also important to make 
# sure the contours are read in a particular order which will be written in the function.

cont, _ = cv2.findContours(combine, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


def get_boxes(num, method="left-to-right"):
    invert = False
    flag = 0
    if method == "right-to-left" or method == "bottom-to-top":
        invert = True
    if method == "top-to-bottom" or method == "bottom-to-top":
        flag = 1
    boxes = [cv2.boundingRect(c) for c in num]
    (num, boxes) = zip(*sorted(zip(num, boxes),
                               key=lambda b: b[1][i], invert=invert))
    return (num, boxes)


cont, boxes = get_boxes(cont, method="top-to-bottom")

# Next, we will retrieve the dimensions of each contour and store them. Since these contours and rectangles, 
# we will have 4 sides. We need to set the dimensions which are up to the user. Here I have set the width 
# to be 500 and height to be 500. These values depend on the size of the image.

final_box = []
for c in cont:
    s1, s2, s3, s4 = cv2.boundingRect(c)
    if (s3 < 500 and s4 < 500):
        rectangle_img = cv2.rectangle(
            read_image, (s1, s2), (s1+s3, s2+s4), (0, 255, 0), 2)
        final_box.append([s1, s2, s3, s4])
graph = plt.imshow(rectangle_img, cmap='gray')
plt.show()

# You can see the boxes are highlighted as shown above and we have the position of each cell of the image 
# stored in the list. But we also need the location of each of the cells so that they can be extracted in 
# order. To get the location we will take the average value of the boxes and then add the height and width 
# with them. Then we will sort them into respective boxes and find the midpoint of these boxes so that they 
# can be aligned well.

dim = [boxes[i][3] for i in range(len(boxes))]
avg = np.mean(dim)
hor = []
ver = []
for i in range(len(box)):
    if(i == 0):
        ver.append(box[i])
        last = box[i]
    else:
        if(box[i][1] <= last[1]+avg/2):
            ver.append(box[i])
            last = box[i]
            if(i == len(box)-1):
                hor.append(ver)
        else:
            hor.append(ver)
            ver = []
            last = box[i]
            ver.append(box[i])
total = 0
for i in range(len(hor)):
    total = len(hor[i])
    if total > total:
        total = total
mid = [int(hor[i][j][0]+hor[i][j][2]/2) for j in range(len(hor[i])) if hor[0]]
mid = np.array(mid)
mid.sort()

#%% Value Extraction
# Since we have the boxes, and the dimensions along with the midpoint we can now move on to our text extraction. 
# But before that, we will make sure that our cells are in the correct order. To do this, follow these steps.

order = []
for i in range(len(hor)):
    arrange = []
    for k in range(total):
        arrange.append([])
    for j in range(len(hor[i])):
        sub = abs(mide-(row[i][j][0]+row[i][j][2]/4))
        lowest = min(sub)
        idx = list(sub).index(lowest)
        arrange[idx].append(row[i][j])
    order.append(arrange)

# Now we will use the pytesseract to perform OCR since it is compatible with OpenCV and Python.
try:
    from PIL import Image
except ImportError:
    import Image

# We will take every box and perform eroding and dilating on it and then extract the information in the cells with OCR.
extract = []
for i in range(len(order)):
    for j in range(len(order[i])):
        inside = ''
        if(len(order[i][j]) == 0):
            extract.append(' ')
        else:
            for k in range(len(order[i][j])):
                side1, side2, width, height = order[i][j][k][0], order[
                    i][j][k][1], order[i][j][k][2], order[i][j][k][3]
                final_extract = bitnot[side2:side2+h, side1:side1+width]
                final_kernel = cv2.getStructuringElement(
                    cv2.MORPH_RECT, (2, 1))
                get_border = cv2.copyMakeBorder(
                    final_extract, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=[255, 255])
                resize = cv2.resize(get_border, None, fx=2,
                                    fy=2, interpolation=cv2.INTER_CUBIC)
                dil = cv2.dilate(resize, final_kernel, iterations=1)
                ero = cv2.erode(dil, final_kernel, iterations=2)
                ocr = pytesseract.image_to_string(ero)
                if(len(ocr) == 0):
                    ocr = pytesseract.image_to_string(ero, config='--psm 3')
                inside = inside + " " + ocr
            extract.append(inside)

