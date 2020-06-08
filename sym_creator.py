import numpy as np
import cv2
from matplotlib import pyplot as plt

original_image = cv2.imread("4.png") ###### put the image from which you want to extract symbols
image = original_image.copy()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#blurred = cv2.GaussianBlur(gray, (3,3), 0)
thresh = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY_INV)[1]
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
#dilate = cv2.dilate(thresh, kernel , iterations=1)

cv2.imshow("thresh", thresh)
#cv2.imshow("dilate", dilate)
im_floodfill = thresh.copy()

# Mask used to flood filling.
# Notice the size needs to be 2 pixels than the image.
h, w = thresh.shape[:2]
mask = np.zeros((h+2, w+2), np.uint8)

# Floodfill from point (0, 0)
cv2.floodFill(im_floodfill, mask, (0,0), (255,255,255))

# Invert floodfilled image
im_floodfill_inv = cv2.bitwise_not(im_floodfill)

# Combine the two images to get the foreground.
im_out = thresh | im_floodfill_inv

cnts = cv2.findContours(im_out.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]

contours = []

#threshold_min_area = 0
#threshold_max_area = 100000

area = []
box_coordinates=[]

idx=0
for c in cnts:
    x,y,w,h = cv2.boundingRect(c)
    #new_img = image | cv2.bitwise_not(image)
    #new_img[y:y+h,x:x+w] = image[y:y+h,x:x+w]
    #cv2.imwrite(str(idx) + '.jpg', new_img)
    roi = image[y-2:y+2+h,x-2:x+w+2]
    cv2.imwrite(str(idx)+ '.jpg', roi)

    #print('{0},{1},{2},{3}'.format(x,y,w,h))
    area.append(cv2.contourArea(c))
    box_coordinates.append((x,y,w,h))
    idx +=1
    

#mean_area = sum(area)/len(area)
mean_area = 0

idx=0
for bcor in box_coordinates:
	cv2.rectangle(original_image, (bcor[0],bcor[1]), (bcor[0]+bcor[2], bcor[1]+bcor[3]), (0,255,0),1)
	
	#idx+=1


#cv2.imshow("Thresholded Image", thresh)
#cv2.imshow("Floodfilled Image", im_floodfill)
#cv2.imshow("Inverted Floodfilled Image", im_floodfill_inv)
#cv2.imshow("Foreground", im_out)    



        

#plt.imshow(original_image,'gray') 
#plt.show()
#cv2.waitKey(0)