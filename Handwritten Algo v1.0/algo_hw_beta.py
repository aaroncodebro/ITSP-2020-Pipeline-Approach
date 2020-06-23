import numpy as np 

from keras import layers
from keras.layers import Input, Conv2D, MaxPool2D, Dense, Dropout, Flatten, BatchNormalization, ZeroPadding2D, Activation
from keras.models import Model
from keras.models import Sequential

import cv2
import matplotlib.pyplot as plt


########################## LOAD MODEL ###################################

SIZE = 32

model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu', input_shape = (SIZE,SIZE,1)))
model.add(BatchNormalization(axis = 3))
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu'))
model.add(BatchNormalization(axis = 3))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))


model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(BatchNormalization(axis = 3))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(BatchNormalization(axis = 3))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(BatchNormalization(axis = 1))
model.add(Dropout(0.5))
model.add(Dense(13, activation = "softmax"))

model.load_weights('model_script_params.h5')

##################################################################

################### IMAGE ANALYSIS ##############################


original_image = cv2.imread("test6.png")
image = original_image.copy()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

thresh = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY_INV)[1]
image_bin = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY)[1]

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

box_coordinates=[]
threshold_min_area = 150

idx=0
done = []
for c in cnts:
    if(cv2.contourArea(c)<threshold_min_area):
        continue
    x,y,w,h = cv2.boundingRect(c)
    
    roi = image_bin[y-2:y+2+h,x-2:x+w+2]

    roi = cv2.resize(roi,(SIZE,SIZE))

    roi = np.expand_dims(roi, 2) 

    test = np.zeros((1, SIZE, SIZE, 1), dtype = np.uint8)

    test[0] = roi

    tag = np.argmax(model.predict(test))

    box_coordinates.append([x,y,w,h,tag])
    done.append(0)
    idx +=1


box_idx = 0

for bcor in box_coordinates:
    
    count = 0
    for con_cor in box_coordinates:
        if con_cor == bcor:
            continue
            
        if con_cor[0] > bcor[0] and (con_cor[0] + con_cor[2]) < (bcor[0] + bcor[2]) and con_cor[1] > bcor[1] and (con_cor[1] + con_cor[3]) < (bcor[1] + bcor[3]):
            count += 1

    if count >= 1:
        box_coordinates[box_idx][4] = -1

    box_idx += 1
       
def Sort_Tuple(tup):  
 
    tup.sort(key = lambda x: x[0])  
    return tup  
Sort_Tuple(box_coordinates)
 
img_y, img_x, _  = image.shape


def func(xi,xf,yi,yf):
    seq=''
    idx = 0

    for bcor in box_coordinates:
        if(done[idx]==1):
            idx+=1
            continue
        if(((bcor[0]>xi)and(bcor[0]<xf))and((bcor[1]>yi) and (bcor[1]<yf))):
            
            if(bcor[4] == 10):
                den = func(bcor[0],bcor[0]+bcor[2],bcor[1],yf)
                num = func(bcor[0],bcor[0]+bcor[2],yi,bcor[1])
                if((den=='')and(num=='')):
                    seq = seq + '-'
                    done[idx] = 1
                else:
                    seq = seq + '\\'+'frac'+'{'+'{}'.format(num)+'}'+'{'+'{}'.format(den)+'}'
                    done[idx] = 1

            

            elif(bcor[4] == -1):
                inside_sqrt = func(bcor[0], bcor[0] + bcor[2], bcor[1], bcor[1] + bcor[3])
                seq = seq + '\\' + 'sqrt' + '{' + '{}'.format(inside_sqrt) + '}'
                done[idx] = 1
            
            else:
                if(bcor[4]==11):
                    seq = seq  + '+'
                    done[idx] = 1
                else:
                    if(bcor[4]==12):
                        seq = seq + 'x'
                    else:
                        seq = seq + str(bcor[4])

                    done[idx] = 1
                    for exp_cor in box_coordinates:
                        if(((exp_cor[0]>xi)and(exp_cor[0]<xf))and((exp_cor[1]>yi) and (exp_cor[1]<yf))):
                            if((exp_cor[1]+exp_cor[3]-bcor[1]<0.75*bcor[3])and(exp_cor[0]>bcor[0])and (exp_cor[0]<bcor[0]+bcor[2]+15)):
                                seq = seq  + '^'
                                break


        idx +=1
    
                
    return seq

print(func(0,img_x,0,img_y))


