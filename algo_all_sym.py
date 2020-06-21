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
model.add(Dense(1024, activation = "relu"))
model.add(BatchNormalization(axis = 1))
model.add(Dropout(0.5))
model.add(Dense(22, activation = "softmax"))

model.load_weights('mini_model.h5')

##################################################################

################### IMAGE ANALYSIS ##############################

img_x = None
img_y = None
image_bin = None

def get_integral(image_path):
    global img_x, img_y, image_bin
    
    original_image = cv2.imread(image_path)
    image = original_image.copy()
    img_y, img_x, _  = image.shape

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

    return get_latex_and_scipy(cnts)

box_coordinates=[]
done = []

def get_latex_and_scipy(contours):
    global box_coordinates, done
    
    idx=0
    for c in contours:
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

    box_coordinates = Sort_Tuple(box_coordinates)

    latex, scipy = func(0,img_x,0,img_y)

    scipy_rectified = rectify_scipy(scipy)

    return (latex, scipy_rectified)


def Sort_Tuple(tup):  
 
    tup.sort(key = lambda x: x[0])  
    return tup  


def func(xi,xf,yi,yf):
    seq=''
    sci=''
    idx = 0

    for bcor in box_coordinates:
        if(done[idx]==1):
            idx+=1
            continue
        if(((bcor[0]>xi)and(bcor[0]<xf))and((bcor[1]>yi) and (bcor[1]<yf))):
            
            if(bcor[4] == 8):
                (den_latex, den_scipy) = func(bcor[0],bcor[0]+bcor[2],bcor[1],yf)
                (num_latex, num_scipy) = func(bcor[0],bcor[0]+bcor[2],yi,bcor[1])
                if((den_latex=='')and(num_latex=='')) and ((den_scipy=='')and(num_scipy=='')):
                    seq = seq + '-'
                    sci = sci + '-'
                    done[idx] = 1
                else:
                    seq = seq + '\\'+'frac'+'{'+'{}'.format(num_latex)+'}'+'{'+'{}'.format(den_latex)+'}'
                    sci = sci + '(' + '{}'.format(num_scipy) + ')/(' + '{}'.format(den_scipy) + ')'  
                    done[idx] = 1

            

            elif(bcor[4] == -1):
                (inside_sqrt_latex, inside_sqrt_scipy) = func(bcor[0], bcor[0] + bcor[2], bcor[1], bcor[1] + bcor[3])
                seq = seq + '\\' + 'sqrt' + '{' + '{}'.format(inside_sqrt_latex) + '}'
                sci = sci + 'sqrt(' + '{}'.format(inside_sqrt_scipy) + ')' 
                done[idx] = 1
            
            else:
                if(bcor[4] == 13):
                    seq = seq  + '+'
                    sci = sci + '+'
                    done[idx] = 1
                else:
                    if(bcor[4] == 20):
                        seq = seq + 'x'
                        sci = sci + 'x'
                    
                    elif(bcor[4] == 2):
                        seq = seq + 'a'
                        sci = sci + 'a'

                    elif(bcor[4] == 3):
                        seq = seq + 'c'
                        sci = sci + 'c'

                    elif(bcor[4] == 4):
                        seq = seq + 'e'
                        sci = sci + 'e'

                    elif(bcor[4] == 5):
                        seq = seq + '5'
                        sci = sci + '5'

                    elif(bcor[4] == 6):
                        seq = seq + '4'
                        sci = sci + '4'

                    elif(bcor[4] == 7):
                        seq = seq + '\infty'
                        sci = sci + 'inf'

                    elif(bcor[4] == 9):
                        seq = seq + 'n'
                        sci = sci + 'n'

                    elif(bcor[4] == 10):
                        seq = seq + '9'
                        sci = sci + '9'

                    elif(bcor[4] == 11):
                        seq = seq + '1'
                        sci = sci + '1'

                    elif(bcor[4] == 12):
                        seq = seq + '\pi'
                        sci = sci + 'pi'

                    elif(bcor[4] == 14):
                        seq = seq + 's'
                        sci = sci + 's'

                    elif(bcor[4] == 15):
                        seq = seq + '7'
                        sci = sci + '7'

                    elif(bcor[4] == 16):
                        seq = seq + '6'
                        sci = sci + '6'

                    elif(bcor[4] == 17):
                        seq = seq + 't'
                        sci = sci + 't'

                    elif(bcor[4] == 18):
                        seq = seq + '3'
                        sci = sci + '3'

                    elif(bcor[4] == 19):
                        seq = seq + '2'
                        sci = sci + '2'
                    
                    elif(bcor[4] == 21):
                        seq = seq + '0'
                        sci = sci + '0'

                    done[idx] = 1
                    for exp_cor in box_coordinates:
                        if(((exp_cor[0]>xi)and(exp_cor[0]<xf))and((exp_cor[1]>yi) and (exp_cor[1]<yf))):
                            if((exp_cor[1]+exp_cor[3]-bcor[1]<0.75*bcor[3])and(exp_cor[0]>bcor[0])and (exp_cor[0]<bcor[0]+bcor[2]+15)):
                                seq = seq  + '^'
                                sci = sci + '**'
                                break


        idx +=1
    
                
    return (seq, sci)


def rectify_scipy(scipy):

    scipy_rectified = ''
    blacklist_num = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    blacklist_var = ['x', 'pi']

    for i in range(len(scipy)):

        if i == (len(scipy) - 1):
            scipy_rectified += scipy[i]
            continue

        if (scipy[i] in blacklist_num and scipy[i + 1] in blacklist_var) or (scipy[i] in blacklist_var and scipy[i + 1] in blacklist_num):
            scipy_rectified = scipy_rectified + scipy[i] + '*'
            continue

        scipy_rectified += scipy[i]

    return scipy_rectified



