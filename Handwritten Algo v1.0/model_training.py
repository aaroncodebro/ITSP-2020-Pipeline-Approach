import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt

random_seed = 1

sym_to_int_dict = {
    '0': 0,
    '1': 1,
    '2': 2,
    '3': 3,
    '4': 4,
    '5': 5,
    '6': 6,
    '7': 7,
    '8': 8,
    '9': 9,
    '-': 10,
    '+': 11,
    'x': 12
}

key_list = list(sym_to_int_dict.keys())
print(key_list)

TRAIN_PATH = os.path.join(os.path.dirname('__file__'), 'data/')
train_files = next(os.walk(TRAIN_PATH))[1]

print(len(train_files))

total_img = 0
for id_ in train_files:
  file_path = TRAIN_PATH + id_ + '/'
  img_list = next(os.walk(file_path))[2]
  lim = 0
  for img_name in img_list:
    total_img += 1

print(total_img)

import cv2
from PIL import Image
from tqdm import tqdm

X_train = np.zeros((total_img, 32, 32, 1), dtype = np.uint8)
Y_train = np.zeros((total_img, 1), dtype = np.int32)

print('Getting training data...')
sys.stdout.flush()

count = 0
for n, id_ in tqdm(enumerate(train_files), total = len(train_files)):
  label = -1
  for i in key_list:
    if id_ == i:
      label = id_
      break
  
  file_path = TRAIN_PATH + id_ + '/'
  img_list = next(os.walk(file_path))[2]
  lim = 0
  for img_name in img_list:
    img_path = file_path + img_name
    img = cv2.imread(img_path)
   
    img = cv2.resize(img,(32,32))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY)[1]
    

    thresh = np.expand_dims(thresh, 2)
    X_train[count] = thresh
    Y_train[count] = sym_to_int_dict[str(label)]
    lim +=1
    count += 1
   
print('Done!')


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

from keras import layers
from keras.layers import Input, Conv2D, MaxPool2D, Dense, Dropout, Flatten, BatchNormalization, ZeroPadding2D, Activation
from keras.models import Model
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.python.client import device_lib

X_train = X_train.reshape((X_train.shape[0], 32, 32, 1))

Y_train = to_categorical(Y_train, num_classes = 13)

random_seed = 1

X_train, X_cv, Y_train, Y_cv = train_test_split(X_train, Y_train, test_size = 0.3, random_state = random_seed)


model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu', input_shape = (32,32,1)))
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


optimizer = Adam(learning_rate = 0.001)

model.compile(optimizer = optimizer, loss = "categorical_crossentropy", metrics = ["accuracy"])

lr_scheduler = ReduceLROnPlateau(monitor = 'loss', patience = 1, verbose = 1, factor = 0.1)

early_stopper = EarlyStopping(monitor = 'loss', patience = 12, verbose = 1)

model.fit(X_train, Y_train, epochs = 20, batch_size = 16, callbacks = [lr_scheduler, early_stopper])

model.evaluate(X_cv, Y_cv, batch_size = 256)

model.save("model_script_params.h5")