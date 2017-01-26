from __future__ import print_function
import tensorflow as tf

import cv2
import pandas as pd
import numpy as np
import glob

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adam
from keras.utils import np_utils

def trainig_generator(y, files, batch_size, val_flag=0):
  """
  Define train generator.
  :param y_full: full vector of y labels/ observations
  :param files_list: full list of filenames for images
  """
  while 1:
    for i in range(0, len(y), batch_size):
      Y_train = y[i:i+batch_size]
      X_train = np.asarray([cv2.imread(j) for j in files[i:i+batch_size]])
      X_train = X_train.astype('float32')
      X_train /= 255
      if val_flag: 
        print(str(i/5+1)+"V", 
              X_train[0][0][0][0],
              Y_train[0], 
              len(Y_train))
      else:
        print(str(i/5+1), 
              X_train[0][0][0][0],
              Y_train[0], 
              len(Y_train))
      yield (X_train, Y_train)

c = 120 # select last <c> frames to from dataset
# load images filenames
files_dir = "/Users/leisure/ai/datasets/output/dataset"
files_list = glob.glob(files_dir + '/center/*.jpg')
files_list = files_list[-c:]

# load labels and select based on timestamps
steering=pd.read_csv(files_dir + '/steering.csv')
camera=pd.read_csv(files_dir + '/camera.csv')
ts_camera = camera[camera['frame_id']=='center_camera'].timestamp.values
i = 0
y_full = []
for j in range(len(steering)):
    if (steering.timestamp.values[j]>ts_camera[i]):
        y_full.append(steering.angle.values[j])
        if i==len(ts_camera)-1:
            break
        else:
            i+=1
y_full = y_full[-c:]

# split training and test set
y_full_tr = np.asarray([y_full[i:i+5] for i in range(0,c,10)]).flatten(-1)
y_full_te = np.asarray([y_full[i:i+5] for i in range(5,c,10)]).flatten(-1)
list_tr = [files_list[i:i+5] for i in range(0,c,10)]
files_list_tr = [item for sublist in list_tr for item in sublist]
list_te = [files_list[i:i+5] for i in range(5,c,10)]
files_list_te = [item for sublist in list_te for item in sublist]

# define simple model
batch_size = 5
nb_classes = 1
nb_epoch = 4
img_rows, img_cols = 480, 640
img_channels = 3

model = Sequential()

model.add(Convolution2D(24, 5, 5, border_mode='same',
                        input_shape=(img_rows, img_cols, img_channels),
                        subsample=(2,2)))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dense(25))
model.add(Activation('relu'))
model.add(Dense(nb_classes))
# model.add(Activation('softmax'))

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss='mean_squared_error', 
              optimizer=adam)

model.fit_generator(trainig_generator(y=y_full_tr, 
                                      files=files_list_tr,
                                      batch_size=batch_size),
                    samples_per_epoch=10, 
                    nb_epoch=nb_epoch,
                    validation_data=trainig_generator(y=y_full_te, 
                                                      files=files_list_te, 
                                                      batch_size=batch_size,
                                                      val_flag=1),
                    nb_val_samples=10)

