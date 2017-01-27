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

window_size = 10

def training_generator(y_full, files_list, batch_size, cnt=0):
  """
  Define train generator.
  :param y_full: full vector of y labels/ observations
  :param files_list: full list of filenames for images
  """
  while 1:
      Y_train = np.asarray(y_full)[(batch_size*cnt):batch_size*(cnt+1)]
      X_train = []
      buffer_lastN = [files_list[(batch_size*cnt)]] * window_size
      for i in files_list[(batch_size*cnt):batch_size*(cnt+1)]:
          for img in buffer_lastN:
              X_train.append(cv2.imread(img))
          buffer_lastN.pop(0)
          buffer_lastN.append(i)
      X_train = np.asarray(X_train)
      X_train = X_train.astype('float32')
      X_train /= 255
      yield (X_train, Y_train)
      cnt += 2


def training_generator_v2(y_full, files_list, batch_size, cnt=0):
  """
  Define train generator.
  :param y_full: full vector of y labels/ observations
  :param files_list: full list of filenames for images
  """
  while 1:
      Y_train = np.asarray(y_full)[(batch_size*cnt):batch_size*(cnt+1)]
      # print(cnt)
      X_train = np.asarray([cv2.imread(i) for i in files_list[(batch_size*cnt):batch_size*(cnt+1)]])
      X_train = X_train.astype('float32')
      X_train /= 255
      yield (X_train, Y_train)
      cnt += 1

# load images filenames
files_dir = "/data/extracted"
files_list = glob.glob(files_dir + '/center/*.jpg')

files_list = files_list[-5212:]
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
y_full = y_full[-5212:]
# define simple model
batch_size = 50
nb_classes = 1
nb_epoch = 2
img_rows, img_cols = 480, 640
img_channels = 3

model = Sequential()

model.add(Convolution2D(24, 5, 5, border_mode='same',
                        input_shape=(img_rows, img_cols, img_channels),
                        subsample=(2,2)))
model.add(Activation('relu'))
# model.add(Convolution2D(36, 5, 5, border_mode='same',
#                        subsample=(2,2)))
# model.add(Activation('relu'))
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

model.fit_generator(training_generator(y_full=y_full, 
                                      files_list=files_list,
                                      batch_size=batch_size),
                    samples_per_epoch=2500, 
                    nb_epoch=nb_epoch,
                    validation_data=training_generator(y_full=y_full, 
                                                      files_list=files_list, 
                                                      batch_size=batch_size, 
                                                      cnt=1),
                    nb_val_samples=2500/nb_epoch)
np.savetxt('y', y_full[:4000], delimiter=',')

yhat = model.predict_generator(training_generator(y_full=y_full,
                                      files_list=files_list,
                                      batch_size=batch_size), val_samples=4000)

np.savetxt('yhat', yhat, delimiter=',')

