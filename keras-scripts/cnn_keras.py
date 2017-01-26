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


files_dir = "/Users/leisure/ai/self-driving-car/datasets/output/dataset/center/"
files_list = glob.glob(files_dir + '*.jpg')
X_train = [cv2.imread(i) for i in files_list[:10000]]
X_test = [cv2.imread(i) for i in files_list[10000:]]

steering = pd.read_csv(
    '/Users/leisure/ai/self-driving-car/datasets/output/dataset/steering.csv')
camera = pd.read_csv(
    '/Users/leisure/ai/self-driving-car/datasets/output/dataset/camera.csv')
ts_camera = camera[camera['frame_id'] == 'center_camera'].timestamp.values
i = 0
y_full = []
for j in range(len(steering)):
    if (steering.timestamp.values[j] > ts_camera[i]):
        y_full.append(steering.angle.values[j])
        if i == len(ts_camera) - 1:
            break
        else:
            i += 1
y = np.asarray(y_full)
Y_train = y[:10000]
Y_test = y[10000:]

# simple model
batch_size = 32
nb_classes = 1
nb_epoch = 10
data_augmentation = True
img_rows, img_cols = 480, 640
img_channels = 3

model = Sequential()

model.add(Convolution2D(24, 5, 5, border_mode='same',
                        input_shape=(img_rows, img_cols, img_channels),
                        subsample=(2, 2)))
model.add(Activation('relu'))
model.add(Convolution2D(36, 5, 5, border_mode='same',
                        subsample=(2, 2)))
model.add(Activation('relu'))
model.add(Convolution2D(48, 5, 5, border_mode='same',
                        subsample=(2, 2)))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

model.fit(X_train, Y_train,
          batch_size=batch_size,
          nb_epoch=nb_epoch,
          validation_data=(X_test, Y_test),
          shuffle=True)
