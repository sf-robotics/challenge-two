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

# load images from jpg
files_dir = "/Users/leisure/ai/datasets/output/dataset/center/"
files_list = glob.glob(files_dir+'*.jpg')
X_train = np.asarray([cv2.imread(i) for i in files_list[-1000:]])
X_test = np.asarray([cv2.imread(i) for i in files_list[-1500:-1000]])

# load labels and select based on timestamps
steering=pd.read_csv('/Users/leisure/ai/datasets/output/dataset/steering.csv')
camera=pd.read_csv('/Users/leisure/ai/datasets/output/dataset/camera.csv')
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

Y_train=np.asarray(y_full)[-1000:]
Y_test=np.asarray(y_full)[-1500:-1000]

# define simple model
batch_size = 32
nb_classes = 1
nb_epoch = 1
img_rows, img_cols = 480, 640
img_channels = 3

model = Sequential()

model.add(Convolution2D(24, 5, 5, border_mode='same',
                        input_shape=(img_rows, img_cols, img_channels),
                        subsample=(2,2)))
model.add(Activation('relu'))
# model.add(Convolution2D(64, 3, 3, border_mode='same'))
# model.add(Activation('relu'))
model.add(Flatten())
# model.add(Dense(500))
# model.add(Activation('relu'))
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dense(25))
model.add(Activation('relu'))
# model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', 
              optimizer=adam)

# images normalization
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

model.fit(X_train, Y_train,
          batch_size=batch_size,
          nb_epoch=nb_epoch,
          validation_data=(X_test, Y_test),
          shuffle=True)

# def generate_arrays_from_file(path):
#     while 1:
#         f = open(path)
#         for line in f:
#             x, y = process_line(line)
#             img = load_images(x)
#             yield (img, y)
#         f.close()

# model.fit_generator(generate_arrays_from_file('/my_file.txt'),
#         samples_per_epoch=10000, nb_epoch=1)

