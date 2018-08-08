# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 12:54:36 2017

@author: Suhail
"""

import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten
from keras.layers.convolutional import Convolution2D,MaxPooling2D
from keras.callbacks import ModelCheckpoint
from keras import backend as K
K.set_image_data_format('channels_last')

seed = 7
np.random.seed(seed)
num_classes = 2


npzfile = 'H:/Workshop/Lecture Plan/Lecture 6/Labels/labels.npz'

dataset =  np.load(npzfile)
x_train = dataset['X_train']
y_train = dataset['Y_train']

x = x_train/255

def CNN_model():
    # create model
    model = Sequential()
    model.add(Convolution2D(32, 5, 5, border_mode= 'valid' , input_shape=(32,32,3),activation= 'relu' ))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation= 'relu' ))
    model.add(Dense(num_classes, activation= 'softmax' ))
    # Compile model
    model.compile(loss= 'categorical_crossentropy' , optimizer= 'adam' , metrics=[ 'accuracy' ])
    return model

model = CNN_model()

model.summary()

check  = ModelCheckpoint('best.hdf5', monitor = 'val_categorical_accuracy' )
checkpoints = [check]

model.fit(x, y_train, validation_split = 0.2, nb_epoch=20, batch_size=64,verbose=2, callbacks = checkpoints)
