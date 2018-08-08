# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 12:54:36 2017

@author: Suhail
"""
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten
from keras.layers.convolutional import Convolution2D,MaxPooling2D
from keras import backend as K
K.set_image_data_format('channels_last')

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
num_classes = 2

img = cv2.imread('testObject1.jpg')
imgRes = cv2.resize(img,(32,32))

X_temp = []
X_temp.append(imgRes)
X = np.asarray(X_temp)
X = X/255

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

model.load_weights('best.hdf5')

y = model.predict_classes(X)
classno = np.ndarray.tolist(y)

dict = {0: 'Burger', 1: 'Pizza'}
objectClass = dict[classno[0]]
print(objectClass)

font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img, objectClass,(50,50), font, 2, (200,255,0), 5, cv2.LINE_AA)
cv2.imshow('Prediction',img)
cv2.waitKey(0)
cv2.destroyAllWindows()