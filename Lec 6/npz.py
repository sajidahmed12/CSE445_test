# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 12:20:49 2017

@author: Suhail
"""
import cv2
import pandas
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

labels = 'H:/Workshop/Lecture Plan/Lecture 6/Labels/labels.csv'
npzfile = 'H:/Workshop/Lecture Plan/Lecture 6/Labels/labels.npz'

df = pandas.read_csv(labels)

rows = df.iterrows()

X_temp = []
Y_temp = []

for row in rows:
    image = row[1][0]
    img = cv2.imread(image)
    img = cv2.resize(img,(32,32))
    imageClass = row[1][1]
    X_temp.append(img)
    Y_temp.append(imageClass)


encoder = LabelEncoder()
encoder.fit(Y_temp)
encoded_Y = encoder.transform(Y_temp)
Y = np_utils.to_categorical(encoded_Y)

np.savez(npzfile, X_train=X_temp,Y_train=Y)