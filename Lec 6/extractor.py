# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 11:43:26 2017

@author: Suhail
"""
import glob

readpath = 'H:/Workshop/Lecture Plan/Lecture 6/Burger/*.jpg'
labels = 'H:/Workshop/Lecture Plan/Lecture 6/Labels/labels.csv'
objectClass = 'burger'

images = glob.glob(readpath)
labelfile = open(labels,'w')

for image in images:
    labelfile.write(image+','+objectClass+'\n')

labelfile.close()
