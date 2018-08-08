# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 12:14:09 2017

@author: Suhail
"""

import random
labels = 'H:/Workshop/Lecture Plan/Lecture 6/Labels/labels.csv'
shuffled_labels = 'H:/Workshop/Lecture Plan/Lecture 6/Labels/shuffled_labels.csv'

labelfile = open(labels, "r")
lines = labelfile.readlines()
labelfile.close()
random.shuffle(lines)

shufflefile = open(shuffled_labels, "w")
shufflefile.writelines(li)
shufflefile.close()