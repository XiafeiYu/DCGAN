# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 19:04:54 2018

@author: Sofie
"""

import cv2
import sys
import os.path
from glob import glob
import math
import numpy as np


def detect_face(file_path, output_height, output_width):
#    if not os.path.isfile(cascade_file):
#        raise RuntimeError("%s: not found" % cascade_file)
    data = []
    counter = 0
    if os.path.exists(file_path) is False:
        raise Exception("[!] No training dataset")
    if os.path.exists('face') is False:
        os.makedirs('face')
    file_list = glob(os.path.join(file_path, '*.jpg'))
    for filename in file_list:
        
        cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
        image = cv2.imread(filename)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
    
        faces = cascade.detectMultiScale(gray,
                                         # detector options
                                         scaleFactor=1.1,
                                         minNeighbors=5,
                                         minSize=(48, 48))
        for i, (x, y, w, h) in enumerate(faces):
            face = image[y: y + h, x:x + w, :]
            face = cv2.resize(face, (output_width, output_height))
            save_filename = '%s.jpg' % (os.path.basename(filename).split('.')[0])
            data.append(face)
            cv2.imwrite("face/" + save_filename, face)
        counter += 1
        if counter % 500 == 0:
            print('please be patient, the convert of image have finished %.2f%%'
                  %(counter/len(file_list)*100))
    return data

def show_mergeimage(x, output_height, output_width, epoch, counter, is_train):       
    line = math.ceil(math.sqrt(len(x)))
    row = math.floor(math.sqrt(len(x)))
    new_image = np.empty([row*output_height, line*output_width, 3])
    for i in range(row):
        for j in range(line):
            new_image[i*output_height:(i+1)*output_height, j*output_width:(j+1)*output_width] = x[i*line + j]
    new_image = (new_image + 1)*127.5
    if is_train:
        if os.path.exists('./train_samples') is False:
            os.makedirs('train_samples')
        cv2.imwrite('./train_samples/train%d_%d.png' %(epoch, counter//200), new_image)
    else:
        if os.path.exists('./test_samples') is False:
            os.makedirs('test_samples')
        cv2.imwrite('./test_samples/test.png', new_image)
            
