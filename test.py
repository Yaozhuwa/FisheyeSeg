# x = [['1','1'],['2','2']]
# def f(x):
#     return list(map(float,x))
#
# print(list(map(f,x)))
# # print(float(x))

import cv2
import numpy as np
import math
import random

def rand_crop(image, annot):
    rate = 0.8
    rows, cols, channels = image.shape

    new_rows = math.floor(rows * rate)
    new_cols = math.floor(cols * rate)

    row_start = math.floor((rows - new_rows) * random.random())
    col_start = math.floor((cols - new_cols) * random.random())

    crop_image = image[row_start:row_start + new_rows, col_start:col_start + new_cols]
    crop_annot = annot[row_start:row_start + new_rows, col_start:col_start + new_cols]

    return crop_image,crop_annot

image = cv2.imread('train.png')
annot = cv2.imread('annot.png',flags=0)
print(image.shape, annot.shape)
image,annot = rand_crop(image, annot)
print(image.shape, annot.shape)


cv2.imshow('image',image)
cv2.imshow('annot',annot*10)
cv2.waitKey(0)
