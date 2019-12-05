import cv2
import numpy as np
import torch

def crop(image):
    np.zeros()
    return image

if __name__ == "__main__":
    img = cv2.imread('F:\\Images\\20191121\\IMG_20191121_153206.jpg')
    shape = img.shape
    img = cv2.resize(img,None,fx=0.2, fy=0.2, interpolation=cv2.INTER_CUBIC)
    img = img[205:-205,9:-10,:]
    cv2.imshow('o-image', img)
    print(img.shape)
    cv2.waitKey(0)