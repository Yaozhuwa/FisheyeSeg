import cv2
import numpy as np 
from data.FishEyeGenerator import FishEyeGenerator
import time
import random

# image = cv2.imread("DataSets\\CityScape\\val_350f\\350frankfurt_000000_000294_leftImg8bit.png")
# shape = image.shape
# print(shape)
# for i in range(shape[0]):
#     for j in range(shape[1]):
#         if (i-shape[0]/2)**2+(j-shape[1]/2)**2<100**2:
#             image[i,j] = [0,0,0]
# cv2.imshow('test',image)

# cv2.waitKey(0)
COLOR_MAP = {0:[250, 168, 40], 1:[220, 220, 0], 2:[255,0,0], 3:[0,0,144], 4:[0,0,72], 
        5:[0, 60,100], 6:[0,80,100], 7:[0,0,225], 8:[119, 12, 32], 9:[127, 65, 128],
        10:[249, 34, 233], 11:[70, 70, 70], 12:[100,100,160], 13:[189, 153, 153], 14:[154, 154, 154],
        15:[106, 142, 34], 16:[154, 251, 154], 17:[70,130,180], 18:[222, 19, 62], 19:[0,0,0],
        20:[255,255,255]}

def label2color(label_image):
    shape = label_image.shape
    result = np.zeros((shape[0],shape[1],3), dtype=np.uint8)
    for i in range(shape[0]):
        for j in range(shape[1]):
            result[i,j] = COLOR_MAP[label_image[i,j]]
    result = result[:,:,(2,1,0)]
    return result



def test_color():
    image = cv2.imread("images/1train.png")
    label = cv2.imread('images/1annot.png', 0)
    result = label2color(label)
    image = cv2.resize(image, None, fx=0.5,fy=0.5,interpolation=cv2.INTER_CUBIC)
    result = cv2.resize(result, None, fx=0.5,fy=0.5,interpolation=cv2.INTER_NEAREST)
    cv2.imshow("origin", image)
    cv2.imshow("label", result)
    cv2.waitKey(0)

def test_trans():
    trans = FishEyeGenerator(400, [640,640])
    img = cv2.imread("images/1train.png")
    im_annot = cv2.imread("images/1annot.png", 0)
    # trans.set_f(300)
    trans.set_ext_params([0, 0, 0, 0, 0, 0])
    # trans.rand_ext_params()
    # s = time.time()
    image = trans.transFromColor(img)
    label_dst = trans.transFromGray(im_annot,reuse=True)
    label = label2color(label_dst)

    # print("Time cost:", e-s)
    # image = cv2.resize(dst, None, fx=0.5,fy=0.5,interpolation=cv2.INTER_CUBIC)
    # label = cv2.resize(label, None, fx=0.5,fy=0.5,interpolation=cv2.INTER_NEAREST)
    
    cv2.imshow("src0", image)
    cv2.imshow("dst0",label)

    cv2.waitKey(0)

def fish_scale(image, annot):
    borderValue = 20
    rate = random.random()*1.5+0.5
    print(rate)
    if rate == 1:
        return image, annot
    rows, cols = annot.shape
    image = cv2.resize(image, None,fx=rate,fy=rate)
    annot = cv2.resize(annot, None,fx=rate,fy=rate, interpolation=cv2.INTER_NEAREST)
    if rate <1:
        dst_image = np.ones((rows,cols,3),dtype=np.uint8)*0
        dst_annot = np.ones((rows,cols),dtype=np.uint8)*borderValue
        row_start = rows//2-annot.shape[0]//2
        col_start = cols//2-annot.shape[1]//2
        dst_image[row_start:row_start+annot.shape[0], col_start:col_start+annot.shape[1]] = image
        dst_annot[row_start:row_start+annot.shape[0], col_start:col_start+annot.shape[1]] = annot
        return dst_image, dst_annot
    if rate>1:
        row_start = image.shape[0]//2-rows//2
        col_start = image.shape[1]//2-cols//2
        crop_image = image[row_start:row_start+rows, col_start:col_start+cols]
        crop_annot = annot[row_start:row_start+rows, col_start:col_start+cols]
        return crop_image, crop_annot

def test_scale():
    img = cv2.imread('DataSets\\CityScape\\val_300f\\300frankfurt_000000_000294_leftImg8bit.png')
    annot = cv2.imread('DataSets\\CityScape\\val_300f_annot\\300frankfurt_000000_000294_leftImg8bit.png', 0)
    image, annot = fish_scale(img,annot)
    cv2.imshow('image',image)
    cv2.imshow('annot', annot*10)
    cv2.waitKey(0)

def test_rand_shift():
    _trans_range = [-200,200]
    image = cv2.imread('images/1train.png')
    annot = cv2.imread('images/1annot.png', 0)*10

    dst_image = cv2.resize(image, None, fx=0.5,fy=0.5)
    dst_annot = cv2.resize(annot, None, fx=0.5, fy = 0.5, interpolation=cv2.INTER_NEAREST)

    s = (dst_annot.shape[1], dst_annot.shape[0])
    x_shift = random.random()*(_trans_range[1]-_trans_range[0])+_trans_range[0]
    y_shift = random.random()*(_trans_range[1]-_trans_range[0])+_trans_range[0]
    M=np.array([[1,0, x_shift],[0,1,y_shift]], dtype=np.float32)
    dst_image = cv2.warpAffine(dst_image, M, s)
    dst_annot = cv2.warpAffine(dst_annot, M, s, flags=cv2.INTER_NEAREST, borderValue=200)
    cv2.imshow('train',dst_image)
    cv2.imshow('annot',dst_annot)
    cv2.waitKey(0)

if __name__ == "__main__":
    # test_trans()
    # test_color()
    test_scale()