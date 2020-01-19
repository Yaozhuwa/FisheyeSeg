import torch
from data.CityScape import CityScape
from data.FishEyeGenerator import FishEyeGenerator
from torchvision.transforms import ToTensor, ColorJitter, Normalize
from PIL import Image
import random
import numpy as np
from torch.utils.data import DataLoader
from torch import nn
from config import DefaultConfig
from models.ERFPSPNet import ERFPSPNet
from models.SwiftNet import SwiftNet, resnet18
import torch.nn.functional as F
from torch import nn
import math
import time
from torch.utils.tensorboard import SummaryWriter
import torchvision
import os
import sys
from loss import FocalLoss2d, CrossEntropyLoss2d
import cv2

def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

Config = DefaultConfig()
MyDevice = get_default_device()
MyCPU = torch.device('cpu')


class MyTransform(object):
    def __init__(self, focal_len, shape=None):
        self._transformer = FishEyeGenerator(focal_len, shape)
        self._F_RAND_FLAG = False
        self._F_RANGE = [200, 400]

        self._EXT_RAND_FLAG = False
        self._EXT_PARAM_RANGE = [0, 0, 0, 0, 0, 0]
        self._transformer.set_ext_param_range(self._EXT_PARAM_RANGE)

        self._RAND_CROP = False
        self._rand_crop_rate = 0.8

        self._NORMAL_SCALE = False
        self._scale_range = [0.5, 2]

        self._FISH_SCALE = False
        self._fish_scale_range = [0.5, 2]
        self._NORMAL_TRANSLATE = False
        self._trans_range = [-20,20]

    def set_crop(self,rand=True, rate=0.8):
        self._RAND_CROP = rand
        self._rand_crop_rate = rate

    def set_bkg(self, bkg_label=20, bkg_color=[0, 0, 0]):
        self._transformer.set_bkg(bkg_label, bkg_color)

    def set_ext_param_range(self, ext_param):
        self._EXT_PARAM_RANGE = list(ext_param)
        self._transformer.set_ext_param_range(self._EXT_PARAM_RANGE)

    def rand_ext_params(self):
        self._EXT_RAND_FLAG = True

    def set_ext_params(self, ext_params):
        self._transformer.set_ext_params(ext_params)
        self._EXT_RAND_FLAG = False

    def set_f(self, focal_len):
        self._transformer.set_f(focal_len)
        self._F_RAND_FLAG = False

    def rand_f(self, f_range=[200, 400]):
        self._F_RANGE = f_range
        self._F_RAND_FLAG = True

    def _rand_crop(self, image, annot):
        rows, cols, channels = image.shape

        new_rows = math.floor(rows*self._rand_crop_rate)
        new_cols = math.floor(cols*self._rand_crop_rate)

        row_start = math.floor((rows-new_rows)*random.random())
        col_start = math.floor((cols-new_cols)*random.random())

        crop_image = image[row_start:row_start+new_rows, col_start:col_start+new_cols]
        crop_annot = annot[row_start:row_start+new_rows, col_start:col_start+new_cols]

        return crop_image, crop_annot
    
    def _fish_scale(self, image, annot):
        borderValue = 20
        rate = random.random()*(self._fish_scale_range[1]-self._fish_scale_range[0])+self._fish_scale_range[0]
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


    def __call__(self, image, annot):
        if self._RAND_CROP:
            image, annot = self._rand_crop(image, annot)
            if self._NORMAL_SCALE:
                scale_rate = random.random()*(self._scale_range[1]-self._scale_range[0])+self._scale_range[0]
                image = cv2.resize(image, None, fx=scale_rate, fy=scale_rate)
                annot = cv2.resize(annot, None, fx=scale_rate, fy=scale_rate, interpolation=cv2.INTER_NEAREST)
        if self._F_RAND_FLAG:
            self._transformer.rand_f(self._F_RANGE)
        if self._EXT_RAND_FLAG:
            self._transformer.rand_ext_params()
        dst_image = self._transformer.transFromColor(image)
        dst_annot = self._transformer.transFromGray(annot, reuse=True)
            
        if self._NORMAL_TRANSLATE:
            x_shift = random.random()*(self._trans_range[1]-self._trans_range[0])+self._trans_range[0]
            y_shift = random.random()*(self._trans_range[1]-self._trans_range[0])+self._trans_range[0]
            M=np.array([[1,0, x_shift],[0,1,y_shift]], dtype=np.float32)
            sz = (dst_annot.shape[1], dst_annot.shape[0])
            dst_image = cv2.warpAffine(dst_image, M, sz)
            dst_annot = cv2.warpAffine(dst_annot, M, sz, flags=cv2.INTER_NEAREST, borderValue=20)

        if self._FISH_SCALE:
            dst_image, dst_annot = self._fish_scale(dst_image, dst_annot)


        dst_image = Image.fromarray(dst_image)
        dst_annot = Image.fromarray(dst_annot)
        brightness, contrast, hue, saturation = 0.1, 0.1, 0.1, 0.1
        dst_image = ColorJitter(brightness, contrast, saturation, hue)(dst_image)
        if (random.random() < 0.5):
            dst_image = dst_image.transpose(Image.FLIP_LEFT_RIGHT)
            dst_annot = dst_annot.transpose(Image.FLIP_LEFT_RIGHT)

        dst_annot = np.asarray(dst_annot)
        dst_annot = torch.from_numpy(dst_annot)
        dst_image = ToTensor()(dst_image)
        # image = Normalize(mean=[.485, .456, .406], std=[.229, .224, .225])(image)

        return dst_image, dst_annot

class RandOneTransform(object):
    def __init__(self, focal_len, shape=None):
        self._transformer = FishEyeGenerator(focal_len, shape)
        self.F = 350
        self._F_RANGE = [200, 400]

        self._EXT_PARAMS = [0, 0, 0, 0, 0, 0]
        self._EXT_PARAM_RANGE = [0, 0, 0, 0, 0, 0]
        self._transformer.set_ext_param_range(self._EXT_PARAM_RANGE)

        self._RAND_CROP = False
        self._rand_crop_rate = 0.8

        self._NORMAL_SCALE = False
        self._scale_range = [0.5, 2]

        self._FISH_SCALE = True
        self._fish_scale_range = [0.5, 2]
        self._NORMAL_TRANSLATE = True
        self._trans_range = [-20,20]

    def set_crop(self,rand=True, rate=0.8):
        self._RAND_CROP = rand
        self._rand_crop_rate = rate

    def set_bkg(self, bkg_label=20, bkg_color=[0, 0, 0]):
        self._transformer.set_bkg(bkg_label, bkg_color)

    def set_ext_param_range(self, ext_param):
        self._EXT_PARAM_RANGE = list(ext_param)
        self._transformer.set_ext_param_range(self._EXT_PARAM_RANGE)

    def set_ext_params(self, ext_params):
        self._EXT_PARAMS = ext_params.copy()
        self._transformer.set_ext_params(ext_params)
        # self._EXT_RAND_FLAG = False

    def set_f(self, focal_len):
        self.F = focal_len
        self._transformer.set_f(focal_len)

    def set_f_range(self, f_range=[200, 400]):
        self._F_RANGE = f_range

    def _rand_crop(self, image, annot):
        rows, cols, channels = image.shape

        new_rows = math.floor(rows*self._rand_crop_rate)
        new_cols = math.floor(cols*self._rand_crop_rate)

        row_start = math.floor((rows-new_rows)*random.random())
        col_start = math.floor((cols-new_cols)*random.random())

        crop_image = image[row_start:row_start+new_rows, col_start:col_start+new_cols]
        crop_annot = annot[row_start:row_start+new_rows, col_start:col_start+new_cols]

        return crop_image, crop_annot
    
    def _fish_scale(self, image, annot):
        borderValue = 20
        rate = random.random()*(self._fish_scale_range[1]-self._fish_scale_range[0])+self._fish_scale_range[0]
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


    def __call__(self, image, annot):
        if self._RAND_CROP:
            image, annot = self._rand_crop(image, annot)
            if self._NORMAL_SCALE:
                scale_rate = random.random()*(self._scale_range[1]-self._scale_range[0])+self._scale_range[0]
                image = cv2.resize(image, None, fx=scale_rate, fy=scale_rate)
                annot = cv2.resize(annot, None, fx=scale_rate, fy=scale_rate, interpolation=cv2.INTER_NEAREST)

        # index = random.randint(0,2)

        # if index==0:
        #     self._transformer.rand_f(self._F_RANGE)
        #     self._transformer.set_ext_params(self._EXT_PARAMS)
        # elif index==1:
        #     # 随机旋转
        #     self._transformer.set_f(self.F)
        #     the_ex_range = self._EXT_PARAM_RANGE.copy()
        #     the_ex_range[3:] = [0,0,0]
        #     self._transformer.set_ext_param_range(the_ex_range)
        #     self._transformer.rand_ext_params()
        # elif index==2:
        #     # 随机平移
        #     self._transformer.set_f(self.F)
        #     the_ex_range = self._EXT_PARAM_RANGE.copy()
        #     the_ex_range[:3] = [0,0,0]
        #     self._transformer.set_ext_param_range(the_ex_range)
        #     self._transformer.rand_ext_params()
    
        dst_image = self._transformer.transFromColor(image)
        dst_annot = self._transformer.transFromGray(annot, reuse=True)
            
        if self._NORMAL_TRANSLATE:
            x_shift = random.random()*(self._trans_range[1]-self._trans_range[0])+self._trans_range[0]
            y_shift = random.random()*(self._trans_range[1]-self._trans_range[0])+self._trans_range[0]
            M=np.array([[1,0, x_shift],[0,1,y_shift]], dtype=np.float32)
            sz = (dst_annot.shape[1], dst_annot.shape[0])
            dst_image = cv2.warpAffine(dst_image, M, sz)
            dst_annot = cv2.warpAffine(dst_annot, M, sz, flags=cv2.INTER_NEAREST, borderValue=20)

        if self._FISH_SCALE:
            dst_image, dst_annot = self._fish_scale(dst_image, dst_annot)

        dst_image = Image.fromarray(dst_image)
        dst_annot = Image.fromarray(dst_annot)
        brightness, contrast, hue, saturation = 0.1, 0.1, 0.1, 0.1
        dst_image = ColorJitter(brightness, contrast, saturation, hue)(dst_image)
        if (random.random() < 0.5):
            dst_image = dst_image.transpose(Image.FLIP_LEFT_RIGHT)
            dst_annot = dst_annot.transpose(Image.FLIP_LEFT_RIGHT)

        dst_annot = np.asarray(dst_annot)
        dst_annot = torch.from_numpy(dst_annot)
        dst_image = ToTensor()(dst_image)
        # image = Normalize(mean=[.485, .456, .406], std=[.229, .224, .225])(image)

        return dst_image, dst_annot


def val(model, dataloader, ignore_index=19, is_print=False):
    start_time = time.time()
    model.to(MyDevice)
    model.eval()
    precision_list = np.zeros(Config.class_num, dtype=np.float)
    recall_list = np.zeros(Config.class_num, dtype=np.float)
    iou_list = np.zeros(Config.class_num, dtype=np.float)

    pix_num_or = np.zeros(Config.class_num, dtype=np.float)
    pix_num_and = np.zeros(Config.class_num, dtype=np.float)
    pix_num_TPFP = np.zeros(Config.class_num, dtype=np.float)
    pix_num_TPFN = np.zeros(Config.class_num, dtype=np.float)
    val_iter_num = math.ceil(500 / Config.val_batch_size)
    with torch.no_grad():
        for i, (image, annot) in enumerate(dataloader):
            tic = time.time()
            input = image.to(MyDevice)
            target = annot.to(MyDevice, dtype=torch.long)
            # batch_size*1*H*W
            predict = torch.argmax(model(input), 1, keepdim=True)
            predict_onehot = torch.zeros(predict.size(
                0), Config.class_num, predict.size(2), predict.size(3)).cuda()
            predict_onehot = predict_onehot.scatter(1, predict, 1).float()
            target.unsqueeze_(1)
            target_onehot = torch.zeros(target.size(
                0), Config.class_num, target.size(2), target.size(3)).cuda()
            target_onehot = target_onehot.scatter(1, target, 1).float()

            mask = (1 - target_onehot[:, ignore_index, :, :]). \
                view(target.size(0), 1, target.size(2), target.size(3))
            predict_onehot *= mask
            target_onehot *= mask   

            area_and = predict_onehot * target_onehot
            area_or = predict_onehot + target_onehot - area_and

            pix_num_TPFP += torch.sum(predict_onehot,
                                      dim=(0, 2, 3)).cpu().numpy()
            pix_num_TPFN += torch.sum(target_onehot,
                                      dim=(0, 2, 3)).cpu().numpy()
            pix_num_and += torch.sum(area_and, dim=(0, 2, 3)).cpu().numpy()
            pix_num_or += torch.sum(area_or, dim=(0, 2, 3)).cpu().numpy()
            toc = time.time()

            print(f"validation step {i}/{val_iter_num-1}, {toc-tic} sec/step ...")

    precision_list = pix_num_and / (pix_num_TPFP + 1e-5)
    recall_list = pix_num_and / (pix_num_TPFN + 1e-5)
    iou_list = pix_num_and / (pix_num_or + 1e-5)

    precision_list[ignore_index] = 0
    recall_list[ignore_index] = 0
    iou_list[ignore_index] = 0

    mean_precision = np.sum(precision_list) / (Config.class_num - 1)
    mean_recall = np.sum(recall_list) / (Config.class_num - 1)
    mean_iou = np.sum(iou_list) / (Config.class_num - 1)

    m_precision_19 = np.sum(precision_list[0:-1]) / (Config.class_num - 2)
    m_racall_19 = np.sum(recall_list[0:-1]) / (Config.class_num - 2)
    m_iou_19 = np.sum(iou_list[0:-1]) / (Config.class_num - 2)

    if is_print:
        print("==================RESULT====================")
        print("Mean precision:", mean_precision,
              '; Mean precision 19:', m_precision_19)
        print("Mean recall:", mean_recall, '; Mean recall 19:', m_racall_19)
        print("Mean iou:", mean_iou, '; Mean iou 19:', m_iou_19)
        print("各类精度：")
        print(precision_list)
        print("各类召回率：")
        print(recall_list)
        print("各类IOU：")
        print(iou_list)
        print(time.time()-start_time, "sec/validation")
        print("===================END======================")

    return mean_precision, mean_recall, mean_iou, m_precision_19, m_racall_19, m_iou_19

def val_distortion(model, dataloader, ignore_index=19, is_print=False):
    start_time = time.time()
    model.to(MyDevice)
    model.eval()
    precision_list = np.zeros(Config.class_num, dtype=np.float)
    recall_list = np.zeros(Config.class_num, dtype=np.float)
    iou_list = np.zeros(Config.class_num, dtype=np.float)

    pix_num_or = np.zeros(Config.class_num, dtype=np.float)
    pix_num_and = np.zeros(Config.class_num, dtype=np.float)
    pix_num_TPFP = np.zeros(Config.class_num, dtype=np.float)
    pix_num_TPFN = np.zeros(Config.class_num, dtype=np.float)
    val_iter_num = math.ceil(500 / Config.val_batch_size)

    mask_distortion = torch.ones(Config.fish_size, device=MyDevice)
    for i in range(Config.fish_size[0]):
        for j in range(Config.fish_size[1]):
            if (i-Config.fish_size[0]/2)**2+(j-Config.fish_size[1]/2)**2 < Config.mask_radius**2:
                mask_distortion[i,j]=0
    with torch.no_grad():
        for i, (image, annot) in enumerate(dataloader):
            tic = time.time()
            input = image.to(MyDevice)
            target = annot.to(MyDevice, dtype=torch.long)
            # batch_size*1*H*W
            predict = torch.argmax(model(input), 1, keepdim=True)
            predict_onehot = torch.zeros(predict.size(
                0), Config.class_num, predict.size(2), predict.size(3)).cuda()
            predict_onehot = predict_onehot.scatter(1, predict, 1).float()
            target.unsqueeze_(1)
            target_onehot = torch.zeros(target.size(
                0), Config.class_num, target.size(2), target.size(3)).cuda()
            target_onehot = target_onehot.scatter(1, target, 1).float()

            mask = (1 - target_onehot[:, ignore_index, :, :]). \
                view(target.size(0), 1, target.size(2), target.size(3))
            predict_onehot *= mask
            predict_onehot *= mask_distortion
            target_onehot *= mask
            target_onehot *= mask_distortion

            area_and = predict_onehot * target_onehot
            area_or = predict_onehot + target_onehot - area_and

            pix_num_TPFP += torch.sum(predict_onehot,
                                      dim=(0, 2, 3)).cpu().numpy()
            pix_num_TPFN += torch.sum(target_onehot,
                                      dim=(0, 2, 3)).cpu().numpy()
            pix_num_and += torch.sum(area_and, dim=(0, 2, 3)).cpu().numpy()
            pix_num_or += torch.sum(area_or, dim=(0, 2, 3)).cpu().numpy()
            toc = time.time()

            print(f"validation step {i}/{val_iter_num-1}, {toc-tic} sec/step ...")

    precision_list = pix_num_and / (pix_num_TPFP + 1e-5)
    recall_list = pix_num_and / (pix_num_TPFN + 1e-5)
    iou_list = pix_num_and / (pix_num_or + 1e-5)

    precision_list[ignore_index] = 0
    recall_list[ignore_index] = 0
    iou_list[ignore_index] = 0

    mean_precision = np.sum(precision_list) / (Config.class_num - 1)
    mean_recall = np.sum(recall_list) / (Config.class_num - 1)
    mean_iou = np.sum(iou_list) / (Config.class_num - 1)

    m_precision_19 = np.sum(precision_list[0:-1]) / (Config.class_num - 2)
    m_racall_19 = np.sum(recall_list[0:-1]) / (Config.class_num - 2)
    m_iou_19 = np.sum(iou_list[0:-1]) / (Config.class_num - 2)

    if is_print:
        print("==================RESULT====================")
        print("Mean precision:", mean_precision,
              '; Mean precision 19:', m_precision_19)
        print("Mean recall:", mean_recall, '; Mean recall 19:', m_racall_19)
        print("Mean iou:", mean_iou, '; Mean iou 19:', m_iou_19)
        print("各类精度：")
        print(precision_list)
        print("各类召回率：")
        print(recall_list)
        print("各类IOU：")
        print(iou_list)
        print(time.time()-start_time, "sec/validation")
        print("===================END======================")

    return mean_precision, mean_recall, mean_iou, m_precision_19, m_racall_19, m_iou_19

def final_eval():
    valid_path = ['\\val_200f','\\val_250f','\\val_300f','\\val_350f','\\val_400f']
    # valid_path = ['\\val_200f','\\val_250f','\\val_300f','\\val_350f','\\val_400f']
    # valid_path = ['\\val_rotate10']

    valid_annot = [x+'_annot' for x in valid_path]
    # model = ERFPSPNet(shapeHW=[640, 640], num_classes=21)
    resnet = resnet18(pretrained=True)
    model = SwiftNet(resnet, num_classes=21)
    model.to(MyDevice)
    checkpoint = torch.load(Config.ckpt_path)
    print("Load",Config.ckpt_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    for i in range(len(valid_path)):
        validation_set = CityScape(Config.data_dir+valid_path[i], Config.data_dir+valid_annot[i])
        validation_loader = DataLoader(
            validation_set, batch_size=Config.val_batch_size, shuffle=False)
        print('\n',valid_path[i])
        val_distortion(model, validation_loader, is_print=True)

def one_eval(model, ckpt_path, valid_path, valid_annot_path, is_distortion=False):
    model.to(MyDevice)
    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    validation_set = CityScape(Config.data_dir+valid_path, Config.data_dir+valid_annot_path)
    validation_loader = DataLoader(
        validation_set, batch_size=Config.val_batch_size, shuffle=False)
    print("ckpt_path", ckpt_path)
    print("valid_path",valid_path)
    if is_distortion:
        val_distortion(model, validation_loader, is_print=True)
    else:
        val(model, validation_loader, is_print=True)

def all_eval():
    resnet = resnet18(pretrained=True)
    model = SwiftNet(resnet, num_classes=21)
    ckpt_index = [x for x in range(22, 26)]
    # ckpt_index = [1, 4, 5, 6, 8, 9, 10, 11]
    ckpt = ["checkpoints/CKPT/"+str(x)+'.pth' for x in ckpt_index]
    # valid_path = ['\\val_400f']
    # valid_path = ['\\val_200f','\\val_250f','\\val_300f','\\val_350f','\\val_400f','\\val_7DOF', '\\val_rotate10']
    valid_path = ['\\val_200f','\\val_250f','\\val_300f','\\val_350f','\\val_400f']
    for i in ckpt:
        for j in valid_path:
            one_eval(model, i, j, j+"_annot", is_distortion=True)

    


def train():
    train_transform = MyTransform(Config.f, Config.fish_size)
    train_transform.set_ext_params(Config.ext_param)
    train_transform.set_ext_param_range(Config.ext_range)
    if Config.rand_f:
        train_transform.rand_f(f_range=Config.f_range)
    if Config.rand_ext:
        train_transform.rand_ext_params()
    train_transform.set_bkg(bkg_label=20, bkg_color=[0, 0, 0])
    train_transform.set_crop(rand=Config.crop, rate=Config.crop_rate)

    # train_transform = RandOneTransform(Config.f, Config.fish_size)
    # train_transform.set_ext_params(Config.ext_param)
    # train_transform.set_ext_param_range(Config.ext_range)
    # train_transform.set_f_range(Config.f_range)
    # train_transform.set_bkg(bkg_label=20, bkg_color=[0, 0, 0])
    # train_transform.set_crop(rand=Config.crop, rate=Config.crop_rate)

    train_set = CityScape(Config.train_img_dir,
                          Config.train_annot_dir, transform=train_transform)
    train_loader = DataLoader(train_set, batch_size=Config.batch_size, shuffle=True,
                              num_workers=Config.dataloader_num_worker)

    validation_set = CityScape(Config.valid_img_dir, Config.valid_annot_dir)
    validation_loader = DataLoader(
        validation_set, batch_size=Config.val_batch_size, shuffle=False)

    # model = ERFPSPNet(shapeHW=[640, 640], num_classes=21)
    resnet = resnet18(pretrained=True)
    model = SwiftNet(resnet, num_classes=21)
    model.to(MyDevice)

    class_weights = torch.tensor([8.6979065, 8.497886 , 8.741297 , 5.983605 , 8.662319 , 8.681756 ,
       8.683093 , 8.763641 , 8.576978 , 2.7114885, 6.237076 , 3.582358 ,
       8.439253 , 8.316548 , 8.129169 , 4.312109 , 8.170293 , 6.91469  ,
       8.135018 , 0.  ,3.6]).cuda()

    # criterion = CrossEntropyLoss2d(weight=class_weights)
    criterion = FocalLoss2d(weight=class_weights)

    lr = Config.learning_rate

    # Pretrained SwiftNet optimizer
    optimizer = torch.optim.Adam([{'params':model.random_init_params()},
                                  {'params':model.fine_tune_params(), 'lr':1e-4, 'weight_decay':2.5e-5}],
                                 lr=4e-4,
                                 weight_decay=1e-4)

    # ERFNetPSP optimizer
    # optimizer = torch.optim.Adam(model.parameters(),
    #                              lr=1e-3,
    #                              betas=(0.9, 0.999),
    #                              eps=1e-08,
    #                              weight_decay=2e-4)

    # scheduler = torch.optim.lr_scheduler.StepLR(
    #     optimizer, step_size=90, gamma=0.1)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 200, 1e-6)

    start_epoch = 0
    step_per_epoch = math.ceil(2975 / Config.batch_size)
    writer = SummaryWriter(Config.logdir)
    # writer.add_graph(model)

    if Config.train_with_ckpt:
        checkpoint = torch.load(Config.ckpt_path)
        print("Load",Config.ckpt_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']+1
        val(model, validation_loader, is_print=True)
        # loss = checkpoint['loss']
        model.train()

    start_time = None
    for epoch in range(start_epoch, Config.max_epoch):

        for i, (image, annot) in enumerate(train_loader):
            if start_time is None:
                start_time = time.time()
            input = image.to(MyDevice)
            target = annot.to(MyDevice, dtype=torch.long)
            model.train()
            optimizer.zero_grad()
            score = model(input)
            # predict = torch.argmax(score, 1)
            loss = criterion(score, target)
            loss.backward()
            optimizer.step()
            
            global_step = step_per_epoch * epoch + i

            if i%20 ==0:
                predict = torch.argmax(score, 1).to(MyCPU, dtype=torch.uint8)
                writer.add_image("Images/original_image", image[0], global_step=global_step)
                writer.add_image("Images/segmentation_output", predict[0].view(1, 640, 640)*10, global_step=global_step)
                writer.add_image("Images/segmentation_ground_truth", annot[0].view(1, 640, 640)*10, global_step=global_step)

            if i % 20 == 0 and global_step > 0:
                writer.add_scalar("Monitor/Loss", loss.item(), global_step=global_step)

            time_elapsed = time.time() - start_time
            start_time = time.time()
            print(f"{epoch}/{Config.max_epoch-1} epoch, {i}/{step_per_epoch} step, loss:{loss.item()}, "
                  f"{time_elapsed} sec/step; global step={global_step}")

        scheduler.step()
        if epoch > 20:
            mean_precision, mean_recall, mean_iou, m_precision_19, m_racall_19, m_iou_19 = val(
                model, validation_loader, is_print=True)
            
            writer.add_scalar("Monitor/precision20", mean_precision, global_step=epoch)
            writer.add_scalar("Monitor/recall20", mean_recall, global_step=epoch)
            writer.add_scalar("Monitor/mIOU20",  mean_iou, global_step=epoch)
            writer.add_scalar("Monitor1/precision19", m_precision_19, global_step=epoch)
            writer.add_scalar("Monitor1/recall19", m_racall_19, global_step=epoch)
            writer.add_scalar("Monitor1/mIOU19", m_iou_19, global_step=epoch)

            print(epoch, '/', Config.max_epoch, ' loss:', loss.item())
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'loss': loss.item(),
                'optimizer_state_dict': optimizer.state_dict()
            }, Config.ckpt_name + "_" + str(epoch) + ".pth")
            print("model saved!")

    val(model, validation_loader, is_print=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, Config.model_path)
    print("Save model to disk!")
    writer.close()


from test import label2color
def run_image(image_path, model):
    from torchvision.transforms import ToTensor, Normalize
    image = cv2.imread(image_path)
    image1 = cv2.imread(image_path)
    
    image = image[:,:,(2,1,0)]
    image = ToTensor()(image)
    
    image = image.to(torch.device('cuda'))
    image.unsqueeze_(0)
    score = model(image)
    predict = torch.argmax(score, 1).to(MyCPU, dtype=torch.uint8)[0].numpy()
    color_label = label2color(predict)
    # cv2.imwrite('image'+image_path, image1)
    # cv2.imwrite('annot'+image_path, color_label)
    return image1, color_label


def real_image_test():
    imgs = sorted([os.path.join("D:\\DataSets\\MyFishData\\0119_image\\", img) for img in os.listdir("D:\\DataSets\\MyFishData\\0119_image\\")])
    resnet = resnet18(pretrained=True).to(torch.device('cuda'))
    model = SwiftNet(resnet, num_classes=21)
    model = model.to(torch.device('cuda'))
    checkpoint = torch.load("checkpoints/CKPT/16.pth")
    # print("Load",Config.ckpt_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model = model.to(torch.device('cuda'))
    for img in imgs:
        image, label = run_image(img, model)
        cv2.imshow("image",image)
        cv2.imshow("label",label)
        esc_flag = False
        while True:
            key = cv2.waitKey(20) & 0xFF
            if key == 27:
                esc_flag = True
                return
            elif key == ord('p'):
                break

if __name__ == '__main__':
    # final_eval()
    # all_eval()
    # train()
    real_image_test()


    
    
    