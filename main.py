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
import torch.nn.functional as F
from torch import nn
import math
import time
from torch.utils.tensorboard import SummaryWriter
import torchvision
import os
import sys
from loss import FocalLoss2d, CrossEntropyLoss2d


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
        self._EXT_PARAM_RANGE = [5, 5, 10, 0.3, 0.3, 0.4]
        self._transformer.set_ext_param_range(self._EXT_PARAM_RANGE)

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

    def __call__(self, image, annot):
        if self._F_RAND_FLAG:
            self._transformer.rand_f(self._F_RANGE)
        if self._EXT_RAND_FLAG:
            self._transformer.rand_ext_params()
        dst_image = self._transformer.transFromColor(image)
        dst_annot = self._transformer.transFromGray(annot, reuse=True)

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


def train():
    train_transform = MyTransform(350, [640, 640])
    train_transform.rand_f([200, 400])
    train_transform.set_ext_param_range([0, 0, 0, 0.3, 0.3, 0.4])
    train_transform.rand_ext_params()
    train_transform.set_bkg(bkg_label=20, bkg_color=[0, 0, 0])

    train_set = CityScape(Config.train_img_dir,
                          Config.train_annot_dir, transform=train_transform)
    train_loader = DataLoader(train_set, batch_size=Config.batch_size, shuffle=True,
                              num_workers=Config.dataloader_num_worker)

    validation_set = CityScape(Config.valid_img_dir, Config.valid_annot_dir)
    validation_loader = DataLoader(
        validation_set, batch_size=Config.val_batch_size, shuffle=False)

    model = ERFPSPNet(shapeHW=[640, 640], num_classes=21)
    model.to(MyDevice)

    class_weights = torch.tensor([8.6979065, 8.497886 , 8.741297 , 5.983605 , 8.662319 , 8.681756 ,
       8.683093 , 8.763641 , 8.576978 , 2.7114885, 6.237076 , 3.582358 ,
       8.439253 , 8.316548 , 8.129169 , 4.312109 , 8.170293 , 6.91469  ,
       8.135018 , 0.  ,3.6]).cuda()

    # criterion = CrossEntropyLoss2d(weight=class_weights)
    criterion = FocalLoss2d(weight=class_weights)

    lr = Config.learning_rate
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=1e-3,
                                 betas=(0.9, 0.999),
                                 eps=1e-08,
                                 weight_decay=2e-4)

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=90, gamma=0.1)

    start_epoch = 0
    step_per_epoch = math.ceil(2975 / Config.batch_size)
    writer = SummaryWriter(Config.logdir)
    # writer.add_graph(model)

    if Config.train_with_ckpt:
        checkpoint = torch.load(Config.ckpt_path)
        print("Load",Config.ckpt_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']+1
        # loss = checkpoint['loss']
        model.train()

    for epoch in range(start_epoch, Config.max_epoch):

        for i, (image, annot) in enumerate(train_loader):
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
            print(f"{epoch}/{Config.max_epoch-1} epoch, {i}/{step_per_epoch} step, loss:{loss.item()}, "
                  f"{time_elapsed} sec/step; global step={global_step}")

        scheduler.step()
        if epoch > 40:
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
                'loss': loss.item()
            }, Config.ckpt_name + "_" + str(epoch) + ".pth")
            print("model saved!")

    val(model, validation_loader, is_print=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'loss': loss
    }, "checkpoints/final_focalloss_model.pth")
    print("Save model to disk!")
    writer.close()


if __name__ == '__main__':
    train()
