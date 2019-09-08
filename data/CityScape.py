import os
import torch
from torch.utils import data
from torchvision.transforms import ToTensor, Normalize
import numpy as np
import cv2

class CityScape(data.Dataset):
    def __init__(self, img_dir, annot_dir, transform=None):
        self._transform = transform

        self._imgs = sorted([os.path.join(img_dir, img) for img in os.listdir(img_dir)
                     if img.endswith('.png')])
        self._imgs_annot = sorted([os.path.join(annot_dir, img) for img in os.listdir(annot_dir)
                     if img.endswith('.png')])

    def __getitem__(self, index):

        image = cv2.imread(self._imgs[index])
        annot = cv2.imread(self._imgs_annot[index], 0)
        image = image[:,:,(2,1,0)]

        if self._transform is not None:
            image, annot = self._transform(image, annot)
            return image, annot
        else:
            image = ToTensor()(image)
            annot = torch.from_numpy(annot)
            return image, annot

    def __len__(self):
        return len(self._imgs)
