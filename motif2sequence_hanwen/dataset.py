from basemodel import BaseModel
import csv
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np
from torch.utils.data import ConcatDataset
import random
import torch
import torch.nn as nn

class Dataset(object):

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __add__(self, other):
        return ConcatDataset([self, other])


def crop1(img, pos, size0):
    ow, oh = img.size
    x1 = pos[0]
    y1 = pos[1]
    tw = th = size0
    if (ow > tw or oh > th):
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img


def flip1(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


class CMP_dataset(Dataset):

    def __init__(self, opt, split_ratio=0.9):
        self.path = 'data/part_dataset.csv'
        self.input = {}
        self.input['A'] = []
        self.input['B'] = []
        f = open(self.path, 'r')
        self.reader_num = csv.reader(f)
        self.sizes = 0
        for i in self.reader_num:
            self.sizes += 1
        self.split_ration = split_ratio
        self.isTrain = opt.isTrain
        self.cut = int(self.sizes * self.split_ration)
        self.data_loading()

    def data_loading(self):
        n = 0
        f1 = open(self.path, 'r')
        self.reader = csv.reader(f1)
        for i in self.reader:
            if self.isTrain:
                if 0 < n < int(self.split_ration * self.sizes):
                    self.input['A'].append(i[12])
                    self.input['B'].append(i[11])
            else:
                if n >= int(self.split_ration * self.sizes):
                    self.input['A'].append(i[12])
                    self.input['B'].append(i[11])
            n = n + 1

    def oneHot(self, sequence):
        oh_dict = {'A': 0, 'T': 1, 'C': 2, 'G': 3, 'M': 4}
        oh = np.zeros([5, len(sequence)])
        for i in range(len(sequence)):
            oh[oh_dict[sequence[i]], i] = 1
        return oh


    def __getitem__(self, item):
        A = self.input['A'][item]
        B = self.input['B'][item]
        A = self.oneHot(A)
        B = self.oneHot(B)
        A1 = np.zeros([5, 128])
        B1 = np.zeros([5, 128])
        A1[:, 0 : 118] = A
        B1[:, 0 : 118] = B
        #A = torch.from_numpy(A)
        #A = A.to(torch.double)
        A = transforms.ToTensor()(A)
        A = torch.squeeze(A)
        A = A.float()
        #A = transforms.Normalize((0.5,), (0.5,))(A)
        #B = torch.from_numpy(B)
        #B = B.double()
        #B = transforms.Normalize((0.5,), (0.5,))(B)
        B = transforms.ToTensor()(B)
        B = torch.squeeze(B)
        B = B.float()
        return {'A': A, 'B': B}

    def __len__(self):
        return len(self.input['A'])
