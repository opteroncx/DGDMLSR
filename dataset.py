import torch.utils.data as data
import torch
import numpy as np
import h5py
from skimage import transform,measure,color
# import cv2
import os
from PIL import Image
from torchvision import transforms
import random
from tqdm import tqdm
import pickle

class FastLoader(data.Dataset):
    '''
    read from disk to ram
    '''
    def __init__(self, file_path):
        super(FastLoader, self).__init__()
        self.readHR = []
        self.readLR = []
        self.HD = []
        self.LD = []
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        self.readHR,self.readLR,self.HD,self.LD = data
    def __getitem__(self, index):
        im_h = self.readHR[index]
        im_l = self.readLR[index]
        HR = transforms.ToTensor()(im_h)
        LR = transforms.ToTensor()(im_l)
        Hd = self.HD[index]
        Ld = self.LD[index]
        return LR,HR,Ld,Hd
        
    def __len__(self):
        len_h = len(self.readHR)
        len_l = len(self.readLR)
        if len_h >= len_l:
            len_file = len_l
        else:
            len_file = len_h
        return len_file

class FastLoader2(data.Dataset):
    '''
    read from disk to ram
    '''
    def __init__(self, file_path):
        super(FastLoader2, self).__init__()
        self.readHR = []
        self.readLR = []
        self.HD = []
        self.LD = []
        # self.bicLR = []
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        self.readHR,self.readLR,self.HD,self.LD = data
    def __getitem__(self, index):
        im_h = self.readHR[index]
        im_l = self.readLR[index]
        h,w = im_h.size
        bic_h = im_l.resize(((h*2,w*2)),Image.BICUBIC)
        HR = transforms.ToTensor()(im_h)
        LR = transforms.ToTensor()(im_l)
        Hd = self.HD[index]
        Ld = self.LD[index]
        bic_HR = transforms.ToTensor()(bic_h)
        return LR,HR,Ld,Hd,bic_HR
        
    def __len__(self):
        len_h = len(self.readHR)
        len_l = len(self.readLR)
        if len_h >= len_l:
            len_file = len_l
        else:
            len_file = len_h
        return len_file

def cal_mean(image):
    im = np.array(image)
    mean = np.mean(im)
    # mean_tensor = torch.from_numpy(mean)
    return mean

def test():
    file_path = "./"
    dfi = FastLoader(file_path)
