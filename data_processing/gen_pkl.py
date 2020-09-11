import os
import random
import numpy as np 
from tqdm import tqdm
from PIL import Image
import pickle
from skimage import color


def save_pkl(num):
    ph,pl,dh,dl = ["./data/HR_%d"%num,"./data/LR_%d"%num,"./data/HRd_%d"%num,"./data/LRd_%d"%num]
    HR = os.path.join(ph)
    LR = os.path.join(pl)
    Hd = os.path.join(dh)
    Ld = os.path.join(dl)
    HR_list = os.listdir(HR)
    LR_list = os.listdir(LR)
    # shuffle images
    # keep the file name of image and depth
    random.shuffle(HR_list)
    random.shuffle(LR_list)
    readHR = []
    readLR = []
    HD = []
    LD = []
    for im in tqdm(HR_list):
        im_h = Image.open(os.path.join(HR,im))
        im_h = convert(im_h)
        readHR.append(im_h)
        im_hd = Image.open(os.path.join(Hd,im))
        cHd = cal_mean(im_hd)
        HD.append(cHd)
    print('HR read ok')
    for im in tqdm(LR_list):
        im_l = Image.open(os.path.join(LR,im))
        im_l = convert(im_l)
        readLR.append(im_l)
        im_ld = Image.open(os.path.join(Ld,im))
        cLd = cal_mean(im_ld)
        LD.append(cLd)
    print('LR read ok')
    data = [readHR,readLR,HD,LD]
    with open('./pklsv/nyu_%d.pkl'%num, 'wb') as f:
        pickle.dump(data, f)

def cal_mean(image):
    im = np.array(image)
    mean = np.mean(im)
    return mean

def convert(im):
    im = np.array(im)
    if im.shape[2]==3:
        nim = color.rgb2ycbcr(im)[:,:,0]
    return Image.fromarray(nim.astype('uint8'))

if __name__ == '__main__':
    for i in range(86,87):
        save_pkl(i)