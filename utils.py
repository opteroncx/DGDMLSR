import argparse
import torch
import os
import cv2
import pyssim
from scipy.ndimage import gaussian_filter
from numpy.lib.stride_tricks import as_strided as ast
from PIL import Image
from torch.autograd import Variable
import numpy as np
import time, math
import scipy.io as sio
from skimage import measure,color
from torchvision import transforms
# from skvideo import measure
import datetime
import shutil

def save_experiment():
    # 保存当前实验内容
    root_path = './experiments'
    if not os.path.exists(root_path):
        os.mkdir(root_path)
    t = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    code_path = os.path.join(root_path,t)
    if not os.path.exists(code_path):
        os.makedirs(code_path)
    copy_files('./',code_path)
    print('code copied to ',code_path)

def copy_files(source, target):
    files = os.listdir(source)
    for f in files:
        if f[-3:] == '.py' or f[-3:] == '.sh':
            print(f)
            shutil.copy(source+f, target)

def run_val_matlab(model,ipath):
    eng = matlab.engine.start_matlab()
    image = Image.open(ipath)
    im = np.array(image)
    im = color.rgb2ycbcr(im)[:,:,0]
    im = Image.fromarray(im)
    im = transforms.ToTensor()(im)
    print(im.shape)
    gen = model(im).cpu().data[0]
    gen_img = transforms.ToPILImage()(gen)
    gen_img.save('./tmp.png')
    niqe = eng.calc_NIQE('./tmp.png',4)
    return niqe

def run_val(model,ipath):
    image = Image.open(ipath)
    im = np.array(image)
    im = color.rgb2ycbcr(im)[:,:,0]
    im = Image.fromarray(im)
    im = transforms.ToTensor()(im)
    im = im.view(1,-1,im.shape[1],im.shape[2])
    # print(im.shape)
    im = im.cuda()
    gen = model(im).cpu()
    gen = gen.data[0].numpy().astype(np.float32)
    gen_img = ToImage(gen)
    gen_img = gen_img.transpose(1,2,0)
    print(gen_img.shape)
    save_figure(gen_img,'tmp.png','./')
    niqe_score = measure.niqe(gen_img)[0]
    return niqe_score

def stack(im):
    # im -> 1c-->3c
    print(im.shape)
    nim = np.zeros([im.shape[0],im.shape[1],3])
    nim[:,:,0] = im
    nim[:,:,1] = im
    nim[:,:,2] = im
    return nim

def ToImage(tensor):
    im_h_y = tensor
    im_h_y = im_h_y*255.
    im_h_y[im_h_y<0] = 0
    im_h_y[im_h_y>255.] = 255.
    # im_h_y = im_h_y[0,:,:]
    return im_h_y

def getYUV(content_rgb):
    image_YUV = cv2.cvtColor(content_rgb, cv2.COLOR_RGB2YUV)
    Y_i, U_i, V_i = cv2.split(image_YUV)
    return Y_i, U_i, V_i

def save_figure(img,name,opath):
    #保存图像
    out_path=opath
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    print('saved '+name)
    cv2.imwrite(out_path+name[:-4]+'.png',img)

def save_figure_rgb(img,name,opath):
    #保存图像
    out_path=opath
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    print('saved '+name)
    img.save(out_path+name[:-4]+'.png')

def PSNR(pred, gt, shave_border=0):
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)

#################################
def convert_rgb_to_y(image, jpeg_mode=False, max_value=255.0):
    if len(image.shape) <= 2 or image.shape[2] == 1:
        return image

    if jpeg_mode:
        xform = np.array([[0.299, 0.587, 0.114]])
        y_image = image.dot(xform.T)
    else:
        xform = np.array([[65.481 / 256.0, 128.553 / 256.0, 24.966 / 256.0]])
        y_image = image.dot(xform.T) + (16.0 * max_value / 256.0)

    return y_image


def convert_rgb_to_ycbcr(image, jpeg_mode=False, max_value=255):
    if len(image.shape) < 2 or image.shape[2] == 1:
        return image

    if jpeg_mode:
        xform = np.array([[0.299, 0.587, 0.114], [-0.169, - 0.331, 0.500], [0.500, - 0.419, - 0.081]])
        ycbcr_image = image.dot(xform.T)
        ycbcr_image[:, :, [1, 2]] += max_value / 2
    else:
        xform = np.array(
            [[65.481 / 256.0, 128.553 / 256.0, 24.966 / 256.0], [- 37.945 / 256.0, - 74.494 / 256.0, 112.439 / 256.0],
             [112.439 / 256.0, - 94.154 / 256.0, - 18.285 / 256.0]])
        ycbcr_image = image.dot(xform.T)
        ycbcr_image[:, :, 0] += (16.0 * max_value / 256.0)
        ycbcr_image[:, :, [1, 2]] += (128.0 * max_value / 256.0)

    return ycbcr_image


def convert_y_and_cbcr_to_rgb(y_image, cbcr_image, jpeg_mode=False, max_value=255.0):
    # if len(y_image.shape) <= 2:
    #     y_image = y_image.reshape[y_image.shape[0], y_image.shape[1], 1]

    if len(y_image.shape) == 3 and y_image.shape[2] == 3:
        y_image = y_image[:, :, 0:1]

    ycbcr_image = np.zeros([y_image.shape[0], y_image.shape[1], 3])
    ycbcr_image[:, :, 0] = y_image
    ycbcr_image[:, :, 1:3] = cbcr_image[:, :, 0:2]

    return convert_ycbcr_to_rgb(ycbcr_image, jpeg_mode=jpeg_mode, max_value=max_value)


def convert_ycbcr_to_rgb(ycbcr_image, jpeg_mode=False, max_value=255.0):
    rgb_image = np.zeros([ycbcr_image.shape[0], ycbcr_image.shape[1], 3])  # type: np.ndarray

    if jpeg_mode:
        rgb_image[:, :, [1, 2]] = ycbcr_image[:, :, [1, 2]] - (128.0 * max_value / 256.0)
        xform = np.array([[1, 0, 1.402], [1, - 0.344, - 0.714], [1, 1.772, 0]])
        rgb_image = rgb_image.dot(xform.T)
    else:
        rgb_image[:, :, 0] = ycbcr_image[:, :, 0] - (16.0 * max_value / 256.0)
        rgb_image[:, :, [1, 2]] = ycbcr_image[:, :, [1, 2]] - (128.0 * max_value / 256.0)
        xform = np.array(
            [[max_value / 219.0, 0, max_value * 0.701 / 112.0],
             [max_value / 219, - max_value * 0.886 * 0.114 / (112 * 0.587), - max_value * 0.701 * 0.299 / (112 * 0.587)],
             [max_value / 219.0, max_value * 0.886 / 112.0, 0]])
        rgb_image = rgb_image.dot(xform.T)

    return rgb_image

def resize_image_by_pil(image, scale, resampling_method="bicubic"):
    width, height = image.shape[1], image.shape[0]
    new_width = int(width * scale)
    new_height = int(height * scale)

    if resampling_method == "bicubic":
        method = Image.BICUBIC
    elif resampling_method == "bilinear":
        method = Image.BILINEAR
    elif resampling_method == "nearest":
        method = Image.NEAREST
    else:
        method = Image.LANCZOS

    if len(image.shape) == 3 and image.shape[2] == 3:
        image = Image.fromarray(image, "RGB")
        image = image.resize([new_width, new_height], resample=method)
        image = np.asarray(image)
    elif len(image.shape) == 3 and image.shape[2] == 4:
        # the image may has an alpha channel
        image = Image.fromarray(image, "RGB")
        image = image.resize([new_width, new_height], resample=method)
        image = np.asarray(image)
    else:
        image = Image.fromarray(image.reshape(height, width))
        image = image.resize([new_width, new_height], resample=method)
        image = np.asarray(image)
        image = image.reshape(new_height, new_width, 1)
    return image