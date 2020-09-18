# -*- coding:utf-8 -*-
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
from skimage import io,color,transform
# from skvideo import measure
# import matplotlib.pyplot as plt
from utils import *
# plt.switch_backend('agg')
# import niqe_score
parser = argparse.ArgumentParser(description="PyTorch LapSRN Test")
parser.add_argument("--cuda", action="store_true", help="use cuda?")
parser.add_argument("--model", default="./checkpoint523/biwgan_model_epoch_330.pth", type=str, help="model path")
parser.add_argument("--image", default="butterfly_GT", type=str, help="image name")
parser.add_argument("--scale", default=2, type=int, help="scale factor, Default: 4")
parser.add_argument("--testdir", default="./test/", type=str, help="")
parser.add_argument("--mode", default="evaluate", type=str, help="")

opt = parser.parse_args()
cuda = opt.cuda

if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

def print_summary(psnr,ssim):
    print("Scale=",opt.scale)
    print("PSNR=", psnr)
    print("SSIM=",ssim)

def sr(test_im_path,mpath,mname,opath):
    save=True
    eva=False
    convert=True
    img = cv2.imread(test_im_path)
    name = os.path.split(test_im_path)[1]
    # print(name)
    niqe = predict(img,save,convert,eva,name,mpath,mname,opath)
    return niqe

def predict(img_read,save,convert,eva,name,mpath,mname,opath):
    if convert:
        if eva:
            h,w,_=img_read.shape
            im_gt_y=convert_rgb_to_y(img_read)
            gt_yuv=convert_rgb_to_ycbcr(img_read)
            im_gt_y=im_gt_y.astype("float32")
            sc=1.0/opt.scale
            img_y=resize_image_by_pil(im_gt_y,sc)
            img_y=img_y[:,:,0]
            im_gt_y=im_gt_y[:,:,0]
        else:
            sc = opt.scale
            img_y=convert_rgb_to_y(img_read).astype("float32")
            img_resize = resize_image_by_pil(img_read,sc)
            gt_yuv=convert_rgb_to_ycbcr(img_resize)
    else:
        im_gt_y,img_y=img_read
        im_gt_y=im_gt_y.astype("float32")
    im_input = img_y/255.
    im_input = Variable(torch.from_numpy(im_input).float()).view(1, -1, im_input.shape[0], im_input.shape[1])
    #使用cuda加速
    model = torch.load(mpath)["model"]
    if cuda:
        model = model.cuda()
        im_input = im_input.cuda()
    else:
        model = model.cpu()

    start_time = time.time()
    HR_2x = model(im_input)
    elapsed_time = time.time() - start_time
    HR_2x = HR_2x[-1].cpu()
    im_h_y = HR_2x.data[0].numpy().astype(np.float32)

    im_h_y = ToImage(im_h_y)
    # res = res.cpu().data[0].numpy().astype(np.float32)
    # disp = ToImage(res)[0,:,:]
    # disp = stack(disp)
    # ui = ui.cpu().data[0].numpy().astype(np.float32)
    # ui = ToImage(ui)[0,:,:]
    # print('res=',disp.shape)
    # Test NIQE
    # niqe = measure.niqe(im_h_y)[0]
    # print('NIQE=',niqe)
    niqe=0
    if save:
        recon=convert_y_and_cbcr_to_rgb(im_h_y, gt_yuv[:, :, 1:3])
        save_figure(recon,mname,opath)
        # save_figure(disp,'res_'+name)
        # save_figure(ui,'uim_'+name)
    if eva:
        #PSNR and SSIM
        psnr_predicted = PSNR(im_gt_y, im_h_y,shave_border=opt.scale)
        ssim_predicted = pyssim.compute_ssim(im_gt_y, im_h_y)
        print("test psnr/ssim=%f/%f"%(psnr_predicted,ssim_predicted))
        return psnr_predicted,ssim_predicted
    return niqe

##################################
def main():
    opt.scale = 2
    model_path = './Urban_models2/model_0/'
    out_path = 'ub_down2_%dx/'%opt.scale
    test_im_path = './data/ubimages/0.png'
    models = os.listdir(model_path)
    opath = out_path
    best_niqe = 10000
    best_model = 0
    for i in range(1,len(models)):
        mpath = model_path+'biwgan_model_epoch_%d.pth'%i
        mname = 'epoch'+str(i)+'.png'
        niqe = sr(test_im_path,mpath,mname,opath)
        print('model=%d,niqe=%.5f'%(i,niqe))
        if niqe < best_niqe:
            best_niqe = niqe
            best_model = i
    print('best_model=%d,best_niqe=%.5f'%(best_model,best_niqe))

if __name__ == "__main__":
    main()

