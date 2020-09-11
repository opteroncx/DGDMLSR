from skimage import io,color,exposure
import os
import numpy as np
import shutil
from tqdm import tqdm
from PIL import Image
import random
import pickle

def data_augment(im,num):
    org_image = im
    if num ==0:    
        ud_image = np.flipud(org_image)
        tranform = ud_image
    elif num ==1:      
        lr_image = np.fliplr(org_image)
        tranform = lr_image
    elif num ==2:
        lr_image = np.fliplr(org_image)
        lrud_image = np.flipud(lr_image)        
        tranform = lrud_image
    elif num ==3:
        rotated_image1 = np.rot90(org_image)        
        tranform = rotated_image1
    elif num ==4: 
        rotated_image2 = np.rot90(org_image, -1)
        tranform = rotated_image2
    elif num ==5: 
        rotated_image1 = np.rot90(org_image) 
        ud_image1 = np.flipud(rotated_image1)
        tranform = ud_image1
    elif num ==6:        
        rotated_image2 = np.rot90(org_image, -1)
        ud_image2 = np.flipud(rotated_image2)
        tranform = ud_image2
    else:
        tranform = org_image
    return tranform

def shave_border(im,border = 12):
    if len(im.shape)==3:
        nim = im[border:-border,border:-border,:]
    else:
        nim = im[border:-border,border:-border]
    return nim

def gird_crop(im,savedir,stride=64,size=128,show=True):
    if not os.path.exists(savedir[0]):
        os.mkdir(savedir[0])
    if not os.path.exists(savedir[1]):
        os.mkdir(savedir[1])
    im0,im1 = im[0],im[1]
    nc = len(im0.shape)
    h,w = im0.shape[:2]
    # print(h,w)
    counter = 0
    mean_sub = []
    for i in range(0,h-size,stride):
        for j in range(0,w-size,stride):
            if nc ==2 :
                sub0 = im0[i:i+size,j:j+size,:]
                sub1 = im1[i:i+size,j:j+size]
            else:
                sub0 = im0[i:i+size,j:j+size,:]
                sub1 = im1[i:i+size,j:j+size]
            # image contrast，ignore depth contrast
            result=exposure.is_low_contrast(sub0)
            print(result)
            if not result:
                io.imsave(savedir[0]+str(counter)+'.png',sub0)
                io.imsave(savedir[1]+str(counter)+'.png',sub1)
                counter += 1
                mean = np.mean(sub1)
                mean_sub.append(mean)
                if show:
                    print('id=%d mean=%f'%(counter,mean))
    return mean_sub

def gen_patch(num):
    img = './nyu_images/%d.bmp'%num
    depth_inv = './depth_inv/%d.png'%num
    img =shave_border(io.imread(img)) 
    depth_inv =shave_border(io.imread(depth_inv)) 
    print(img.shape,depth_inv.shape)
    size = 256
    stride = size//2
    scale = 2
    # crop image
    mean_sub = gird_crop([img,depth_inv],['./data/img0_%d/'%num,'./data/depth0_%d/'%num],stride,size,show=False)
    # mean_sub = gird_crop(depth_inv,,stride,size,show=False)
    # smaller crop
    mean_sub_small = gird_crop([img,depth_inv],['./data/img1_%d/'%num,'./data/depth1_%d/'%num],stride//scale,size//scale,show=False)
    # mean_sub_small = gird_crop(depth_inv,'./depth1/',stride//2,size//2,show=False)

    mean_depth = np.mean(mean_sub)
    sigma_depth = np.std(mean_sub)
    print('mean=%f,std=%f'%(mean_depth,sigma_depth))
    HR_ids = []
    LR_ids = []
    for i in range(len(mean_sub)):
        if mean_sub[i]>mean_depth:
            HR_ids.append(i)

    # mean_depth = np.mean(mean_sub_small)
    # sigma_depth = np.std(mean_sub_small)
    # print('mean=%f,std=%f'%(mean_depth,sigma_depth))
    for i in range(len(mean_sub_small)):
        if mean_sub_small[i]<mean_depth:
            LR_ids.append(i)
    
    print(len(HR_ids))
    print(len(LR_ids))
    # move to HR-LR Dir
    if not os.path.exists('./data/LR_%d/'%num):
        os.mkdir('./data/LR_%d/'%num)
    if not os.path.exists('./data/HR_%d/'%num):
        os.mkdir('./data/HR_%d/'%num)
    if not os.path.exists('./data/LRd_%d/'%num):
        os.mkdir('./data/LRd_%d/'%num)
    if not os.path.exists('./data/HRd_%d/'%num):
        os.mkdir('./data/HRd_%d/'%num)        
    for HR_id in HR_ids:
        name = str(HR_id)
        im = io.imread('./data/img0_%d/'%num+name+'.png')
        # imy = color.rgb2ycbcr(im)[:,:,0]
        imy = im
        imd = io.imread('./data/depth0_%d/'%num+name+'.png')
        print('dshape',imd.shape)
        for i in range(8):
            t=imy
            t = data_augment(t,i)
            d = data_augment(imd,i)
            print('t==>',i,t.shape)
            # io.imsave('out/im'+str(i)+'.jpg',t)
            io.imsave('./data/HR_%d/'%num+name+'_'+str(i)+'.png',t.astype('uint8'))
            io.imsave('./data/HRd_%d/'%num+name+'_'+str(i)+'.png',d.astype('uint8'))
        # shutil.copyfile('./img0/'+name,'./HR/'+name)
    for LR_id in LR_ids:
        name = str(LR_id)
        im = io.imread('./data/img1_%d/'%num+name+'.png')
        # imy = color.rgb2ycbcr(im)[:,:,0]
        imy = im
        imd = io.imread('./data/depth1_%d/'%num+name+'.png')
        for i in range(8):
            t=imy
            t = data_augment(t,i)
            d = data_augment(imd,i)
            print('t==>',i,t.shape)
            io.imsave('./data/LR_%d/'%num+name+'_'+str(i)+'.png',t.astype('uint8'))
            io.imsave('./data/LRd_%d/'%num+name+'_'+str(i)+'.png',d.astype('uint8'))
        # shutil.copyfile('./img1/'+name,'./LR/'+name)

if __name__ == '__main__':
    # Test image_86 of nyu
    gen_patch(i)