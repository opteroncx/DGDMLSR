# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import h5py
import os 
f=h5py.File("nyu_depth_v2_labeled.mat")
images=f["images"]
images=np.array(images)
# images = images.transpose((0,1,3,2))

path_converted='./nyu_images'
if not os.path.isdir(path_converted):
    os.makedirs(path_converted)

from PIL import Image
images_number=[]
for i in range(len(images)):
    images_number.append(images[i])
    a=np.array(images_number[i])
#    print len(img)
    #img=img.reshape(3,480,640)
 #   print img.shape
    r = Image.fromarray(a[0]).convert('L')
    g = Image.fromarray(a[1]).convert('L')
    b = Image.fromarray(a[2]).convert('L')
    img = Image.merge("RGB", (r, g, b))
    img = img.transpose(Image.ROTATE_270)
    #img = img.rotate(270) 
    # plt.imshow(img)
    # plt.axis('off')
    # plt.show()
    iconpath='./nyu_images/'+str(i)+'.bmp'
    img.save(iconpath,optimize=True)

f=h5py.File("nyu_depth_v2_labeled.mat")
depths=f["depths"]
depths=np.array(depths)

path_converted='./nyu_depths/'
if not os.path.isdir(path_converted):
    os.makedirs(path_converted)

max = depths.max()

depths = depths / max * 255
depths = depths.transpose((0,2,1))


for i in range(len(depths)):
    # labels_number.append(labels[i])
    # labels_0=np.array(labels_number[i])
    # print labels_0.shape
    # print type(labels_0)
    print(str(i) + '.png') 
    depths_img= Image.fromarray(np.uint8(depths[i]))
    depths_img = depths_img.transpose(Image.FLIP_LEFT_RIGHT)
    #depths_img = depths_img.transpose((1,0,2));
    # depths_img = depths_img.rotate(270)
    iconpath=path_converted+str(i)+'.png'
    depths_img.save(iconpath, 'PNG', optimize=True)

f=h5py.File("nyu_depth_v2_labeled.mat")
labels=f["labels"]
labels=np.array(labels)

path_converted='./nyu_labels'
if not os.path.isdir(path_converted):
    os.makedirs(path_converted)

labels_number=[]
for i in range(len(labels)):
    labels_number.append(labels[i])
    labels_0=np.array(labels_number[i])
    #print labels_0.shape
    print(type(labels_0))
    label_img=Image.fromarray(np.uint8(labels_number[i]))  
    label_img = label_img.rotate(270)    
    iconpath='./nyu_labels/'+str(i)+'.png'
    label_img.save(iconpath, 'PNG', optimize=True)
