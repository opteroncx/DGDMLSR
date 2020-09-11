import numpy as np 
from skimage import io,transform,color
import os

'''
Inverse depth
closer points have higher values
'''

path = './nyu_depths/'
outpath = './depth_inv/'
if not os.path.exists(outpath):
    os.mkdir(outpath)
inames = os.listdir(path)
for iname in inames:
    im = io.imread(path+iname)
    print(im.shape)
    im = 255-im
    io.imsave(outpath+iname,im)