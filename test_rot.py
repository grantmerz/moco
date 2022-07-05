import torch
import random
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import torchvision.datasets as datasets

import torchvision.transforms as transforms
import os
import moco.loader
import glob

from PIL import Image
import h5py
from astropy.visualization import MinMaxInterval

import skimage.transform






class RandomRotate:
  def __call__(self, image):
    # print("RR", image.shape, image.dtype)
    # print("RR", image.shape, image.dtype)
    image = np.transpose(image, (1, 2, 0))
    a=np.float32(360*np.random.rand(1))[0]
    print(a)
    rot = skimage.transform.rotate(image, a).astype(np.float32)
    #rot = skimage.transform.rotate(image, 140).astype(np.float32)
    return np.transpose(rot, (2, 0, 1))



path = '/home/g4merz/data/DES_jobs/DR2_sample_alltiles_imagcut/31a844eb0a5f42b9a9f62e9dc118b316/magcut13_16/sorted_data_clean.h5'
f = h5py.File(path, 'r')
i = f['images'][0]

ir = RandomRotate()(i)
print(ir)
