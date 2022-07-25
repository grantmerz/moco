#import logging
#import glob
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

#import sys
#sys.path.append('/home/g4merz/galaxyQuery/moco')


#code to reporduce aggregate MAD for images
#train_data = data.reshape(data.shape[0],228*228,3)
#MAD = scipy.stats.median_abs_deviation(train_data,axis=1)
#print(MAD.shape)
#agMAD1 = scipy.stats.median_abs_deviation(MAD,axis=0)



class RandomRotate:
  def __call__(self, image):
    # print("RR", image.shape, image.dtype)
    # print("RR", image.shape, image.dtype)
    image = np.transpose(image, (1, 2, 0))
    a = np.float32(360*np.random.rand(1))[0]
    rot = skimage.transform.rotate(image, a).astype(np.float32)
    return np.transpose(rot, (2, 0, 1))

class JitterCropPNG:
  def __init__(self, outdim, jitter_lim=None):
    self.outdim = outdim
    self.jitter_lim = jitter_lim
    self.offset = self.outdim//2

  def __call__(self, image):
    #print(image)
    #print("JC", image.size)

    image = transforms.ToTensor()(image).numpy()
    #print(image.shape)
    image = np.transpose(image, (1, 2, 0))

    center_x = image.shape[0]//2
    center_y = image.shape[1]//2
    if self.jitter_lim:
      center_x += int(np.random.randint(-self.jitter_lim, self.jitter_lim+1, 1))
      center_y += int(np.random.randint(-self.jitter_lim, self.jitter_lim+1, 1))

    crop=image[(center_x-self.offset):(center_x+self.offset), (center_y-self.offset):(center_y+self.offset)]
    #print(center_x-self.offset,center_x+self.offset)
    #print(crop.shape)
    return transforms.ToTensor()(crop)

class JitterCropFITS:
    def __init__(self, outdim, jitter_lim=None):
        self.outdim = outdim
        self.jitter_lim = jitter_lim
        self.offset = self.outdim//2

    def __call__(self, image):
        #print(image.shape)
        #print("JC", image.size)

        image = torch.from_numpy(image)
        #print(image.shape)
        image = np.transpose(image, (1, 2, 0))

        center_x = image.shape[0]//2
        center_y = image.shape[1]//2
        if self.jitter_lim:
            center_x += int(np.random.randint(-self.jitter_lim, self.jitter_lim+1, 1))
            center_y += int(np.random.randint(-self.jitter_lim, self.jitter_lim+1, 1))

        crop=image[(center_x-self.offset):(center_x+self.offset), (center_y-self.offset):(center_y+self.offset)]
        #print(center_x-self.offset,center_x+self.offset)
        #print(crop.shape)
        
        #return crop
        return torch.permute(crop, (2, 0, 1))

class AddGaussianNoise:
    def __init__(self, mean, std):
        self.std = std
        self.mean = mean
        
    def __call__(self, image):
        
        gaus = torch.randn(image.size())
        var = torch.tensor(self.std*np.random.uniform(1,3))
        noiseim = image+ torch.einsum('bcd,b->bcd', gaus, var)
        return noiseim.float()
        
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class Normalize:
    def __init__(self):
        self.interval = MinMaxInterval()

    def __call__(self,image):
        image = image.numpy()
        for i in range(image.shape[0]):
            image[i] = self.interval(image[i])
        return image


class NormalizeRange:
    def __init__(self,rng):
        #self.interval = MinMaxScaler(feature_range=(-1,1))
        self.rng=rng
    def __call__(self,image):
        image = image.numpy()
        
        interval = (image.max()-image.min())
        image = (image-image.min())/interval * (self.rng[1]-self.rng[0]) + self.rng[0]
        return image

      
class Scale:
    def __init__(self,scale):
        self.scale = scale

    def __call__(self,image):
        image = image/self.scale
        return image

class Shift:
    def __call__(self,image):
        #image = image-image.min()+1
        image = np.log10(image)
        return image

    
class Clip:
    def __call__(self,image):
        #image = np.maximum(0,image)
        maskind = image<=0
        image[maskind]=1e-1
        mask = np.ones((image.shape))
        mask[maskind] = 1e-6
        
        return image#,mask


class BandCut:
    def __init__(self,bands):
        self.bandlist = {
            "g": 0,
            "r": 1,
            "i": 2,
            "z": 3,
            "y": 4,
        }

        self.indices=[]

        for b in bands:
          ind = self.bandlist[b]
          self.indices.append(ind)

    def __call__(self,image):
        imi = image[self.indices,:,:]
        return imi

      
class DESPNGDataset(Dataset):
    def __init__(self,path,transform,png_type):
        pngpath = path+'*'+png_type+'.png'
        print(pngpath)
        self.image_paths = glob.glob(pngpath)  # Should contain a list of image paths of your desired class: e.g. ['./data/class0/img0.png', './data/class0/img1.png', ...]
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)    
     
    
    def pil_loader(self,impath):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(impath, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")
        

    def __getitem__(self, idx):
        imp=self.image_paths[idx]
        image=self.pil_loader(imp)
        
        return self.transform(image), torch.tensor([0])



class DESFITSDataset(Dataset):
    def __init__(self, path, transform):
        self.path = path
        self.transform = transform
        #self._open_file()

    def _open_file(self):
        #print('Loading Data')
        self.file = h5py.File(self.path, 'r')
        #self.size = self.file['obj_ids'].shape[0]

    def __len__(self):
        with h5py.File(self.path, 'r') as _f:
          size = _f['obj_ids'].shape[0]
        return size

    def __getitem__(self, idx):
        self._open_file()
        image = self.file['images'][idx]
        label=torch.tensor([0])
    
        return self.transform(image), label


  
def get_data_loader(data, scale, nrange, png, png_type,aug_plus, crop_size, jc_jit_limit, distributed):

    traindir = data
    #traindir = os.path.join(data, 'ae_data')

    #agMAD = np.array([0.00784314, 0.0117647,  0.01176471])

    #normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                 std=[0.229, 0.224, 0.225])
    if aug_plus:
        if png:
        # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
            augmentation = [
                #transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
                #transforms.RandomApply([
                #    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                #], p=0.8),
                #transforms.RandomGrayscale(p=0.2),
                #transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.5),
                #transforms.RandomHorizontalFlip(),
                transforms.RandomRotation((0,360)),
                JitterCropPNG(outdim=crop_size,jitter_lim=jc_jit_limit),
                #transforms.RandomRotation((0,360)),
                #AddGaussianNoise(0,agMAD),
                #transforms.ToTensor(),
                #normalize
            ]
            
            train_dataset = DESPNGDataset(
            traindir,
            moco.loader.TwoCropsTransform(transforms.Compose(augmentation)),png_type)


            
        else:
            augmentation = [
                #transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
                #transforms.RandomApply([
                #    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                #], p=0.8),
                #transforms.RandomGrayscale(p=0.2),
                #transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.5),
                #Scale(scale),
                BandCut(['i']),
                RandomRotate(),
                JitterCropFITS(outdim=crop_size,jitter_lim=jc_jit_limit),
                #Normalize()
                Clip(),
                Shift(),
                Scale(scale)
                #NormalizeRange(nrange)
                ##AddGaussianNoise(0,agMAD),
                #transforms.ToTensor(),
                #normalize
            ]
            
            train_dataset = DESFITSDataset(
            traindir,
            moco.loader.TwoCropsTransform(transforms.Compose(augmentation)))
        
    else:
        # MoCo v1's aug: the same as InstDisc https://arxiv.org/abs/1805.01978
        augmentation = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]


    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,shuffle=True)
    else:
        train_sampler = None

    return train_dataset, train_sampler

class SDSSDataset(Dataset):
  def __init__(self, num_classes, files_pattern, transform, load_specz, load_ebv, specz_upper_lim=None):
    self.num_classes = num_classes
    self.files_pattern = files_pattern
    self.transform = transform
    self.load_specz = load_specz
    self.load_ebv = load_ebv
    self.specz_upper_lim = specz_upper_lim
    self._get_files_stats()

  def _get_files_stats(self):
    self.files_paths = glob.glob(self.files_pattern)
    self.n_files = len(self.files_paths)
    with h5py.File(self.files_paths[0], 'r') as _f:
      self.n_samples_per_file = _f['images'].shape[0]
    self.n_samples = self.n_files * self.n_samples_per_file

    self.files = [None for _ in range(self.n_files)]
    logging.info("Found {} at path {}. Number of examples: {}".format(self.n_files, self.files_pattern, self.n_samples))

  def _open_file(self, ifile):
    self.files[ifile] = h5py.File(self.files_paths[ifile], 'r')

  def __len__(self):
    return self.n_samples

  def __getitem__(self, global_idx):
    ifile = int(global_idx/self.n_samples_per_file)
    local_idx = int(global_idx%self.n_samples_per_file)

    if not self.files[ifile]:
      self._open_file(ifile)

    if self.load_specz:
      specz = self.files[ifile]['specz_redshift'][local_idx]
      # hard-coded numbers are specific to the dataset used in this tutorial
      if specz >= self.specz_upper_lim:
        specz = self.specz_upper_lim - 1e-6
      specz_bin = torch.tensor(int(specz//(self.specz_upper_lim/self.num_classes)))

    # we flip channel axis because all tranforms we use assume HWC input,
    # last transform, ToTensor, reverts this operation
    image = np.swapaxes(self.files[ifile]['images'][local_idx], 0, 2)

    if self.load_ebv:
      ebv = self.files[ifile]['e_bv'][local_idx]
      out = [image, ebv]
    else:
      out = image

    if self.load_specz:
      return self.transform(out), specz_bin, torch.tensor(specz)
    else:
      return self.transform(out)
