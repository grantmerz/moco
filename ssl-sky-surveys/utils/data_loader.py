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
#import skimage.transform
#import h5py
#from utils.sdss_dr12_galactic_reddening import SDSSDR12Reddening

import sys
sys.path.append('/home/g4merz/galaxyQuery/moco')

class RandomRotate:
  def __call__(self, image):
    # print("RR", image.shape, image.dtype)
    return (skimage.transform.rotate(image, np.float32(360*np.random.rand(1)))).astype(np.float32)

class JitterCrop:
  def __init__(self, outdim, jitter_lim=None):
    self.outdim = outdim
    self.jitter_lim = jitter_lim
    self.offset = self.outdim//2

  def __call__(self, image):
    # print("JC", image.shape, image.dtype)
    center_x = image.shape[0]//2
    center_y = image.shape[0]//2
    if self.jitter_lim:
      center_x += int(np.random.randint(-self.jitter_lim, self.jitter_lim+1, 1))
      center_y += int(np.random.randint(-self.jitter_lim, self.jitter_lim+1, 1))

    return image[(center_x-self.offset):(center_x+self.offset), (center_y-self.offset):(center_y+self.offset)]

def get_data_loader(data, aug_plus, crop_size, jc_jit_limit, distributed):

    traindir = os.path.join(data, 'ae_data')
    #normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                 std=[0.229, 0.224, 0.225])
    if aug_plus:
        # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
        augmentation = [
            #transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            #transforms.RandomApply([
            #    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            #], p=0.8),
            #transforms.RandomGrayscale(p=0.2),
            #transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.5),
            #transforms.RandomHorizontalFlip(),
            JitterCrop(outdim=crop_size,jitter_lim=jc_jit_limit),
            transforms.RandomRotation((0,360)),
            transforms.ToTensor(),
            #normalize
        ]
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

    train_dataset = datasets.ImageFolder(
        traindir,
        moco.loader.TwoCropsTransform(transforms.Compose(augmentation)))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
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
