import os
import argparse
from collections import OrderedDict

import torch
import torch.nn as nn
import logging

import models.resnet
#from utils.YParams import YParams
from utils.data_loader import get_data_loader
import torchvision.models as basemodels

def load_experiment(yaml_config_file='./config/photoz.yaml', config='default', load_best_ckpt=True, device="cpu"):
  params = YParams(yaml_config_file, config)

  # setup output directory
  expDir = os.path.join('./expts', config)
  if not os.path.isdir(expDir):
    logging.error("%s not found"%expDir)
    exit(1)

  params['experiment_dir'] = os.path.abspath(expDir)
  params['checkpoint_path'] = os.path.join(expDir, 'checkpoints/ckpt.tar')

  if not os.path.isfile(params.checkpoint_path):
    logging.error("%s not found"%params.checkpoint_path)
    exit(1)

  if load_best_ckpt and not os.path.isfile(params.checkpoint_path.replace('.tar', '_best.tar')):
    logging.warning("No best checkpoint exists, loading last checkpoint instead")
    load_best_ckpt = False

  params.log()
  params['log_to_screen'] = True

  checkpoint_path = params.checkpoint_path
  if load_best_ckpt:
    checkpoint_path = checkpoint_path.replace('.tar', '_best.tar')

  model = load_model_from_checkpoint(checkpoint_path, params.num_channels, num_classes=params.num_classes, device=device)

  return model, params

def load_model_from_checkpoint(checkpoint_path, basenet=False, num_channels=3, num_classes=128, device="cpu",mlp=False):
  if basenet:
    model = basemodels.resnet50()#.to(device)

  else:
    model = models.resnet.resnet50(num_channels=num_channels, num_classes=num_classes)#.to(device)
    
  if mlp:
    dim_mlp = model.fc.weight.shape[1]
    model.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), model.fc)

  model = model.to(device)
    
  logging.info("Loading checkpoint %s"%checkpoint_path)
  restore_checkpoint(model, checkpoint_path,mlp)

  return model

def restore_checkpoint(model, checkpoint_path,mlp):
  checkpoint = torch.load(checkpoint_path, map_location='cpu')

  # some checkpoints have a different name for the model key
  model_key = 'model_state' if 'model_state' in checkpoint else 'state_dict'

  new_model_state = OrderedDict()
  for key in checkpoint[model_key].keys():
    if 'encoder' in key:
      if 'encoder_q' in key:
        name = str(key).replace('module.encoder_q.', '')
        new_model_state[name] = checkpoint[model_key][key]
    elif 'module.' in key:
      name = str(key).replace('module.', '')
      new_model_state[name] = checkpoint[model_key][key]

  msg = model.load_state_dict(new_model_state, strict=False)
  #if msg.missing_keys == ['fc.weight', 'fc.bias']:
  #  logging.info("Printing a pretrained model without FC layers")
  #  model.fc = Identity()

  #added in to match moco 2 layer MLP projection head

  if not mlp:
    logging.info("Printing a pretrained model without FC layers")
    model.fc = Identity()

  logging.info("Chekpoint loaded. Checkpoint epoch %d"%checkpoint['epoch'])

class Identity(torch.nn.Module):
  def __init__(self):
    super(Identity, self).__init__()

  def forward(self, x):
    return x
