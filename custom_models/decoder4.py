import torch.nn as nn
import torch
import numpy as np
import torchvision.transforms as transforms

class decoder(nn.Module):
    def __init__(self, input_dim, output_shape):
        super().__init__()
        #self.t_conv1 = nn.ConvTranspose2d(48, 24, 4, stride=2,padding=1)
        #self.t_conv2 = nn.ConvTranspose2d(24, 8, 4, stride=2,padding=1)
        #self.t_conv3 = nn.Conv2d(8, 3, 3, stride=1,padding=1)
        
        #self.t_conv1 = nn.ConvTranspose2d(5, 16, 3, stride=1,padding=1)
        #self.t_conv2 = nn.ConvTranspose2d(16, 32, 3, stride=1,padding=1)
        #self.t_conv3 = nn.ConvTranspose2d(32, 64, 3, stride=1,padding=1)
        #self.t_conv4 = nn.ConvTranspose2d(64, 128, 3, stride=1,padding=1,output_padding=1)
        #self.t_conv5 = nn.ConvTranspose2d(12, 5, 7, stride=1,padding=3,output_padding=1)

        
        
        self.dec_model=nn.Sequential(
            nn.Linear(input_dim,128*32*32),
            nn.ReLU(),
            nn.Unflatten(1,unflattened_size=(128,32,32)),
            nn.ConvTranspose2d(128, 256, 3, stride=1,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 512, 3, stride=1,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 1024, 3, stride=1,padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            transforms.Resize(64,interpolation=0),
            nn.ConvTranspose2d(1024, 512, 3, stride=1,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 3, stride=1,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            transforms.Resize(128,interpolation=0),
            nn.ConvTranspose2d(256, 128, 3, stride=1,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, output_shape[0], 3, stride=1,padding=1),            
            #nn.Sigmoid()
            nn.ReLU()
            )

        self.dec_model.apply(self.weight_init)

    def weight_init(self,m):
        if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            nn.init.zeros_(m.bias)
        
        

    def forward(self, x):
        recon = self.dec_model(x)
        return recon
