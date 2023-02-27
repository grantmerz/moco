import torch.nn as nn
import torch
import numpy as np
import torchvision.transforms as transforms

class Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.map_model=nn.Sequential(
            nn.Linear(input_dim,70),
            #nn.BatchNorm1d(70),
            nn.PReLU(),
            nn.Linear(70,80),
            #nn.BatchNorm1d(100),
            nn.PReLU(),
            nn.Linear(80,90),
            nn.PReLU(),
            nn.Linear(90,100),
            nn.PReLU(),
            nn.Linear(100,110),
            nn.PReLU(),
            nn.Linear(110,output_dim),
            nn.Tanh()
            )

    #    self.map_model.apply(self.weight_init)

    #def weight_init(self,m):
    #    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
    #        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
    #        nn.init.zeros_(m.bias)
        
        

    def forward(self, x):
        mapping = self.map_model(x)
        return mapping
