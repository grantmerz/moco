import torch.nn as nn
import torch
import numpy as np
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
            nn.Linear(input_dim,np.prod(output_shape)),
            nn.PReLU(),
            nn.Unflatten(1,unflattened_size=output_shape),
            nn.ConvTranspose2d(128, 64, 3, stride=1,padding=1),
            nn.PReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=1,padding=1),
            nn.PReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=1,padding=1,output_padding=1),
            nn.PReLU(),
            nn.ConvTranspose2d(16, 8, 3, stride=1,padding=1,output_padding=1),
            nn.PReLU(),
            nn.ConvTranspose2d(8, output_shape[0], 3, stride=1,padding=1),
            
            )

    def forward(self, x):
        recon = self.dec_model(x)
        return recon
