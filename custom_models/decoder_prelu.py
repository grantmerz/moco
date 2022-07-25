import torch.nn as nn
import torch
import numpy as np
class decoder(nn.Module):
    def __init__(self, input_dim, output_shape):
        super().__init__()
        '''        
        self.t_conv1 = nn.ConvTranspose2d(output_shape[0], 128, 3, stride=1,padding=1)
        self.t_conv2 = nn.ConvTranspose2d(128, 64, 3, stride=1,padding=1)
        self.t_conv3 = nn.ConvTranspose2d(64, 32, 3, stride=1,padding=1)
        self.t_conv4 = nn.ConvTranspose2d(32, 16, 3, stride=1,padding=1)
        self.t_conv5 = nn.ConvTranspose2d(16, output_shape[0], 3, stride=1,padding=1)

        
        self.relu = nn.ReLU(inplace=True)
        self.prelu = nn.PReLU()
        #self.sigmoid = nn.Sigmoid()
        self.fcd1 = nn.Linear(input_dim, np.prod(output_shape))
        #self.outputlayer = nn.Linear(
        
        #self.unflatten = nn.Unflatten(1,
        #unflattened_size=(48, 32, 32))
        self.identity=nn.Identity()
        #self.tanh = nn.Tanh()
        
        self.unflatten = nn.Unflatten(1,
        unflattened_size=output_shape)
        '''

        self.dec_model=nn.Sequential(
            nn.Linear(input_dim, np.prod(output_shape)),
            nn.PReLU(),
            nn.Unflatten(1,unflattened_size=output_shape),
            nn.ConvTranspose2d(output_shape[0], 128, 3, stride=1,padding=1),
            nn.PReLU(),
            nn.BatchNorm2d(128,momentum=0.99),
            nn.ConvTranspose2d(128, 64, 3, stride=1,padding=1),
            nn.PReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=1,padding=1),
            nn.PReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=1,padding=1),
            nn.PReLU(),
            nn.ConvTranspose2d(16, output_shape[0], 3, stride=1,padding=1),
            #nn.Identity()
            )

        self.dec_model.apply(self.weight_init)

    def weight_init(self,m):
        if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            nn.init.zeros_(m.bias)

        

    def forward(self, x):
        '''
        x = self.fcd1(x)
        x = self.prelu(x)
        x = self.unflatten(x)
        x = self.t_conv1(x)
        x = self.prelu(x)
        x = self.t_conv2(x)
        x = self.prelu(x)
        x = self.t_conv3(x)
        x = self.prelu(x)
        x = self.t_conv4(x)
        x = self.prelu(x)
        x = self.t_conv5(x)
        recon = self.identity(x)
        #recon = self.tanh(x)
        '''
        recon = self.dec_model(x)
        
        return recon
