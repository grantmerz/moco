import torch.nn as nn
import torch

class decoder(nn.Module):
    def __init__(self, input_dim,outshape):
        super().__init__()
        self.t_conv1 = nn.ConvTranspose2d(2, 64, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.t_conv3 = nn.ConvTranspose2d(32, 3, 3, stride=1,dilation=1,padding=1)

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.fcd1 = nn.Linear(input_dim, 2048)



    def forward(self, x):
        x = self.fcd1(x)
        x = torch.reshape(x,(-1,2,32,32))
        x = self.t_conv1(x)
        x = self.relu(x)
        x = self.t_conv2(x)
        x = self.relu(x)
        x = self.t_conv3(x)
        recon = self.sigmoid(x)
        return recon
