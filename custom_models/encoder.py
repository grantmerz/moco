import torch.nn as nn

class encoder(nn.Module):
    def __init__(self,num_channels,num_classes):
        super().__init__()
        #self.t_conv1 = nn.ConvTranspose2d(48, 24, 4, stride=2,padding=1)
        #self.t_conv2 = nn.ConvTranspose2d(24, 8, 4, stride=2,padding=1)
        #self.t_conv3 = nn.Conv2d(8, 3, 3, stride=1,padding=1)
        
        self.t_conv1 = nn.Conv2d(3, 128, 3, stride=2,padding=1)
        self.t_conv2 = nn.Conv2d(128, 64, 3, stride=2,padding=1)
        self.t_conv3 = nn.Conv2d(64, 32, 3, stride=1,padding=1)
        self.t_conv4 = nn.Conv2d(32, 16, 3, stride=1,padding=1)
        self.t_conv5 = nn.Conv2d(16, 3, 3, stride=1,padding=1)

        
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.fcd1 = nn.Linear(3*32*32, num_classes)
        
        self.flatten = nn.Flatten()

        self.unflatten = nn.Unflatten(1,
        unflattened_size=(3, 128, 128))

    def forward(self, x):
        #x = self.fcd1(x)
        #x = self.unflatten(x)
        x = self.t_conv1(x)
        x = self.relu(x)
        x = self.t_conv2(x)
        x = self.relu(x)
        x = self.t_conv3(x)
        x = self.relu(x)
        x = self.t_conv4(x)
        x = self.relu(x)
        x = self.t_conv5(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.fcd1(x)
        enc = self.relu(x)
        
        return enc
