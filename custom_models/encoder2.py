import torch.nn as nn

class encoder(nn.Module):
    def __init__(self,num_channels,num_classes):
        super().__init__()
        self.enc_model=nn.Sequential(
            nn.Conv2d(num_channels, 8, 3, stride=1,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(8,momentum=0.99),
            nn.Conv2d(8, 16, 3, stride=2,padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2,padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=1,padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128*32*32, 1024),
            nn.ReLU(),
            nn.Linear(1024,512),
            nn.ReLU(),
            nn.Linear(512,num_classes),
            nn.ReLU()
            )

    def forward(self, x):

        enc = self.enc_model(x)
        
        return enc
