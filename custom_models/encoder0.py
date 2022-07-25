import torch.nn as nn

class encoder(nn.Module):
    def __init__(self,num_channels,num_classes):
        super().__init__()
        self.enc_model=nn.Sequential(
            nn.Conv2d(num_channels, 128, 3, stride=2,padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, stride=2,padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, 3, stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 5, 3, stride=1,padding=1),
            nn.Flatten(),
            nn.Linear(5*32*32, num_classes),
            nn.ReLU()
            )

    def forward(self, x):

        enc = self.enc_model(x)
        
        return enc
