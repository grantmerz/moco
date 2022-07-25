import torch.nn as nn

class encoder(nn.Module):
    def __init__(self,num_channels,num_classes):
        super().__init__()
        self.enc_model=nn.Sequential(
            nn.Conv2d(num_channels, 128, 3, stride=1,padding=1),
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.Conv2d(128, 256, 3, stride=2,padding=1),
            nn.BatchNorm2d(256),
            nn.PReLU(),
            nn.Conv2d(256, 512, 3, stride=1,padding=1),
            nn.BatchNorm2d(512),
            nn.PReLU(),
            nn.Conv2d(512, 1024, 3, stride=2,padding=1),
            nn.BatchNorm2d(1024),
            nn.PReLU(),
            nn.Conv2d(1024, 512, 3, stride=1,padding=1),
            nn.BatchNorm2d(512),
            nn.PReLU(),
            nn.Conv2d(512, 256, 3, stride=1,padding=1),
            nn.BatchNorm2d(256),
            nn.PReLU(),
            nn.Conv2d(256, 128, 3, stride=1,padding=1),
            nn.BatchNorm2d(128),
            nn.PReLU(),
            #nn.Conv2d(128, 128, 3, stride=1,padding=1),
            #nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128*32*32,num_classes),
            nn.PReLU()
            #nn.ReLU()
            )

        self.enc_model.apply(self.weight_init)

    def weight_init(self,m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            nn.init.zeros_(m.bias)

    def forward(self, x):

        enc = self.enc_model(x)
        
        return enc
