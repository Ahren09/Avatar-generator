# coding:utf8
from torch import nn

class GLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False):
        super(GLayer, self).__init__()
        self.layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.layer(x)

class G(nn.Module):
    def __init__(self, dim_noise, dim_G):
        super(G, self).__init__()
        self.net = nn.Sequential(
            GLayer(dim_noise, dim_G*8, kernel_size=4, stride=1, padding=0),
            GLayer(dim_G*8, dim_G*4, kernel_size=4, stride=2, padding=1),
            GLayer(dim_G*4, dim_G*2, kernel_size=4, stride=2, padding=1),
            GLayer(dim_G*2, dim_G, kernel_size=4, stride=2, padding=1),

            nn.ConvTranspose2d(dim_G, 3, kernel_size=5, stride=3, padding=1),
            nn.Tanh()
        )

    
    def forward(self, x):
        return self.net(x)

class DLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False):
        super(DLayer, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    
    def forward(self, x):
        return self.layer(x)


class D(nn.Module):
    def __init__(self, dim_D):
        super(D, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, dim_D, kernel_size=5, stride=3, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            DLayer(dim_D, dim_D*2, kernel_size=4, stride=2, padding=1),
            DLayer(dim_D*2, dim_D*4, kernel_size=4, stride=2, padding=1),
            DLayer(dim_D*4, dim_D*8, kernel_size=4, stride=2, padding=1),

            nn.Conv2d(dim_D*8, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid() # Output a probability
        )

    
    def forward(self, x):
        x = self.net(x)
        x = x.view(-1)
        return x



