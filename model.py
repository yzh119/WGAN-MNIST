import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# Replace the original activation layer with SELU function.
class Generator(nn.Module):
    def __init__(self, nz, nc, ngf, isize, ngpu):
        super(Generator, self).__init__()
        net = nn.Sequential(
            nn.Linear(nz, ngf),
            nn.SELU(),
            nn.Linear(ngf, ngf),
            nn.SELU(),
            nn.Linear(ngf, ngf),
            nn.SELU(),
            nn.Linear(ngf, nc * isize * isize)
        )
        self.net = net
        self.nz = nz
        self.isize = isize
        self.nc = nc
        self.ngpu = ngpu
        self.para = torch.nn.DataParallel(self.net, device_ids=range(self.ngpu))
    
    def forward(self, input):
        input = input.view(input.size(0), input.size(1))
        output = self.para(input)
        return output.view(output.size(0), self.nc, self.isize, self.isize)

class Discriminator(nn.Module):
    def __init__(self, nz, nc, ndf, isize, ngpu):
        super(Discriminator, self).__init__()
        net = nn.Sequential(
            nn.Linear(nc * isize * isize, ndf),
            nn.SELU(),
            nn.Linear(ndf, ndf),
            nn.SELU(),
            nn.Linear(ndf, ndf),
            nn.SELU(),
            nn.Linear(ndf, 1)
        )
        self.net = net
        self.nz = nz
        self.nc = nc
        self.ndf = ndf
        self.isize = isize
        self.ngpu = ngpu
        self.para = torch.nn.DataParallel(self.net, device_ids=range(self.ngpu))
    
    def forward(self, input):
        input = input.view(input.size(0), -1)
        output = self.para(input)
        output = output.mean(0) # The reason doing so is that we only pass a batch of true samples/bad samples.
        return output