# full assembly of the sub-parts to form the complete net

import torch
import torch.nn.functional as F
from unet.se_module import SELayer
#from .unet_parts import *
from unet.unet_parts import *
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.se1 = SELayer(64)
        self.down1 = down(64, 128)
        self.se2 = SELayer(128)
        self.down2 = down(128, 256)
        self.se3 = SELayer(256)
        self.down3 = down(256, 512)
        self.se4 = SELayer(512)
        self.down4 = down(512, 512)
        self.se5 = SELayer(512)
        self.up1 = up(1024, 256)
        self.se6 = SELayer(256)
        self.up2 = up(512, 128)
        self.se7 = SELayer(128)
        self.up3 = up(256, 64)
        self.se8 = SELayer(64)
        self.up4 = up(128, 64)
        self.se9 = SELayer(64)
        self.outc = outconv(64, n_classes)
        #new pyramid networkd addition
        self.up45 = up(1024,512)
        self.up34 = up(512,256)
        self.up23 = up(256,128)
        self.up12 = up(128,64)

    def forward(self, x):
        x1 = self.inc(x)
        x1 = self.se1(x1)
        x2 = self.down1(x1)
        x2 = self.se2(x2)
        x3 = self.down2(x2)
        x3 = self.se3(x3)
        x4 = self.down3(x3)
        x4 = self.se4(x4)
        x5 = self.down4(x4)
        x5 = self.se5(x5)
        #new pyramid network
        x45 = self.up45(x5,x4,1)
        x = self.up1(x5,x45,0)
        x = self.se6(x)
        x34 = self.up34(x45,x3,1)
        x = self.up2(x,x34,0)
        x = self.se7(x)
        x23 = self.up23(x34,x2,1)
        x = self.up3(x,x23,0)
        x = self.se8(x)
        x12 = self.up12(x23,x1,1)
        x = self.up4(x,x12,0)
        x = self.se9(x)
        x = self.outc(x)
        #old network
        #x = self.up1(x5, x4,0)
        #x = self.se6(x)
        #x = self.up2(x, x3,0)
        #x = self.se7(x)
        #x = self.up3(x, x2,0)
        #x = self.se8(x)
        #x = self.up4(x, x1,0)
        #x = self.se9(x)
        #x = self.outc(x)
        return x

