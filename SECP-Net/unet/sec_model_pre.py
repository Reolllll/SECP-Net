# full assembly of the sub-parts to form the complete net
# pyramid network without se-module in conv block
import torch
import torch.nn.functional as F
from unet.se_module import SELayer
#from .unet_parts import *
from unet.unet_parts import *
from unet.unet_model_bf import UNet
class PREUNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(PREUNet, self).__init__()
        #model = '/home/zexi/SEConnection-UNet/checkpoints/unet96.pth'
        a = UNet(1, 14)
        #a.load_state_dict(torch.load(model)) 此处注释用于x2pre训练
        self.inc = a.inc
        self.down1 = a.down1
        self.down2 = a.down2
        self.down3 = a.down3
        self.down4 = a.down4
        self.up1 = a.up1
        self.up2 = a.up2
        self.up3 = a.up3
        self.up4 = a.up4
        self.outc = outconv(64, n_classes)
        #new pyramid network addition
        self.up45 = up(1024,512)
        self.up34 = up(512,256)
        self.up23 = up(256,128)
        self.up12 = up(128,64)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        #new pyramid network
        x45 = self.up45(x5,x4,1)
        x = self.up1(x5,x45)
        x34 = self.up34(x45,x3,1)
        x = self.up2(x,x34)
        x23 = self.up23(x34,x2,1)
        x = self.up3(x,x23)
        x12 = self.up12(x23,x1,1)
        x = self.up4(x,x12)
        x = self.outc(x)

        return x















































