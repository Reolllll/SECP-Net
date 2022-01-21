# full assembly of the sub-parts to form the complete net

import torch
import torch.nn.functional as F

#from .unet_parts import *
from unet.unet_set_parts import *
class SUBUNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(SUBUNet, self).__init__()
        self.inc2 = inconv(n_channels, 64)
        self.down12 = down(64, 128)
        self.down22 = down(128, 256)
        self.down32 = down(256, 512)
        self.down42 = down(512, 512)
        self.up12 = up(1024, 256)
        self.up22 = up(512, 128)
        self.up32 = up(256, 64)
        self.up42 = up(128, 64)
        self.outc2 = outconv(64, n_classes)

    def forward(self, x):
        x12 = self.inc2(x)
        x22 = self.down12(x12)
        x32 = self.down22(x22)
        x42 = self.down32(x32)
        x52 = self.down42(x42)
        x = self.up12(x52, x42)
        x = self.up22(x, x32)
        x = self.up32(x, x22)
        x = self.up42(x, x12)
        x = self.outc2(x)
        return x

# if __name__ == '__main__':
#
#     model=UNet(1,14)
#     model.cuda()
#     x = torch.rand(12,1,256,256)
#     x = x.cuda()
#     model.eval()
#     y=model(x)
#     print(y.shape)
