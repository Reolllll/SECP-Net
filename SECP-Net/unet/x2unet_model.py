# full assembly of the sub-parts to form the complete net

import torch
import torch.nn.functional as F

#from .unet_parts import *
from unet.unet_parts import *
#from unet.unet_model import UNet
#from unet.sec_model import UNet
from unet.subunet_model import SUBUNet
from collections import OrderedDict

class X2UNet(nn.Module):
    def __init__(self, n_channels, n_classes,model):
        super(X2UNet, self).__init__()
        self.UNet1 = SUBUNet(n_channels,n_classes)
        #for p in self.parameters():
        #    p.requires_grad = False
        if model == 1:
            self.init_ = torch.load('/home/zexi/SEConnection-UNet/checkpoints/unet96.pth')
            #self.init_ = torch.load('/home/zexi/SEConnection-UNet/checkpoints/seconnectionwin97.pth')
        else:
            self.init_ = torch.load('/home/zexi/SEConnection-UNet/checkpoints/seconnectionret100.pth')
        #self.new_state_dict = OrderedDict()
        
        #for k, v in self.init_.items():
        #    # print(type(k))
        #    name = k[8:]
        #    self.new_state_dict[name] = v
        #self.UNet1.load_state_dict(self.new_state_dict)
        self.UNet1.load_state_dict(self.init_)
        for p in self.parameters():
            p.requires_grad = False
        self.UNet2 = SUBUNet(n_channels + n_classes,n_classes)   #auto-context
        #self.UNet2 = SUBUNet(n_classes,n_classes)    #net concat
    def forward(self, x):
        u1 = self.UNet1(x)
        u1p = F.softmax(u1,dim=1)
        #print(u1.shape)
        xin = torch.cat([u1p,x],dim=1) #auto-context
        #xin = u1p  #net concat
        u2 = self.UNet2(xin)
        return u2

#if __name__ == '__main__':
#
#    model=X2UNet(1,14)
#    model.cuda()
#    x = torch.rand(10,1,256,256)
#    x = x.cuda()
#    model.eval()
#    y=model(x)
#    print(y.shape)
