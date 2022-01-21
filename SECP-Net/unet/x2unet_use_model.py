# full assembly of the sub-parts to form the complete net

import torch
import torch.nn.functional as F

#from .unet_parts import *
from unet.unet_parts import *
#from unet.unet_model import UNet
from unet.sec_model_pre import PREUNet
from unet.subunet_model import SUBUNet
from collections import OrderedDict

class X2UNet(nn.Module):
    def __init__(self, n_channels, n_classes):                #原来的class参数里最后有个model
    
        super(X2UNet, self).__init__()
        self.UNet1 = PREUNet(n_channels,n_classes)
        
        #if model == 1:
        #    self.init_ = torch.load('/home/zexi/SEConnection-UNet/checkpoints/pretrained_newpyramid_noseinconvblock/pretrained_newpyramid_noseinconvblock_best.pth')
        #else:
        #    self.init_ = torch.load('/home/zexi/SEConnection-UNet/checkpoints/unet96.pth')
        #
        #self.UNet1.load_state_dict(self.init_)
        #for p in self.parameters():
        #    p.requires_grad = False
        #self.UNet2 = UNet(n_channels,n_classes)
        self.UNet2 = SUBUNet(n_channels+n_classes,n_classes)   #auto-context
        #self.UNet2 = SUBUNet(1,n_classes)    #net concat
    def forward(self, x):
        u1 = self.UNet1(x)
        #u1p = u1.max(1)[1].float()
        #u1p = u1p.unsqueeze(1)
        u1p = F.softmax(u1,dim=1)
        #xin = x
        xin = torch.cat([u1p,x],dim=1) #auto-context
        #xin = u1p  #net concat
        u2 = self.UNet2(xin)
        return u2
        #return u1p

#if __name__ == '__main__':
#
#    model=X2UNet(1,14)
#    model.cuda()
#    x = torch.rand(10,1,256,256)
#    x = x.cuda()
#    model.eval()
#    y=model(x)
#    print(y.shape)
