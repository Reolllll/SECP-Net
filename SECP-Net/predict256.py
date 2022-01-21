import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image

from unet.unet_model_bf import UNet
#from unet import UNet
from unetpp import NestedUNet
from AttentionUNet.attentionUnet import AttU_Net
from CENet.cenet import CE_Net_
from BaseNet import CPFNet
from unet.x2unet_use_model import X2UNet
from utils import resize_and_crop, normalize, split_img_into_squares, hwc_to_chw, merge_masks #, dense_crf
from utils import plot_img_and_mask

from torchvision import transforms

def predict_img(net,
                full_img,
                # true_mask,
                scale_factor=0.5,
                out_threshold=0.5,
                use_dense_crf=True,
                use_gpu=True):

    net.eval()
    img_height = full_img.size[1]
    img_width = full_img.size[0]

    img = resize_and_crop(full_img, scale=scale_factor)
    img = normalize(img)

    img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)

    
    if use_gpu:

        img = img.cuda()

    with torch.no_grad():

        output = net(img)
        pre = torch.argmax(output,dim=1)


        out_prob = pre.squeeze(0).squeeze(0).cpu().numpy()
        full_mask = out_prob
        #out_prob = out_prob / 255
        #out_prob = torch.from_numpy(out_prob)
        #tf = transforms.Compose(
        #    [
        #        transforms.ToPILImage(mode=None),
        #        transforms.Resize(img_height),
        #        transforms.ToTensor()
        #    ]
        #)
        #out_prob = tf(out_prob.float().cpu())
        #out_prob1 = out_prob.squeeze().cpu().numpy()

        #full_mask = out_prob1 * 255



    # full_mask = pre.suqeeze()

    return full_mask
    #return full_mask > out_threshold



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', default='checkpoints/x2_pretrained_newpyramid_noseinconvblock/x2_pretrained_newpyramid_noseinconvblock_best.pth', 
                        metavar='FILE',
                        help="Specify the file in which is stored the model"
                             " (default : 'MODEL.pth')")
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+',
                        help='filenames of input images', default= 'nothing',
                        )

    parser.add_argument('--output', '-o', metavar='INPUT', nargs='+',
                        help='filenames of ouput images')
    # parser.add_argument('--output', '-o', metavar='INPUT', nargs='+',
    #                     help='filenames of ouput images', default='predict/00C1190248_14.png')
    parser.add_argument('--cpu', '-c', action='store_true',
                        help="Do not use the cuda version of the net",
                        default=False)
    parser.add_argument('--viz', '-v', action='store_true',
                        help="Visualize the images as they are processed",
                        default=False)
    parser.add_argument('--no-save', '-n', action='store_true',
                        help="Do not save the output masks",
                        default=False)
    parser.add_argument('--no-crf', '-r', action='store_true',
                        help="Do not use dense CRF postprocessing",
                        default=False)
    parser.add_argument('--mask-threshold', '-t', type=float,
                        help="Minimum probability value to consider a mask pixel white",
                        default=0.5)
    parser.add_argument('--scale', '-s', type=float,
                        help="Scale factor for the input images",
                        default=0.5)

    return parser.parse_args()

def get_output_filenames(args):

    #dir = 'data/test/train'
    dir = 'data/testmore/train/'
    in_files = os.listdir(dir)
    # in_files = in_files.split(',')
    out_files = []

    if not args.output:
        for f in in_files:
            pathsplit = os.path.splitext(f)
            out_files.append("{}_OUT{}".format(pathsplit[0], pathsplit[1]))
    elif len(in_files) != len(args.output):
        print("Error : Input files and output files are not of the same length")
        raise SystemExit()
    else:
        out_files = args.output

    return out_files

def mask_to_image(mask):
# 50修改到了15
    # print(mask.max())
    return Image.fromarray((mask * 15).astype(np.uint8))

if __name__ == "__main__":
    args = get_args()
    dir = 'data/testmore/train'
    #dir = 'papertest/'
    in_files = os.listdir(dir)

    # in_files = in_files.split(',')
    out_files = get_output_filenames(args)

    #net = AttU_Net()                               #change
    #net = UNet(1,14)
    #net = NestedUNet(1,14)
    #net = CE_Net_()
    #net = CPFNet(14)
    net = X2UNet(1,14)
    print("Loading model {}".format(args.model))

    if not args.cpu:
        print("Using CUDA version of the net, prepare your GPU !")
        net.cuda()
        net.load_state_dict(torch.load(args.model))
    else:
        net.cpu()
        net.load_state_dict(torch.load(args.model, map_location='cpu'))
        print("Using CPU version of the net, this may be very slow")

    print("Model loaded !")

    for i, fn in enumerate(in_files):
        print("\nPredicting image {} ...".format(fn))
        os.chdir('/home/zexi/SEConnection-UNet/data/testmore/train')
        #os.chdir('/home/hzx/下载/project1/SE-UNet/data/test/train')
        img = Image.open(fn)

        if img.size[0] < img.size[1]:
            print("Error: image height larger than the width")


        # mask = Image.open("dataset/00C1190248_17_mask.png")

        mask = predict_img(net=net,
                           full_img=img,
                           # true_mask=mask,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           use_dense_crf=not args.no_crf,
                           use_gpu=not args.cpu)
        if args.viz:
            print("Visualizing results for image {}, close to continue ...".format(fn))
            plot_img_and_mask(img, mask)
        os.chdir('/home/zexi/SEConnection-UNet/predictall/SCPNetTest')        #change
        #os.chdir('/home/hzx/下载/project1/SE-UNet/predict/')
        if not args.no_save:
            out_fn = out_files[i]
            result = mask_to_image(mask)
            result.save(out_files[i])

            print("Mask saved to {}".format(out_files[i]))
