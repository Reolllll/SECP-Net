import sys
import os
from optparse import OptionParser
import numpy as np
from tensorboardX import SummaryWriter
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import copy
from torch import optim
from torch.autograd import Variable
from eval1 import eval_net
from utils import get_ids, split_ids, split_train_val, get_imgs_and_masks, batch, five_fold_cross_validation, ten_fold_cross_validation
import time
import random
#from unet.x2unet_model import X2UNet
#from unet.x2unet_use_model import X2UNet #new network
#from unetpp import NestedUNet
#from BaseNet import CPFNet
from unet.sec_model import UNet
#from unet.subunet_model import SUBUNet
def train_net(net,
              epochs=10,
              epochsk=10,
              batch_size=12,
              lr=0.01,
              val_percent=0.2,
              save_cp=True,
              gpu=True,
              img_scale=0.5):

    #dir_img = 'data/traingraynoxyf/'
    #dir_mask = 'data/train_masks_noxyf/'
    dir_img = 'data/trainnrrd/'
    dir_mask = 'data/train_masksnrrd/'

    dir_checkpoint = 'checkpoints/nrrdsec/'
    writer = SummaryWriter('Tensorboard/nrrdsec/') #need to change tensorboard name
    ids = get_ids(dir_img)

    #print(len(ids))
    ids = split_ids(ids)
    #print(len(ids))
    timetotal = 0
    iddataset, data_len = split_train_val(ids, val_percent)
    print(iddataset['val'])
    valsum = 0
    #print(int(train_len))
    print('''
    Starting training:
        Epochs: {}

        Batch size: {}
        Learning rate: {}
        Training size: {}
        Validation size: {}
        Checkpoints: {}
        CUDA: {}
    '''.format(epochs*epochsk, batch_size, lr, int(0.8*data_len),
               int(0.2*(data_len)), str(save_cp), str(gpu)))

    N_train = int(0.8*data_len)

    # optimizer = optim.Adam(net.parameters(),
    #                       lr=lr,
    #                       )


    criterion = nn.CrossEntropyLoss()
    #print(iddataset['train'],iddataset['val'])
    #lrd = lr
    DD = []
    #mload = 'checkpoints/unet96.pth'
    #mload = 'checkpoints/2dnoxyfCP{}'.format(num*10 + epoch + 1) +'.pth'
    #net.load_state_dict(torch.load(mload))
    #print('Model loaded from {}'.format(mload))
    for num in range(epochsk):  #epochsk为5
        #train = get_imgs_and_masks(iddataset['train'], dir_img, dir_mask, img_scale)
        
        for epoch in range(epochs):  #epoch为10
            #optimizer = optim.SGD(filter(lambda p:p.requires_grad,net.parameters()),
            #                      lr=lr*((1-(num*10+epoch+1)/(epochsk*epochs))**0.9),momentum=0.9
            #                       )
            
            # print(lr*((1-(num*10+epoch+1)/(epochsk*epochs))**0.9))

            #mload = 'checkpoints/unet96.pth'
            #mload = 'checkpoints/2dnoxyfCP{}'.format(num*10 + epoch + 1) +'.pth'
            #net.load_state_dict(torch.load(mload))
            #print('Model loaded from {}'.format(mload))

            optimizer = optim.Adam(net.parameters(),
                                   lr=lr * ((1 - (num * 10 + epoch + 1) / (epochsk * epochs)) ** 0.9)
                                   )

            traincopy = copy.deepcopy(iddataset['train'])
            random.shuffle(traincopy)
            #print(traincopy == CC)
            #DD = CC
            print('Starting epoch {}/{}.'.format(epoch + 1 + num * 10, epochs * epochsk))

            net.train()
            epoch_loss = 0
            img_n = 0
            val = get_imgs_and_masks(iddataset['val'], dir_img, dir_mask, img_scale)
            train = get_imgs_and_masks(traincopy,dir_img, dir_mask, img_scale)
            #print(train)
            start1 = time.time()

            for i, b in enumerate(batch(train, batch_size)):
                #1==1
                imgs = np.array([i[0] for i in b]).astype(np.float32)
                true_masks = np.array([i[1] for i in b])
               
                imgs = torch.from_numpy(imgs)
                imgs = imgs.unsqueeze(1)
                true_masks = torch.from_numpy(true_masks)

                # if gpu:
                #     imgs = imgs.cuda()
                #     true_masks = true_masks.cuda()

                imgs = imgs.cuda()
                masks_pred = net(imgs)
                input = masks_pred.cuda()



                true_masks = true_masks.long()
                target = true_masks.cuda()


                criterion.cuda()
                loss = criterion(input, target)
                epoch_loss += loss.item()
                img_n = img_n + 1
                #print('{0:.4f} --- batchloss: {1:.6f}'.format((img_n) * batch_size / N_train, loss.item()))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            end1 = time.time()
            print('time used:', end1 - start1)
            timetotal = timetotal + (end1 - start1)
            print('Epoch finished ! Loss: {}'.format(epoch_loss / N_train))

            file_handle = open('nrrdsec1.txt', mode='a+')    #need to change txt name
            file_handle.write('\n')
            file_handle.write('Loss of Epoch' + str(epoch  + 1 + num * 10) + ' is: {} \n'.format(epoch_loss / N_train))
            file_handle.write('Epoch time used' + str(epoch  + 1 + num * 10) + ' is: {} \n'.format(end1 - start1))

            if 1:
                val_dice,val_jac = eval_net(net, val, 0.2*data_len, batch_size, gpu)
                vals = torch.sum(val_dice)
                if valsum<vals:
                    valsum = vals
                    maxepoch = epoch + 1 +num*10
                    if save_cp:
                        torch.save(net.state_dict(),
                                   dir_checkpoint + 'nrrdsec1best.pth')
                        print('Checkpoint {} saved !'.format(epoch + 1 + num * 10))

                for op in range(13):
                    print('Dice Coeff of OAR' + str(op+1)+ ': {}'.format(val_dice[op+1]))
                    print('Jaccard of OAR' + str(op+1)+ ': {}'.format(val_jac[op+1]))
                    writer.add_scalar('DiceOfOrgan' + str(op + 1), val_dice[op + 1], epoch + 1 + num * 10)

                file_handle.write('TemporalLobe_L dice:{} '.format(val_dice[1]) + 'Jaccard:{}'.format(val_jac[1])+'\n')
                file_handle.write('TemporalLobe_R dice:{} '.format(val_dice[2]) + 'Jaccard:{}'.format(val_jac[2])+'\n')
                file_handle.write('Eye_L dice:{} '.format(val_dice[3]) + 'Jaccard:{}'.format(val_jac[3])+'\n')
                file_handle.write('Eye_R dice:{} '.format(val_dice[4]) + 'Jaccard:{}'.format(val_jac[4])+'\n')
                file_handle.write('Mandible_L dice:{} '.format(val_dice[5]) + 'Jaccard:{}'.format(val_jac[5])+'\n')
                file_handle.write('Mandible_R dice:{} '.format(val_dice[6]) + 'Jaccard:{}'.format(val_jac[6])+'\n')
                file_handle.write('Brainstem dice:{} '.format(val_dice[7]) + 'Jaccard:{}'.format(val_jac[7])+'\n')
                file_handle.write('Parotid_L dice:{} '.format(val_dice[8]) + 'Jaccard:{}'.format(val_jac[8])+'\n')
                file_handle.write('Parotid_R dice:{} '.format(val_dice[9]) + 'Jaccard:{}'.format(val_jac[9])+'\n')
                file_handle.write('Spinal Cord dice:{} '.format(val_dice[10]) + 'Jaccard:{}'.format(val_jac[10])+'\n')
                file_handle.write('Submandibular_L dice:{} '.format(val_dice[11]) + 'Jaccard:{}'.format(val_jac[11])+'\n')
                file_handle.write('Submandibular_R dice:{} '.format(val_dice[12]) + 'Jaccard:{}'.format(val_jac[12])+'\n')
                file_handle.write('Thyroid dice:{}'.format(val_dice[13]) + 'Jaccard:{}'.format(val_jac[13])+'\n')
                file_handle.write('\n')
                if epoch + 1 + num*10 == 100:
                    file_handle.write('valmax:{}'.format(valsum) + 'maxepoch:{}'.format(maxepoch))
                file_handle.close()
            #if save_cp and (epoch + 1 + num*10)>90:
            #    torch.save(net.state_dict(),
            #               dir_checkpoint + 'UNet++{}.pth'.format(epoch + 1 + num * 10))    #need to change pth name
            #    print('Checkpoint {} saved !'.format(epoch + 1 + num * 10))
    print(timetotal)

    #file_handle.write('Total time:{}'.format(timetotal))
    #file_handle.close()



def get_args():
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=10, type='int',
                      help='number of epochs')
    parser.add_option('-b', '--batch-size', dest='batchsize', default=16,
                      type='int', help='batch size')
    parser.add_option('-l', '--learning-rate', dest='lr', default=0.01,
                      type='float', help='learning rate')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu',
                      default=True, help='use cuda')
    parser.add_option('-c', '--load', dest='load',
                      default=False, help='load file model')  
    parser.add_option('-s', '--scale', dest='scale', type='float',
                      default=0.5, help='downscaling estfactor of the images')  

    (options, args) = parser.parse_args()
    return options

if __name__ == '__main__':
    args = get_args()
    os.environ['CUDA_VISIBLE_DEVICES']='1'
    #net = X2UNet(1,14,2)
    #net = SUBUNet(1,7)
    net = UNet(1,7)
    #net = CPFNet(14)
    if args.load:
        net.load_state_dict(torch.load(args.load))
        print('Model loaded from {}'.format(args.load))

    if args.gpu:
        net.cuda()
        cudnn.benchmark = True # faster convolutions, but more memory

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  gpu=args.gpu,
                  img_scale=args.scale)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
