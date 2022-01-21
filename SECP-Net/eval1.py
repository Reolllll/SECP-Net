import torch
import torch.nn.functional as F
from dice_loss import MulticlassDiceLoss
import numpy as np
from utils import batch
import time
#from dice_loss import dice_coeff


def eval_net(net, dataset, slicetotal, batch_size=12, gpu=True):
    """Evaluation without the densecrf with the dice coefficient"""

    net.eval()
    start = time.time()
    dice_ = torch.zeros(14).cuda()
    jac_ = torch.zeros(14).cuda()
    NE = torch.zeros(14).cuda()
    JNE = torch.zeros(14).cuda()
    ia = 0
    #dice_all = torch.zeros(14).cuda()
    #NE = torch.Tensor([0,3241,3241,3611,3611,2911,2911,3201,3161,3161,1110,3530,3530,3281]).cuda()
    print(1)
    with torch.no_grad():
        for i, b in enumerate(batch(dataset, batch_size)):
            #print('eval',i)


            imgs = np.array([k[0] for k in b]).astype(np.float32)
            true_masks = np.array([k[1] for k in b])

            imgs = torch.from_numpy(imgs)
            imgs = imgs.unsqueeze(1)
            true_masks = torch.from_numpy(true_masks)
            #imgs = imgs.permute(1, 2, 0)
            #imgs = imgs.unsqueeze(0).unsqueeze(0)
            pre_masks_eval = torch.zeros(true_masks.shape[0],14,256,256)
            true_masks_eval = torch.zeros(true_masks.shape[0],14,256,256)
            batchshape = true_masks.shape[0]

            batch_dice = torch.zeros(14).cuda()
            if gpu:
                imgs = imgs.cuda()
                true_masks = true_masks.cuda()
                net.cuda()

            output_img = net(imgs)
            #pre_masks = output_img.cuda()
            input = output_img.cuda()
            #input = input.squeeze(0).permute(3, 0, 1, 2) 
            #pre_masks = F.softmax(pre_masks,dim=1)
            pre_masks = input.max(1)[1].float() 
            for ak in range(14):
                if ak == 0:
                    continue
                pre_masks_eval[:,ak] = (pre_masks==ak)
                true_masks_eval[:,ak] = (true_masks==ak)
                premasks = pre_masks_eval[:,ak].view(true_masks.shape[0],-1)
                truemasks = true_masks_eval[:,ak].view(true_masks.shape[0],-1)

                intersection = premasks * truemasks
                TP = intersection.sum(1)
                FP = premasks.sum(1) - TP
                FN = truemasks.sum(1) - TP

                for bk in range(true_masks.shape[0]):
                    if TP[bk] == 0 and FP[bk] == 0 and FN[bk] == 0:
                        NE[ak] += 1
                        JNE[ak] += 1
                    else:
                        batch_dice[ak] = batch_dice[ak] + 2*TP[bk] / (2*TP[bk] + FP[bk] + FN[bk])
                        jac_[ak] = jac_[ak] + TP[bk] / (TP[bk] + FP[bk] + FN[bk])
                #batch_dice[ak] = (2*TP.float() / (2*TP.float() + FP.float() + FN.float() + 0.01)).sum()
                #if ak>6:     
                #    print(batch_dice[ak])

            dice_ = dice_ + batch_dice
            #ia += 1
        #print(i*12)
        for knum in range(14):
            dice_[knum] = dice_[knum] / (slicetotal - NE[knum])
            jac_[knum] = jac_[knum] / (slicetotal - JNE[knum])
            #print(NE[knum])
    end = time.time()
    print('time used:',end - start)
    #print(dice_all)
    #return dice_all
    return dice_,jac_

    # net.eval()
    # tot = 0
    # for i, b in enumerate(dataset):
    #     img = b[0]
    #     true_mask = b[1]
    #
    #
    #     img = torch.from_numpy(img).unsqueeze(0)
    #     true_mask = torch.from_numpy(true_mask).unsqueeze(0)
    #
    #     if gpu:
    #         img = img.cuda()
    #         true_mask = true_mask.cuda()
    #
    #     mask_pred = net(img)
    #     c = mask_pred.shape[1]
        # for j in range(c):




        # mask_pred = net(img)[0]
        # mask_pred = (mask_pred > 0.5).float()


        # tot += MulticlassDiceLoss().forward(mask_pred,true_mask).item()

        # tot += dice_coeff(mask_pred, true_mask).item()  #计算相似度
    # return tot / (i+1)
