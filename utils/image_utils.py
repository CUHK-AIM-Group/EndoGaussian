#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

@torch.no_grad()
def psnr(img1, img2, mask=None):
    if mask is None:
        mse_mask = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    else:
        # img1 = img1 * mask
        # img2 = img2 * mask
        # mse = (((img1-img2))**2).view(img1.shape[0], -1).mean(1, keepdim=True)
        if mask.shape[1] == 3:
            mse_mask = (((img1-img2)**2)*mask).sum() / ((mask.sum()+1e-10))
        else:
            mse_mask = (((img1-img2)**2)*mask).sum() / ((mask.sum()+1e-10)*3.0)

    return 20 * torch.log10(1.0 / torch.sqrt(mse_mask))
