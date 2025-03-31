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

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
# from lpipsPyTorch import lpips
import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from utils.image_utils import rmse
from utils.image_utils import flip

from argparse import ArgumentParser
import numpy as np

to8b = lambda x : (255*x).to(torch.uint8)

def array2tensor(array, device="cuda", dtype=torch.float32):
    return torch.tensor(array, dtype=dtype, device=device)

# Learned Perceptual Image Patch Similarity
class LPIPS(object):
    """
    borrowed from https://github.com/huster-wgm/Pytorch-metrics/blob/master/metrics.py
    """
    def __init__(self, device="cuda"):
        self.model = lpips.LPIPS(net='alex').to(device)

    def __call__(self, y_pred, y_true, normalized=True):
        """
        args:
            y_true : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
            y_pred : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
            normalized : change [0,1] => [-1,1] (default by LPIPS)
        return LPIPS, smaller the better
        """
        if normalized:
            y_pred = y_pred * 2.0 - 1.0
            y_true = y_true * 2.0 - 1.0
        error =  self.model.forward(y_pred, y_true)
        return torch.mean(error)
    
lpips = LPIPS()
def cal_lpips(a, b, device="cuda", batch=2):
    """Compute lpips.
    a, b: [batch, H, W, 3]"""
    if not torch.is_tensor(a):
        a = array2tensor(a, device)
    if not torch.is_tensor(b):
        b = array2tensor(b, device)

    lpips_all = []
    for a_split, b_split in zip(a.split(split_size=batch, dim=0), b.split(split_size=batch, dim=0)):
        out = lpips(a_split, b_split)
        lpips_all.append(out)
    lpips_all = torch.stack(lpips_all)
    lpips_mean = lpips_all.mean()
    return lpips_mean

def readImages(renders_dir, gt_dir, depth_dir, gtdepth_dir, masks_dir):
    renders = []
    gts = []
    image_names = []
    depths = []
    gt_depths = []
    masks = []
    
    for fname in os.listdir(renders_dir):
        render = np.array(Image.open(renders_dir / fname))
        gt = np.array(Image.open(gt_dir / fname))
        depth = np.array(Image.open(depth_dir / fname))
        gt_depth = np.array(Image.open(gtdepth_dir / fname))
        mask = np.array(Image.open(masks_dir / fname))
        
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        depths.append(torch.from_numpy(depth).unsqueeze(0).unsqueeze(1)[:, :, :, :].cuda())
        gt_depths.append(torch.from_numpy(gt_depth).unsqueeze(0).unsqueeze(1)[:, :3, :, :].cuda())
        masks.append(tf.to_tensor(mask).unsqueeze(0)[:, 0:1, :, :].cuda())
        
        image_names.append(fname)
    return renders, gts, depths, gt_depths, masks, image_names

def evaluate(model_paths):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")
    
    with torch.no_grad():
        for scene_dir in model_paths:
            print("Scene:", scene_dir)
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}
            full_dict_polytopeonly[scene_dir] = {}
            per_view_dict_polytopeonly[scene_dir] = {}

            test_dir = Path(scene_dir) / "test"

            for method in os.listdir(test_dir):
                print("Method:", method)

                full_dict[scene_dir][method] = {}
                per_view_dict[scene_dir][method] = {}
                full_dict_polytopeonly[scene_dir][method] = {}
                per_view_dict_polytopeonly[scene_dir][method] = {}

                method_dir = test_dir / method
                gt_dir = method_dir/ "gt"
                renders_dir = method_dir / "renders"
                depth_dir = method_dir / "depth"
                gt_depth_dir = method_dir / "gt_depth"
                masks_dir = method_dir / "masks"
                
                renders, gts, depths, gt_depths, masks, image_names = readImages(renders_dir, gt_dir, depth_dir, gt_depth_dir, masks_dir)

                ssims = []
                psnrs = []
                psnrs_star = []
                lpipss = []
                rmses = []
                        
                render_wmask = []
                gt_wmask = []     
                for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                    render, gt, depth, gt_depth, mask = renders[idx], gts[idx], depths[idx], gt_depths[idx], masks[idx]
                    
                    psnrs_star.append(psnr(render, gt, mask))
                    
                    render = render * mask
                    gt = gt * mask
                    render_wmask.append(render)
                    gt_wmask.append(gt)
                    psnrs.append(psnr(render, gt))
                    ssims.append(ssim(render, gt))
                    lpipss.append(cal_lpips(render, gt))
                    
                    if (gt_depth!=0).sum() < 10:
                        continue
                    
                    tmp_mask = gt_depth != 0
                    depth_mask = torch.logical_and(tmp_mask, mask)
                    depth = depth * depth_mask
                    gt_depth = gt_depth * depth_mask
                    rmses.append(rmse(depth, gt_depth))
                
                flipped_metrics = flip([to8b(e) for e in render_wmask], [to8b(g) for g in gt_wmask], interval=10)

                print("Scene: ", scene_dir,  "SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
                print("Scene: ", scene_dir,  "PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
                print("Scene: ", scene_dir,  "PSNR* : {:>12.7f}".format(torch.tensor(psnrs_star).mean(), ".5"))
                print("Scene: ", scene_dir,  "LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
                print("Scene: ", scene_dir,  "FLIP: {:>12.7f}".format(torch.tensor(flipped_metrics).mean(), ".5"))
                print("Scene: ", scene_dir,  "RMSE: {:>12.7f}".format(torch.tensor(rmses).mean(), ".5"))

                full_dict[scene_dir][method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                                        "PSNR": torch.tensor(psnrs).mean().item(),
                                                        "PSNR*": torch.tensor(psnrs_star).mean().item(),
                                                        "LPIPS": torch.tensor(lpipss).mean().item(),
                                                        "FLIP": torch.tensor(flipped_metrics).mean().item(),
                                                        "RMSE": torch.tensor(rmses).mean().item()})
                per_view_dict[scene_dir][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                            "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                            "PSNR*": {name: psnr for psnr, name in zip(torch.tensor(psnrs_star).tolist(), image_names)},
                                                            "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)},
                                                            # "FLIP": {name: lp for lp, name in zip(torch.tensor(flipped_metrics).tolist(), image_names)},
                                                            "RMSES": {name: lp for lp, name in zip(torch.tensor(rmses).tolist(), image_names)}})

            with open(scene_dir + "/results.json", 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            with open(scene_dir + "/per_view.json", 'w') as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)


if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    args = parser.parse_args()
    evaluate(args.model_paths)
