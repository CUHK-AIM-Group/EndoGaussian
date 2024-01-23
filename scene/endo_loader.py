import warnings

warnings.filterwarnings("ignore")

import json
import os
import random
import os.path as osp

import numpy as np
from PIL import Image
from tqdm import tqdm
from scene.utils import Camera
from typing import NamedTuple
from torch.utils.data import Dataset
from utils.general_utils import PILtoTorch
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import glob
from torchvision import transforms as T
import open3d as o3d
from tqdm import trange
import imageio.v2 as iio
import cv2
import copy
import torch
import torch.nn.functional as F



class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    depth: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    time : float
    mask: np.array
    Zfar: float
    Znear: float


def normalize(v):
    """Normalize a vector."""
    return v / np.linalg.norm(v)


def average_poses(poses):
    """
    Calculate the average pose, which is then used to center all poses
    using @center_poses. Its computation is as follows:
    1. Compute the center: the average of pose centers.
    2. Compute the z axis: the normalized average z axis.
    3. Compute axis y': the average y axis.
    4. Compute x' = y' cross product z, then normalize it as the x axis.
    5. Compute the y axis: z cross product x.

    Note that at step 3, we cannot directly use y' as y axis since it's
    not necessarily orthogonal to z axis. We need to pass from x to y.
    Inputs:
        poses: (N_images, 3, 4)
    Outputs:
        pose_avg: (3, 4) the average pose
    """
    # 1. Compute the center
    center = poses[..., 3].mean(0)  # (3)
    # 2. Compute the z axis
    z = normalize(poses[..., 2].mean(0))  # (3)
    # 3. Compute axis y' (no need to normalize as it's not the final output)
    y_ = poses[..., 1].mean(0)  # (3)
    # 4. Compute the x axis
    x = normalize(np.cross(z, y_))  # (3)
    # 5. Compute the y axis (as z and x are normalized, y is already of norm 1)
    y = np.cross(x, z)  # (3)
    pose_avg = np.stack([x, y, z, center], 1)  # (3, 4)
    return pose_avg


def center_poses(poses, blender2opencv=None):
    """
    Center the poses so that we can use NDC.
    See https://github.com/bmild/nerf/issues/34
    Inputs:
        poses: (N_images, 3, 4)
    Outputs:
        poses_centered: (N_images, 3, 4) the centered poses
        pose_avg: (3, 4) the average pose
    """
    if blender2opencv is not None:
        poses = poses @ blender2opencv
    pose_avg = average_poses(poses)  # (3, 4)
    pose_avg_homo = np.eye(4)
    pose_avg_homo[:3] = pose_avg  # convert to homogeneous coordinate for faster computation
    pose_avg_homo = pose_avg_homo
    # by simply adding 0, 0, 0, 1 as the last row
    last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1))  # (N_images, 1, 4)
    poses_homo = np.concatenate([poses, last_row], 1)  # (N_images, 4, 4) homogeneous coordinate
    poses_centered = np.linalg.inv(pose_avg_homo) @ poses_homo  # (N_images, 4, 4)
    poses_centered = poses_centered[:, :3]  # (N_images, 3, 4)

    return poses_centered, pose_avg_homo


def farthest_point_sample(xyz, npoint): 

    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    B, N, C = xyz.shape
    centroids = np.zeros((B, npoint))
    distance = np.ones((B, N)) * 1e10
    batch_indices = np.arange(B)
    barycenter = np.sum((xyz), 1)
    barycenter = barycenter/xyz.shape[1]
    barycenter = barycenter.reshape(B, 1, C)
    dist = np.sum((xyz - barycenter) ** 2, -1)
    farthest = np.argmax(dist,1)
    
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].reshape(B, 1, C)
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)

    return centroids.astype(np.int32)


class EndoNeRF_Dataset(object):
    def __init__(
        self,
        datadir,
        downsample=1.0,
        test_every=8
    ):
        self.img_wh = (
            int(640 / downsample),
            int(512 / downsample),
        )
        self.root_dir = datadir
        self.downsample = downsample 
        self.blender2opencv = np.eye(4)
        self.transform = T.ToTensor()
        self.white_bg = False

        self.load_meta()
        print(f"meta data loaded, total image:{len(self.image_paths)}")
        
        n_frames = len(self.image_paths)
        self.train_idxs = [i for i in range(n_frames) if (i-1) % test_every != 0]
        self.test_idxs = [i for i in range(n_frames) if (i-1) % test_every == 0]
        self.video_idxs = [i for i in range(n_frames)]
        
        self.maxtime = 1.0
        
    def load_meta(self):
        """
        Load meta data from the dataset.
        """
        # load poses
        poses_arr = np.load(os.path.join(self.root_dir, "poses_bounds.npy"))
        poses = poses_arr[:, :-2].reshape([-1, 3, 5])  # (N_cams, 3, 5)
        # coordinate transformation OpenGL->Colmap, center poses
        H, W, focal = poses[0, :, -1]
        focal = focal / self.downsample
        self.focal = (focal, focal)
        self.K = np.array([[focal, 0 , W//2],
                                    [0, focal, H//2],
                                    [0, 0, 1]]).astype(np.float32)
        poses = np.concatenate([poses[..., :1], -poses[..., 1:2], -poses[..., 2:3], poses[..., 3:4]], -1)
        # poses, _ = center_poses(poses)  # Re-center poses so that the average is near the center.
        
        # prepare poses
        self.image_poses = []
        self.image_times = []
        for idx in range(poses.shape[0]):
            pose = poses[idx]
            c2w = np.concatenate((pose, np.array([[0, 0, 0, 1]])), axis=0) # 4x4
            w2c = np.linalg.inv(c2w)
            R = w2c[:3, :3]
            T = w2c[:3, -1]
            R = np.transpose(R)
            self.image_poses.append((R, T))
            self.image_times.append(idx / poses.shape[0])
        
        # get paths of images, depths, masks, etc.
        agg_fn = lambda filetype: sorted(glob.glob(os.path.join(self.root_dir, filetype, "*.png")))
        self.image_paths = agg_fn("images")
        self.depth_paths = agg_fn("depth")
        self.masks_paths = agg_fn("masks")

        assert len(self.image_paths) == poses.shape[0], "the number of images should equal to the number of poses"
        assert len(self.depth_paths) == poses.shape[0], "the number of depth images should equal to number of poses"
        assert len(self.masks_paths) == poses.shape[0], "the number of masks should equal to the number of poses"
        
    def format_infos(self, split):
        cameras = []
        
        if split == 'train': idxs = self.train_idxs
        elif split == 'test': idxs = self.test_idxs
        else:
            idxs = self.video_idxs
        
        depths = []
        depths_tensor = []
        depths_cnt = []
        colors = []
        colors_tensor = []
        colors_cnt = []
        mask_nps = []
        original_idxs = []
        
        for idx in tqdm(idxs):
            mask_path = self.masks_paths[idx]
            depth_path = self.depth_paths[idx]
            mask = Image.open(mask_path)
            mask = 1 - np.array(mask) / 255.0
            depth = np.array(Image.open(depth_path))
            depth_mask = np.ones(depth.shape).astype(np.float32)
            close_depth = np.percentile(depth[depth!=0], 3.0)
            inf_depth = np.percentile(depth[depth!=0], 98.0)
            depth_mask[depth>inf_depth] = 0
            depth_mask[np.bitwise_and(depth<close_depth, depth!=0)] = 0
            depth_mask[depth==0] = 0
            mask_np = np.logical_and(depth_mask, mask)
            
            depth[mask_np==0] = 0
            depth_tmp = copy.deepcopy(depth)
            depth_cnt = np.ones_like(depth_tmp)
            depth_cnt[mask_np==0] = 0
            depth_tmp = torch.from_numpy(depth_tmp).unsqueeze(0).unsqueeze(1).float()
            depth_cnt = torch.from_numpy(depth_cnt).unsqueeze(0).unsqueeze(1).float()
            
            image_path = self.image_paths[idx]
            image_name = f"{idx}"
            color = np.array(Image.open(image_path))/255.0
            color[mask_np==0, :] = np.zeros(3)
            color_tmp = copy.deepcopy(color)
            color_cnt = np.ones_like(depth)
            color_cnt[mask_np==0] = 0
            color_tmp = torch.from_numpy(color_tmp).unsqueeze(0).float()
            color_tmp = color_tmp.permute(0, 3, 1, 2).contiguous()
            color_cnt = torch.from_numpy(color_cnt).unsqueeze(0).unsqueeze(1).float()
            
            depths.append(depth)
            depths_tensor.append(depth_tmp)
            mask_nps.append(mask_np)
            depths_cnt.append(depth_cnt)
            colors.append(color)
            colors_tensor.append(color_tmp)
            colors_cnt.append(color_cnt)
            original_idxs.append(idx)
        
        depth_tmp_tensor = torch.cat(depths_tensor, dim=0).cuda()
        depth_cnt_tensor = torch.cat(depths_cnt, dim=0).cuda()
        with torch.no_grad():
            depth_tmp_tensor = F.conv2d(depth_tmp_tensor, weight=torch.ones(1,1,251,251).cuda(), stride=(1,1), padding='same')
            ones_tmp_tensor = F.conv2d(depth_cnt_tensor, weight=torch.ones(1,1,251,251).cuda(), stride=(1,1), padding='same')
        
        color_list = []
        color_tmp_tensor = torch.cat(colors_tensor, dim=0).cuda()
        color_cnt_tensor = torch.cat(colors_cnt, dim=0).cuda()
        with torch.no_grad():
            for i in range(color_tmp.shape[1]):
                color_tmp_c = F.conv2d(color_tmp_tensor[:,i].unsqueeze(1), weight=torch.ones(1, 1, 251, 251).cuda(), stride=(1, 1), padding='same')
                color_list.append(color_tmp_c)
        color_tmp_tensor = torch.cat(color_list, dim=1)
        color_ones_tmp = F.conv2d(color_cnt_tensor, weight=torch.ones(1,1,251,251).cuda(), stride=(1, 1), padding='same')
        color_ones_tmp_tensor = color_ones_tmp.repeat_interleave(3, dim=1)
        
        for idx in range(len(depths)):
            # depth & mask
            depth = depths[idx]
            mask_np = mask_nps[idx]
            ones_tmp = ones_tmp_tensor[idx].detach().cpu()
            depth_tmp = depth_tmp_tensor[idx].detach().cpu()
            depth_tmp[ones_tmp!=0] = depth_tmp[ones_tmp!=0] / ones_tmp[ones_tmp!=0]
            depth_tmp = depth_tmp.squeeze().detach().cpu().numpy()
            depth = depth * (mask_np!=0) + depth_tmp * (mask_np==0)
            depth = self.transform(depth)
            mask = self.transform(mask_np).bool()
            # color
            color = colors[idx]
            color_ones_tmp = color_ones_tmp_tensor[idx].detach().cpu()
            color_tmp = color_tmp_tensor[idx].detach().cpu()
            color_tmp[color_ones_tmp!=0] = color_tmp[color_ones_tmp!=0] / color_ones_tmp[color_ones_tmp!=0]
            color_tmp = color_tmp.permute(1, 2, 0).contiguous()
            color_tmp = color_tmp.detach().cpu().numpy()
            color = color * (mask_np!=0)[..., None] + color_tmp * (mask_np==0)[..., None]
            image = self.transform(color)
            # times           
            time = self.image_times[original_idxs[idx]]
            # poses
            R, T = self.image_poses[original_idxs[idx]]
            # fov
            FovX = focal2fov(self.focal[0], self.img_wh[0])
            FovY = focal2fov(self.focal[1], self.img_wh[1])
            
            cameras.append(CameraInfo(uid=original_idxs[idx], R=R, T=T, FovY=FovY, FovX=FovX, image=image, depth=depth, mask=mask,
                                image_path=image_path, image_name=image_name, width=image.shape[2], height=image.shape[1],
                                time=time, Znear=None, Zfar=None))
    
        return cameras
            
    def generate_cameras(self, mode='fixidentity'):
        cameras = []
        image = Image.open(self.image_paths[0])
        image = self.transform(image)
        if mode == 'fixidentity':
            render_times = self.image_times
            for idx, time in enumerate(render_times):
                R, T = self.image_poses[0]
                FovX = focal2fov(self.focal[0], self.img_wh[0])
                FovY = focal2fov(self.focal[0], self.img_wh[1])
                cameras.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX,
                                          image=image, image_path=None, image_name=None, depth=None, mask=None,
                                          width=self.img_wh[0], height=self.img_wh[1], time=time, Znear=None, Zfar=None))
            return cameras
        else:
            raise ValueError(f'{mode} not implemented yet')
    
    def get_sparse_pts(self, sample=True):
        # import pdb; pdb.set_trace()
        # xyz & color
        R, T = self.image_poses[0]
        c2w = self.get_camera_poses((R, T))
        depth = np.array(Image.open(self.depth_paths[0]))
        mask = 1 - np.array(Image.open(self.masks_paths[0]))/255.0
        color = np.array(Image.open(self.image_paths[0]))/255.0
        pts_cam, color, _ = self.get_pts_cam(depth, mask, color)
        pts_cam_homo = np.concatenate((pts_cam, np.ones((pts_cam.shape[0], 1))), axis=-1)
        pts_wld = np.transpose(c2w @ np.transpose(pts_cam_homo))
        xyz = pts_wld[:, :3]
        
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(xyz)
        # pcd.colors = o3d.utility.Vector3dVector(color)
        # o3d.io.write_point_cloud('before_combine.ply', pcd)
        
        # xyzi, colori = self.search_pts(mask)
        
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(xyzi)
        # colori = np.concatenate((colori[..., 2:3], colori[..., 1:2], colori[..., 0:1]), axis=-1)
        # pcd.colors = o3d.utility.Vector3dVector(colori)
        # o3d.io.write_point_cloud('complements.ply', pcd)
        
        
        # xyz = np.concatenate((xyz, xyzi), axis=0)
        # color = np.concatenate((color, colori), axis=0)
        normals = np.zeros((xyz.shape[0], 3))
        
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(xyz)
        # pcd.colors = o3d.utility.Vector3dVector(color)
        # o3d.io.write_point_cloud('after_combine.ply', pcd)
    
        if sample:
            num_sample = int(0.1*xyz.shape[0])
            sel_idxs = np.random.choice(xyz.shape[0], num_sample, replace=False)
            # sel_idxs = farthest_point_sample(xyz[np.newaxis, :], num_sample)[0, :]
            xyz = xyz[sel_idxs, :]
            color = color[sel_idxs, :]
            normals = normals[sel_idxs, :]
        return xyz, color, normals
    
    def get_sparse_pts_v2(self, sample=True):
        R, T = self.image_poses[0]
        depth = np.array(Image.open(self.depth_paths[0]))
        mask = 1 - np.array(Image.open(self.masks_paths[0]))/255.0
        color = np.array(Image.open(self.image_paths[0]))/255.0
        
        depth_mask = np.ones(depth.shape).astype(np.float32)
        close_depth = np.percentile(depth[depth!=0], 3.0)
        inf_depth = np.percentile(depth[depth!=0], 98.0)
        depth_mask[depth>inf_depth] = 0
        depth_mask[np.bitwise_and(depth<close_depth, depth!=0)] = 0
        depth_mask[depth==0] = 0
        mask = np.logical_and(depth_mask, mask)
        depth[mask==0] = 0
        
        depth_tmp = copy.deepcopy(depth)
        depth_cnt = np.ones_like(depth_tmp)
        depth_cnt[mask==0] = 0
        depth_tmp = torch.from_numpy(depth_tmp).unsqueeze(0).unsqueeze(1).float()
        depth_cnt = torch.from_numpy(depth_cnt).unsqueeze(0).unsqueeze(1).float()
       
        # depth - convolution
        depth_tmp = F.conv2d(depth_tmp, weight=torch.ones(1,1,251,251), stride=(1,1), padding='same')
        ones_tmp = F.conv2d(depth_cnt, weight=torch.ones(1,1,251,251), stride=(1,1), padding='same')
        depth_tmp[ones_tmp!=0] = depth_tmp[ones_tmp!=0] / ones_tmp[ones_tmp!=0]
        depth_tmp = depth_tmp.squeeze().detach().cpu().numpy()
        depth = depth * (mask!=0) + depth_tmp * (mask==0)
        
        # color- convolution
        color[mask==0, :] = np.zeros(3)
        color_tmp = copy.deepcopy(color)
        color_cnt = np.ones_like(depth_tmp)
        color_cnt[mask==0] = 0
        color_tmp = torch.from_numpy(color_tmp).unsqueeze(0).float()
        color_tmp = color_tmp.permute(0, 3, 1, 2).contiguous()
        color_cnt = torch.from_numpy(color_cnt).unsqueeze(0).float()
        color_list = []
        
        for i in range(color_tmp.shape[1]):
            color_tmp_c = F.conv2d(color_tmp[:,i].unsqueeze(1), weight=torch.ones(1, 1, 251, 251), stride=(1, 1), padding='same')
            color_list.append(color_tmp_c)
        color_tmp = torch.cat(color_list, dim=1)
        ones_tmp = F.conv2d(color_cnt.unsqueeze(0), weight=torch.ones(1,1,251,251), stride=(1, 1), padding='same')
        ones_tmp = ones_tmp.repeat_interleave(3, dim=1)
        color_tmp[ones_tmp!=0] = color_tmp[ones_tmp!=0] / ones_tmp[ones_tmp!=0]
        color_tmp = color_tmp.permute(0, 2, 3, 1).contiguous()
        color_tmp = color_tmp.squeeze(0).detach().cpu().numpy()
        color = color * (mask!=0)[..., None] + color_tmp * (mask==0)[..., None]
        
        # color - Maxpooling
        # iteration_times = 20
        # for i in range(iteration_times):
        #     color_tmp = torch.nn.functional.max_pool2d(color_tmp, kernel_size=(11,11), stride=(1, 1), padding=(5, 5))
        #     cv2.imwrite("color_tmp.png", np.clip((color_tmp.permute(0, 2, 3, 1).squeeze(0).cpu().numpy()*255), 0, 255).astype(np.uint8))
        # color_tmp = color_tmp.permute(0, 2, 3, 1).squeeze(0).cpu().numpy()

        pts, colors, _ = self.get_pts_cam(depth, mask, color, disable_mask=True)
        c2w = self.get_camera_poses((R, T))
        pts = self.transform_cam2cam(pts, c2w)
        pts, colors = self.search_pts_colors(pts, colors, mask, c2w)
        normals = np.zeros((pts.shape[0], 3))

        if sample:
            num_sample = int(0.1 * pts.shape[0])
            sel_idxs = np.random.choice(pts.shape[0], num_sample, replace=False)
            # sel_idxs = farthest_point_sample(xyz[np.newaxis, :], num_sample)[0, :]
            pts = pts[sel_idxs, :]
            colors = colors[sel_idxs, :]
            normals = normals[sel_idxs, :]
        
        return pts, colors, normals
    
    def search_pts(self, mask):
        import copy
        iteration_times = 1
        mask = mask.astype(bool)
        mask_paths = copy.deepcopy(self.masks_paths)
        xyzi_list, colori_list = [], []
        
        for j in range(iteration_times):
            max_val = -1
            max_ind = -1
            max_mask = None
            
            for i in range(1, len(mask_paths)):
                mask_i = (1 - np.array(Image.open(mask_paths[i]))/255.0).astype(bool)
                intersec = (~mask * mask_i).sum()
                if intersec > max_val:
                    max_val = intersec
                    max_ind = i
                    max_mask = mask_i
            
            Ri, Ti = self.image_poses[max_ind]
            c2wi = self.get_camera_poses((Ri, Ti))
            depthi = np.array(Image.open(self.depth_paths[max_ind]))
            mask_left = (~mask * max_mask).astype(np.float32)
            colori = np.array(Image.open(self.image_paths[max_ind]))/255.0
            
            colori = np.concatenate((colori[..., 2:], colori[...,1:2], colori[...,0:1]), axis=-1)
            cv2.imwrite('color.png', (colori*255).astype(np.uint8))
            cv2.imwrite('mask.png', ((1-max_mask)*255).astype(np.uint8))
            cv2.imwrite('mask_left.png', ((1-mask_left)*255).astype(np.uint8))
            
            pts_cami, colori, _ = self.get_pts_cam(depthi, mask_left, colori)
            pts_cam_homoi = np.concatenate((pts_cami, np.ones((pts_cami.shape[0], 1))), axis=-1)
            pts_wldi = np.transpose(c2wi @ np.transpose(pts_cam_homoi))
            xyzi = pts_wldi[:, :3]
            
            mask = np.logical_or(mask, mask_left.astype(bool))
            mask_paths.pop(max_ind)
            
            xyzi_list.append(xyzi)
            colori_list.append(colori)

        xyzi = np.concatenate(xyzi_list, axis=0)
        colori = np.concatenate(colori_list, axis=0)
        
        return xyzi, colori

    def search_pts_colors(self, ref_pts, ref_color, ref_mask, ref_c2w):
    
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(ref_pts))
        pcd.colors = o3d.utility.Vector3dVector(np.array(ref_color))
        o3d.io.write_point_cloud('before.ply', pcd)

        for j in range(1, len(self.image_poses)):
            ref_mask_not = np.logical_not(ref_mask)
            R, T = self.image_poses[j]
            c2w = self.get_camera_poses((R, T))
            c2ref = np.linalg.inv(ref_c2w) @ c2w
            depth = np.array(Image.open(self.depth_paths[j]))
            color = np.array(Image.open(self.image_paths[j]))/255.0
            mask = 1 - np.array(Image.open(self.masks_paths[j]))/255.0
            
            depth_mask = np.ones(depth.shape).astype(np.float32)
            close_depth = np.percentile(depth[depth!=0], 3.0)
            inf_depth = np.percentile(depth[depth!=0], 98.0)
            depth_mask[depth>inf_depth] = 0
            depth_mask[np.bitwise_and(depth<close_depth, depth!=0)] = 0
            depth_mask[depth==0] = 0
            mask = np.logical_and(depth_mask, mask)
            depth[mask==0] = 0
            
            pts, colors, mask_refine = self.get_pts_cam(depth, mask, color)
            pts = self.transform_cam2cam(pts, c2ref) # Nx3
            X, Y, Z = pts[..., 0], pts[..., 1], pts[..., 2]
            X, Y, Z = X[Z!=0], Y[Z!=0], Z[Z!=0]
            X_Z, Y_Z = X / Z, Y / Z
            X_Z = (X_Z * self.focal[0] + self.img_wh[0]/2).astype(np.int32)
            Y_Z = (Y_Z * self.focal[1] + self.img_wh[1]/2).astype(np.int32)
            X_Z = np.clip(X_Z, 0, self.img_wh[0]-1)
            Y_Z = np.clip(Y_Z, 0, self.img_wh[1]-1)
            coords = np.stack((Y_Z, X_Z), axis=-1)
            proj_mask = np.zeros((self.img_wh[1], self.img_wh[0])).astype(np.float32)
            proj_mask[coords[:, 0], coords[:, 1]] = 1
            compl_mask = (ref_mask_not * proj_mask)
            index_mask = compl_mask.reshape(-1)[mask_refine]
            compl_idxs = np.nonzero(index_mask.reshape(-1))[0]
            if compl_idxs.shape[0] <= 50:
                continue
            compl_pts = pts[compl_idxs, :]
            compl_pts = self.transform_cam2cam(compl_pts, ref_c2w)
            compl_colors = colors[compl_idxs, :]

            ref_pts = np.concatenate((ref_pts, compl_pts), axis=0)
            ref_color = np.concatenate((ref_color, compl_colors), axis=0)
            ref_mask = np.logical_or(ref_mask, compl_mask)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(ref_pts))
        pcd.colors = o3d.utility.Vector3dVector(np.array(ref_color))
        # pcd, _ = pcd.remove_radius_outlier(nb_points=5,
        #                              radius=np.asarray(pcd.compute_nearest_neighbor_distance()).mean() * 20.)
        o3d.io.write_point_cloud(f'after.ply', pcd)        
        return ref_pts, ref_color
         
    def get_camera_poses(self, pose_tuple):
        R, T = pose_tuple
        R = np.transpose(R)
        w2c = np.concatenate((R, T[...,None]), axis=-1)
        w2c = np.concatenate((w2c, np.array([[0, 0, 0, 1]])), axis=0)
        c2w = np.linalg.inv(w2c)
        return c2w
    
    def get_pts_cam(self, depth, mask, color, disable_mask=False):
        W, H = self.img_wh
        i, j = np.meshgrid(np.linspace(0, W-1, W), np.linspace(0, H-1, H))
        X_Z = (i-W/2) / self.focal[0]
        Y_Z = (j-H/2) / self.focal[1]
        Z = depth
        X, Y = X_Z * Z, Y_Z * Z
        pts_cam = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)
        color = color.reshape(-1, 3)
        
        if not disable_mask:
            mask = mask.reshape(-1).astype(bool)
            pts_valid = pts_cam[mask, :]
            color_valid = color[mask, :]
        else:
            mask = mask.reshape(-1).astype(bool)
            pts_valid = pts_cam
            color_valid = color
            color_valid[mask==0, :] = np.ones(3)
                    
        return pts_valid, color_valid, mask
        
    def get_maxtime(self):
        return self.maxtime
    
    def transform_cam2cam(self, pts_cam, pose):
        pts_cam_homo = np.concatenate((pts_cam, np.ones((pts_cam.shape[0], 1))), axis=-1)
        pts_wld = np.transpose(pose @ np.transpose(pts_cam_homo))
        xyz = pts_wld[:, :3]
        return xyz


class SCARED_Dataset(object):
    def __init__(
        self,
        datadir,
        downsample=1.0,
        skip_every=2,
        test_every=8
    ):
        if "dataset_1" in datadir:
            skip_every = 2
        elif "dataset_2" in datadir:
            skip_every = 1
        elif "dataset_3" in datadir:
            skip_every = 4
        elif "dataset_6" in datadir:
            skip_every = 8
        elif "dataset_7" in datadir:
            skip_every = 8
            
        self.img_wh = (
            int(1280 / downsample),
            int(1024 / downsample),
        )
        self.root_dir = datadir
        self.downsample = downsample
        self.skip_every = skip_every
        self.transform = T.ToTensor()
        self.white_bg = False
        self.depth_far_thresh = 300.0
        self.depth_near_thresh = 0.03

        self.load_meta()
        n_frames = len(self.rgbs)
        print(f"meta data loaded, total image:{n_frames}")
        
        self.train_idxs = [i for i in range(n_frames) if (i-1) % test_every!=0]
        self.test_idxs = [i for i in range(n_frames) if (i-1) % test_every==0]

        self.maxtime = 1.0
        
    def load_meta(self):
        """
        Load meta data from the dataset.
        """
        # prepare paths
        calibs_dir = osp.join(self.root_dir, "data", "frame_data")
        rgbs_dir = osp.join(self.root_dir, "data", "left_finalpass")
        disps_dir = osp.join(self.root_dir, "data", "disparity")
        reproj_dir = osp.join(self.root_dir, "data", "reprojection_data")
        frame_ids = sorted([id[:-5] for id in os.listdir(calibs_dir)])
        frame_ids = frame_ids[::self.skip_every]
        n_frames = len(frame_ids)
        
        rgbs = []
        bds = []
        masks = []
        depths = []
        pose_mat = []
        camera_mat = []
        
        for i_frame in trange(n_frames, desc="Process frames"):
            frame_id = frame_ids[i_frame]
            
            # intrinsics and poses
            with open(osp.join(calibs_dir, f"{frame_id}.json"), "r") as f:
                calib_dict = json.load(f)
            K = np.eye(4)
            K[:3, :3] = np.array(calib_dict["camera-calibration"]["KL"])
            camera_mat.append(K)
            c2w = np.linalg.inv(np.array(calib_dict["camera-pose"]))
            if i_frame == 0:
                c2w0 = c2w
            c2w = np.linalg.inv(c2w0) @ c2w
            pose_mat.append(c2w)
            
            # rgbs and depths
            rgb_dir = osp.join(rgbs_dir, f"{frame_id}.png")
            rgb = iio.imread(rgb_dir)
            rgbs.append(rgb)
            disp_dir = osp.join(disps_dir, f"{frame_id}.tiff")
            disp = iio.imread(disp_dir).astype(np.float32)
            h, w = disp.shape
            with open(osp.join(reproj_dir, f"{frame_id}.json"), "r") as json_file:
                Q = np.array(json.load(json_file)["reprojection-matrix"])
            fl = Q[2,3]
            bl =  1 / Q[3,2]
            disp_const = fl * bl
            mask_valid = (disp != 0)    
            depth = np.zeros_like(disp)
            depth[mask_valid] = disp_const / disp[mask_valid]
            depth[depth>self.depth_far_thresh] = 0
            depth[depth<self.depth_near_thresh] = 0
            depths.append(depth)
            
            # masks
            depth_mask = (depth != 0).astype(float)
            kernel = np.ones((int(w/128), int(w/128)),np.uint8)
            mask = cv2.morphologyEx(depth_mask, cv2.MORPH_CLOSE, kernel)
            masks.append(mask)
            
            # bounds
            bound = np.array([depth[depth!=0].min(), depth[depth!=0].max()])
            bds.append(bound)

        self.rgbs = np.stack(rgbs, axis=0).astype(np.float32) / 255.0
        self.pose_mat = np.stack(pose_mat, axis=0).astype(np.float32)
        self.camera_mat = np.stack(camera_mat, axis=0).astype(np.float32)
        self.depths = np.stack(depths, axis=0).astype(np.float32)
        self.masks = np.stack(masks, axis=0).astype(np.float32)
        self.bds = np.stack(bds, axis=0).astype(np.float32)
        self.times = np.linspace(0, 1, num=len(rgbs)).astype(np.float32)
        self.frame_ids = frame_ids
        
    def format_infos(self, split):
        cameras = []
        if split == 'train':
            idxs = self.train_idxs
        elif split == 'test':
            idxs = self.test_idxs
        else:
            self.generate_cameras(mode='fixidentity')
        
        for idx in idxs:
            image = self.rgbs[idx]
            image = self.transform(image)
            mask = self.masks[idx]
            mask = self.transform(mask).bool()
            depth = self.depths[idx]
            depth = self.transform(depth)
            time = self.times[idx]
            c2w = self.pose_mat[idx]
            w2c = np.linalg.inv(c2w)
            R, T = w2c[:3, :3], w2c[:3, -1]
            R = np.transpose(R)
            camera_mat = self.camera_mat[idx]
            focal_x, focal_y = camera_mat[0, 0], camera_mat[1, 1]
            FovX = focal2fov(focal_x, self.img_wh[0])
            FovY = focal2fov(focal_y, self.img_wh[1])
            cameras.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image, depth=depth, mask=mask,
                                image_path=None, image_name=None, width=self.img_wh[0], height=self.img_wh[1],
                                time=time, Znear=self.depth_near_thresh, Zfar=self.depth_far_thresh))
        return cameras
            
    def generate_cameras(self, mode='fixidentity'):
        cameras = []
        image = self.rgbs[0]
        image = self.transform(image)
        c2w = self.pose_mat[0]
        w2c = np.linalg.inv(c2w)
        R, T = w2c[:3, :3], w2c[:3, -1]
        R = np.transpose(R)
        camera_mat = self.camera_mat[0]
        focal_x, focal_y = camera_mat[0, 0], camera_mat[1, 1]
        FovX = focal2fov(focal_x, self.img_wh[0])
        FovY = focal2fov(focal_y, self.img_wh[1])
        
        if mode == 'fixidentity':
            render_times = self.times
            for idx, time in enumerate(render_times):
                FovX = focal2fov(focal_x, self.img_wh[0])
                FovY = focal2fov(focal_y, self.img_wh[1])
                cameras.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX,
                                          image=image, image_path=None, image_name=None, depth=None, mask=None,
                                          width=self.img_wh[0], height=self.img_wh[1], time=time, 
                                          Znear=self.depth_near_thresh, Zfar=self.depth_far_thresh))
            return cameras
        else:
            raise ValueError(f'{mode} not implemented yet')
    
    def get_pts(self, mode='o3d'):
        if mode == 'o3d':
            pose = self.pose_mat[0]
            K = self.camera_mat[0][:3, :3]
            rgb = self.rgbs[0]
            rgb_im = o3d.geometry.Image((rgb*255.0).astype(np.uint8))
            depth_im = o3d.geometry.Image(self.depths[0])
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_im, depth_im,
                                                                            depth_scale=1.,
                                                                            depth_trunc=self.bds.max(),
                                                                            convert_rgb_to_intensity=False)
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd_image, 
                o3d.camera.PinholeCameraIntrinsic(self.img_wh[0], self.img_wh[1], K),
                np.linalg.inv(pose),
                project_valid_depth_only=True,
            )
            pcd = pcd.random_down_sample(0.01)
            pcd, _ = pcd.remove_radius_outlier(nb_points=5,
                                        radius=np.asarray(pcd.compute_nearest_neighbor_distance()).mean() * 10.)
            xyz, rgb = np.asarray(pcd.points).astype(np.float32), np.asarray(pcd.colors).astype(np.float32)
            normals = np.zeros((xyz.shape[0], 3))
            return xyz, rgb, normals
        elif mode == 'naive':
            pass
        else:
            raise ValueError(f'Mode {mode} has not been implemented yet')
    
    def get_maxtime(self):
        return self.maxtime
        

def save_pcd(xyz, color, name):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(xyz))
    pcd.colors = o3d.utility.Vector3dVector(np.array(color))
    o3d.io.write_point_cloud(f"{name}", pcd)