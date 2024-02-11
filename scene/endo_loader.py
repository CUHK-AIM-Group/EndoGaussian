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
import torch
import time


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
        test_every=8,
        mode='binocular'
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
        self.mode = mode

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
        if self.mode == 'binocular':
            self.depth_paths = agg_fn("depth")
        elif self.mode == 'monocular':
            self.depth_paths = agg_fn("monodepth")
        else:
            raise ValueError(f"{self.mode} has not been implemented.")
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
        
        for idx in tqdm(idxs):
            # mask / depth
            mask_path = self.masks_paths[idx]
            mask = Image.open(mask_path)
            mask = 1 - np.array(mask) / 255.0
            depth_path = self.depth_paths[idx]
            if self.mode == 'binocular':
                depth = np.array(Image.open(depth_path))
                close_depth = np.percentile(depth[depth!=0], 3.0)
                inf_depth = np.percentile(depth[depth!=0], 99.8)
                depth = np.clip(depth, close_depth, inf_depth)
            elif self.mode == 'monocular':
                depth = np.array(Image.open(depth_path))[..., 0]/255.0
                depth[depth!=0] = (1 / depth[depth!=0])*0.1
                depth[depth==0] = depth.max()
                depth = depth[...,None]
            else:
                raise ValueError(f"{self.mode} has not been implemented.")
            depth = torch.from_numpy(depth)
            mask = self.transform(mask).bool()
            # color
            color = np.array(Image.open(self.image_paths[idx]))/255.0
            image = self.transform(color)
            # times           
            time = self.image_times[idx]
            # poses
            R, T = self.image_poses[idx]
            # fov
            FovX = focal2fov(self.focal[0], self.img_wh[0])
            FovY = focal2fov(self.focal[1], self.img_wh[1])
            
            cameras.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image, depth=depth, mask=mask,
                                image_path=self.image_paths[idx], image_name=self.image_paths[idx], width=image.shape[2], height=image.shape[1],
                                time=time, Znear=None, Zfar=None))
    
        return cameras
    
    def get_init_pts(self):
        pts_total, colors_total = [], []
        for idx in self.train_idxs:
            color, depth, mask = self.get_color_depth_mask(idx, mode=self.mode)
            pts, colors, _ = self.get_pts_cam(depth, mask, color, disable_mask=False)
            pts = self.get_pts_wld(pts, self.image_poses[idx])
            num_pts = pts.shape[0]
            sel_idxs = np.random.choice(num_pts, int(0.01*num_pts), replace=True)
            pts_sel, colors_sel = pts[sel_idxs], colors[sel_idxs]
            pts_total.append(pts_sel)
            colors_total.append(colors_sel)

        pts_total = np.concatenate(pts_total)
        colors_total = np.concatenate(colors_total)
        sel_idxs = np.random.choice(pts_total.shape[0], 30_000, replace=True)
        pts, colors = pts_total[sel_idxs], colors_total[sel_idxs]
        normals = np.zeros((pts.shape[0], 3))
        
        return pts, colors, normals
        
    def get_pts_wld(self, pts, pose):
        R, T = pose
        R = np.transpose(R)
        w2c = np.concatenate((R, T[...,None]), axis=-1)
        w2c = np.concatenate((w2c, np.array([[0, 0, 0, 1]])), axis=0)
        c2w = np.linalg.inv(w2c)
        pts_cam_homo = np.concatenate((pts, np.ones((pts.shape[0], 1))), axis=-1)
        pts_wld = np.transpose(c2w @ np.transpose(pts_cam_homo))
        pts_wld = pts_wld[:, :3]
        return pts_wld
    
    def get_color_depth_mask(self, idx, mode):
        if mode == 'binocular':
            depth = np.array(Image.open(self.depth_paths[idx]))
            close_depth = np.percentile(depth[depth!=0], 3.0)
            inf_depth = np.percentile(depth[depth!=0], 99.8)
            depth = np.clip(depth, close_depth, inf_depth)
        else:
            depth = np.array(Image.open(self.depth_paths[idx]))[...,0]/255.0
            depth[depth!=0] = (1 / depth[depth!=0])
            depth[depth==0] = depth.max()
            
        mask = 1 - np.array(Image.open(self.masks_paths[idx]))/255.0
        color = np.array(Image.open(self.image_paths[idx]))/255.0
        return color, depth, mask
             
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

class SCARED_Dataset(object):
    def __init__(
        self,
        datadir,
        downsample=1.0,
        skip_every=2,
        test_every=8,
        init_pts=200_000,
        mode='binocular'
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
        self.depth_near_thresh = 30.0
        self.mode = mode
        self.init_pts = init_pts

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
        monodisps_dir = osp.join(self.root_dir, "data", "left_monodepth")
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
            
            if self.mode == 'binocular':
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
            elif self.mode == 'monocular':
                disp_dir = osp.join(monodisps_dir, f"{frame_id}_depth.png")
                disp = iio.imread(disp_dir).astype(np.float32)[...,0] / 255.0
                h, w = disp.shape
                disp[disp!=0] = (1 / disp[disp!=0])
                disp[disp==0] = disp.max()
                depth = disp
                depth = (depth - depth.min()) / (depth.max()-depth.min())
                depth = self.depth_near_thresh + (self.depth_far_thresh-self.depth_near_thresh)*depth
            else:
                raise ValueError(f"{self.mode} is not implemented!")
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
        
        camera_mat = self.camera_mat[0]
        self.focal = (camera_mat[0, 0], camera_mat[1, 1])
        
    def format_infos(self, split):
        cameras = []
        if split == 'train':
            idxs = self.train_idxs
        elif split == 'test':
            idxs = self.test_idxs
        else:
            idxs = sorted(self.train_idxs + self.test_idxs)
        
        for idx in idxs:
            image = self.rgbs[idx]
            image = self.transform(image)
            mask = self.masks[idx]
            mask = self.transform(mask).bool()
            depth = self.depths[idx]
            close_depth = np.percentile(depth[depth!=0], 3.0)
            inf_depth = np.percentile(depth[depth!=0], 99.8)
            depth = np.clip(depth, close_depth, inf_depth)
            depth = torch.from_numpy(depth)
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
            
    def get_init_pts(self, mode='hgi'):
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
            
            o3d.io.write_point_cloud('tmp.ply', pcd)
            
            return xyz, rgb, normals
        
        elif mode == 'hgi':
            pts_total, colors_total = [], []
            for idx in self.train_idxs:
                color, depth, mask = self.rgbs[idx], self.depths[idx], self.masks[idx]
                pts, colors, _ = self.get_pts_cam(depth, mask, color, disable_mask=False)
                pts = self.get_pts_wld(pts, self.pose_mat[idx])
                num_pts = pts.shape[0]
                sel_idxs = np.random.choice(num_pts, int(0.1*num_pts), replace=True)
                pts_sel, colors_sel = pts[sel_idxs], colors[sel_idxs]
                pts_total.append(pts_sel)
                colors_total.append(colors_sel)
            
            pts_total = np.concatenate(pts_total)
            colors_total = np.concatenate(colors_total)
            sel_idxs = np.random.choice(pts_total.shape[0], self.init_pts, replace=True)
            pts, colors = pts_total[sel_idxs], colors_total[sel_idxs]
            normals = np.zeros((pts.shape[0], 3))
            
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(np.array(pts_total))
            pcd.colors = o3d.utility.Vector3dVector(np.array(colors_total))
            o3d.io.write_point_cloud('tmp.ply', pcd)
            
            return pts, colors, normals

        elif mode == 'hgi_single':
            idx = self.train_idxs[0]
            color, depth, mask = self.rgbs[idx], self.depths[idx], self.masks[idx]
            close_depth = np.percentile(depth[depth!=0], 3.0)
            inf_depth = np.percentile(depth[depth!=0], 99.8)
            depth = np.clip(depth, close_depth, inf_depth)
            
            pts, colors, _ = self.get_pts_cam(depth, mask, color, disable_mask=False)
            pts = self.get_pts_wld(pts, self.pose_mat[idx])
            num_pts = pts.shape[0]
            sel_idxs = np.random.choice(num_pts, int(0.05*num_pts), replace=True)
            pts, colors = pts[sel_idxs], colors[sel_idxs]
            normals = np.zeros((pts.shape[0], 3))
            
            return pts, colors, normals
            
        else:
            raise ValueError(f'Mode {mode} has not been implemented yet')
    
    def get_pts_wld(self, pts, pose):
        c2w = pose
        pts_cam_homo = np.concatenate((pts, np.ones((pts.shape[0], 1))), axis=-1)
        pts_wld = np.transpose(c2w @ np.transpose(pts_cam_homo))
        pts_wld = pts_wld[:, :3]
        return pts_wld
             
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
            pts_valid = pts_cam
            color_valid = color
                    
        return pts_valid, color_valid, mask
        
    def get_maxtime(self):
        return self.maxtime
