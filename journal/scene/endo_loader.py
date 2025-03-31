import warnings

warnings.filterwarnings("ignore")

import json
import os
import random
import os.path as osp

import numpy as np
from PIL import Image
from tqdm import tqdm
from scene.cameras import Camera
from utils.general_utils import PILtoTorch, percentile_torch
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import glob
from torchvision import transforms
import open3d as o3d
from tqdm import trange
import imageio.v2 as iio
import cv2
import torch
import fpsample

import torch.nn.functional as F



class EndoNeRF_Dataset(object):
    def __init__(
        self,
        datadir,
        test_every=8,
        mode='binocular'
    ):
        # img parameters
        self.img_wh = (640, 512)
        self.root_dir = datadir
        self.blender2opencv = np.eye(4)
        self.transform = transforms.ToTensor()
        self.white_bg = False
        self.mode = mode
        
        # load poses and intrinsics 
        poses_arr = np.load(os.path.join(self.root_dir, "poses_bounds.npy"))
        poses = poses_arr[:, :-2].reshape([-1, 3, 5])  # (N_cams, 3, 5)
        H, W, focal = poses[0, :, -1]
        self.focal = (focal, focal)
        self.K = np.array([[focal, 0 , W//2],
                                    [0, focal, H//2],
                                    [0, 0, 1]]).astype(np.float32)
        poses = np.concatenate([poses[..., :1], poses[..., 1:2], poses[..., 2:3], poses[..., 3:4]], -1)
        
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
        image_paths = agg_fn("images")
        masks_paths = agg_fn("masks")
        image_interp_paths = agg_fn("images_interp")
                        
        if self.mode == 'binocular':
            depth_paths = agg_fn("depth")
        elif self.mode == 'monocular':
            depth_paths = agg_fn("monodepth")
        else:
            raise ValueError(f"{self.mode} has not been implemented.")
        
        assert len(image_paths) == len(depth_paths) == len(masks_paths) == poses.shape[0], \
            "the number of images should equal to the number of poses"
        print(f"meta data loaded, total image:{len(image_paths)}")
        
        images, masks, depths = self.load_files(image_paths, depth_paths, masks_paths)
        images_interp = self.load_interp_files(image_interp_paths)
        
        self.images = images
        self.masks = masks
        self.depths = depths
        self.images_interp = images_interp
        
        # training/test/video split
        n_frames = len(images)
        self.train_idxs = [i for i in range(n_frames) if (i-1) % test_every != 0]
        self.test_idxs = [i for i in range(n_frames) if (i-1) % test_every == 0]
        self.video_idxs = [i for i in range(n_frames)]
        
        self.maxtime = 1.0
        self.znear = 0.01
        self.zfar = 120
        
    def load_files(self, image_paths, depth_paths, masks_paths):
        images, masks, depths = [], [], []
        for (image_path, depth_path, mask_path) in zip(image_paths, depth_paths, masks_paths):
            # color
            color = np.array(Image.open(image_path))/255.0
            image = self.transform(color)
            # mask
            mask = Image.open(mask_path)
            mask = 1 - np.array(mask) / 255.0
            mask = self.transform(mask).bool()
            # depth
            if self.mode == 'binocular':
                depth = np.array(Image.open(depth_path))
                close_depth = np.percentile(depth[depth!=0], 3.0)
                inf_depth = np.percentile(depth[depth!=0], 99.8)
                depth = np.clip(depth, close_depth, inf_depth)
            elif self.mode == 'monocular':
                depth = np.array(Image.open(depth_path))[...,0] / 255.0
                depth[depth!=0] = (1 / depth[depth!=0])*0.4
                depth[depth==0] = depth.max()
                depth = depth[...,None]
            else:
                raise ValueError(f"{self.mode} has not been implemented.")
            depth = torch.from_numpy(depth)
            
            images.append(image)
            masks.append(mask)
            depths.append(depth)
            
        return images, masks, depths
    
    def load_interp_files(self, image_paths):
        images = []
        for image_path in image_paths:
            color = np.array(Image.open(image_path))/255.0
            image = self.transform(color)
            images.append(image)
        
        return images
    
    def format_infos(self, split):
        cameras = []
        if split != 'interp':
            if split == 'train': idxs = self.train_idxs
            elif split == 'test': idxs = self.test_idxs
            else:
                idxs = self.video_idxs
            
            for idx in tqdm(idxs):
                image = self.images[idx]
                mask = self.masks[idx]
                depth = self.depths[idx]
                time = self.image_times[idx]
                R, T = self.image_poses[idx]
                FovX = focal2fov(self.focal[0], self.img_wh[0])
                FovY = focal2fov(self.focal[1], self.img_wh[1])
                
                cameras.append(Camera(colmap_id=idx,R=R,T=T,FoVx=FovX,FoVy=FovY,image=image, depth=depth, mask=mask, gt_alpha_mask=None,
                            image_name=f"{idx}",uid=idx,data_device=torch.device("cuda"),time=time,
                            Znear=self.znear, Zfar=self.zfar))
        else:
            interp_step = 2
            for idx in tqdm(range(len(self.images_interp))):
                image = self.images_interp[idx]
                # mask = self.masks_interp[idx][0:1]
                if idx % interp_step == 0:
                    depth = self.depths[self.train_idxs[idx//interp_step]]
                    time = self.image_times[self.train_idxs[idx//interp_step]]
                    mask = self.masks[self.train_idxs[idx//interp_step]]
                else:
                    depth = None
                    mask_lower = self.masks[self.train_idxs[idx//interp_step]]
                    mask_upper = self.masks[self.train_idxs[idx//interp_step+1]]
                    mask = torch.logical_and(mask_lower, mask_upper)
                    # mask = (-F.max_pool2d(-(mask.float()).unsqueeze(0), kernel_size=5, stride=1, padding=2)).squeeze(0).bool()
                    time_lower = self.image_times[self.train_idxs[idx//interp_step]]
                    time_upper = self.image_times[self.train_idxs[idx//interp_step+1]]
                    time = time_lower + (time_upper - time_lower) * (idx % interp_step) * (1 / interp_step)
                    
                R, T = self.image_poses[0]
                FovX = focal2fov(self.focal[0], self.img_wh[0])
                FovY = focal2fov(self.focal[1], self.img_wh[1])
                    
                cameras.append(Camera(colmap_id=idx,R=R,T=T,FoVx=FovX,FoVy=FovY,image=image, depth=depth, mask=mask, gt_alpha_mask=None,
                            image_name=f"{idx}",uid=idx,data_device=torch.device("cuda"),time=time,
                            Znear=self.znear, Zfar=self.zfar))
    
        return cameras
    
    def get_init_pts(self, sampling='random'):
        if self.mode == 'binocular':
            
            initial_idx = self.train_idxs[0]
            color, depth, mask = self.images[initial_idx].numpy(), self.depths[initial_idx].numpy(), self.masks[initial_idx].numpy()
            pts, colors, _ = self.get_pts_cam(depth, mask, color, disable_mask=False)
            pts = self.get_pts_wld(pts, self.image_poses[initial_idx])
            
            # min_coords = np.min(pts, axis=0) 
            # max_coords = np.max(pts, axis=0)
            # pts = np.random.uniform(
            #     low=min_coords, 
            #     high=max_coords,
            #     size=(50000, 3)
            # )
            # colors = np.zeros((pts.shape[0], 3))
            # normals = np.zeros((pts.shape[0], 3))
            
            idxs = np.random.choice(np.arange(pts.shape[0]), 50000, replace=False)
            pts = pts[idxs, :]
            colors = colors[idxs, :]
            normals = np.zeros((pts.shape[0], 3))
            
        elif self.mode == 'monocular':
            initial_idx = self.train_idxs[0]
            color, depth, mask = self.images[initial_idx].numpy(), self.depths[initial_idx].numpy(), self.masks[initial_idx].numpy()
            pts, colors, _ = self.get_pts_cam(depth, mask, color, disable_mask=False)
            pts = self.get_pts_wld(pts, self.image_poses[initial_idx])
            idxs = np.random.choice(np.arange(pts.shape[0]), 50000, replace=False)
            pts = pts[idxs, :]
            colors = colors[idxs, :]
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
        self.depth_near_thresh = 0.03
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
        monodisps_dir = osp.join(self.root_dir, "data", "left_monodam")
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
                # disp_dir = osp.join(monodisps_dir, f"{frame_id}_depth.png")
                # disp = iio.imread(disp_dir).astype(np.float32)[...,0] / 255.0
                # h, w = disp.shape
                # disp[disp!=0] = (1 / disp[disp!=0])
                # disp[disp==0] = disp.max()
                # depth = disp
                # depth = (depth - depth.min()) / (depth.max()-depth.min())
                # depth = self.depth_near_thresh + (self.depth_far_thresh-self.depth_near_thresh)*depth
                disp_dir = osp.join(monodisps_dir, f"{frame_id}.png")
                depth = iio.imread(disp_dir).astype(np.float32) / 255.0
                h, w = depth.shape
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
            
            cameras.append(Camera(colmap_id=idx,R=R,T=T,FoVx=FovX,FoVy=FovY,image=image, depth=depth, mask=mask, gt_alpha_mask=None,
                          image_name=f"{idx}",uid=idx,data_device=torch.device("cuda"),time=time,
                          Znear=self.depth_near_thresh, Zfar=self.depth_far_thresh))
        return cameras
            
    def get_init_pts(self, mode='hgi', sampling='random'):
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
            pcd = pcd.random_down_sample(0.1)
            # pcd, _ = pcd.remove_radius_outlier(nb_points=5,
            #                             radius=np.asarray(pcd.compute_nearest_neighbor_distance()).mean() * 10.)
            xyz, rgb = np.asarray(pcd.points).astype(np.float32), np.asarray(pcd.colors).astype(np.float32)
            normals = np.zeros((xyz.shape[0], 3))
            
            # o3d.io.write_point_cloud('tmp.ply', pcd)
            
            return xyz, rgb, normals
        
        elif mode == 'hgi':
            pts_total, colors_total = [], []
            for idx in self.train_idxs:
                color, depth, mask = self.rgbs[idx], self.depths[idx], self.masks[idx]
                if self.mode == 'binocular':
                    mask = np.logical_and(mask, (depth>self.depth_near_thresh), (depth<self.depth_far_thresh))
                pts, colors, _ = self.get_pts_cam(depth, mask, color, disable_mask=False)
                pts = self.get_pts_wld(pts, self.pose_mat[idx])
                pts_total.append(pts)
                colors_total.append(colors)
                
                num_pts = pts.shape[0]
                if sampling == 'fps':
                    sel_idxs = fpsample.bucket_fps_kdline_sampling(pts, int(0.1*num_pts), h=3)
                elif sampling == 'random':
                    sel_idxs = np.random.choice(num_pts, int(0.1*num_pts), replace=False)
                else:
                    raise ValueError(f'{sampling} sampling has not been implemented yet.')
                
                pts_sel, colors_sel = pts[sel_idxs], colors[sel_idxs]
                pts_total.append(pts_sel)
                colors_total.append(colors_sel)
            
            pts_total = np.concatenate(pts_total)
            colors_total = np.concatenate(colors_total)
            sel_idxs = np.random.choice(pts_total.shape[0], self.init_pts, replace=True)
            pts, colors = pts_total[sel_idxs], colors_total[sel_idxs]
            normals = np.zeros((pts.shape[0], 3))
            
            return pts, colors, normals

        elif mode == 'hgi_mono':
            idx = self.train_idxs[0]
            color, depth, mask = self.rgbs[idx], self.depths[idx], self.masks[idx]
            pts, colors, _ = self.get_pts_cam(depth, mask, color, disable_mask=False)
            pts = self.get_pts_wld(pts, self.pose_mat[idx])
            num_pts = pts.shape[0]
            sel_idxs = np.random.choice(num_pts, int(0.5*num_pts), replace=True)
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


from dataclasses import dataclass

@dataclass
class Intrinsics:
    width: int
    height: int
    focal_x: float
    focal_y: float
    center_x: float
    center_y: float

    def scale(self, factor: float):
        nw = round(self.width * factor)
        nh = round(self.height * factor)
        sw = nw / self.width
        sh = nh / self.height
        self.focal_x *= sw
        self.focal_y *= sh
        self.center_x *= sw
        self.center_y *= sh
        self.width = int(nw)
        self.height = int(nh)

    def __repr__(self):
        return (f"Intrinsics(width={self.width}, height={self.height}, "
                f"focal_x={self.focal_x}, focal_y={self.focal_y}, "
                f"center_x={self.center_x}, center_y={self.center_y})")
    
    
class Hamlyn_Dataset(object):
    def __init__(
        self,
        datadir,
        mode='binocular'
    ):  
        # basic attrs
        self.img_wh = (640, 480)
        self.root_dir = datadir
        self.transform = transforms.ToTensor()
        self.mode = mode
        
        # dummy poses
        poses = np.eye(4).astype(np.float32)
        
        # intrinsics 
        intrinsics_matrix = np.loadtxt(os.path.join(datadir, 'intrinsics.txt'))
        intrinsics = Intrinsics(
            width=self.img_wh[0], height=self.img_wh[1], focal_x=intrinsics_matrix[0,0], focal_y=intrinsics_matrix[1,1], 
            center_x=intrinsics_matrix[0,2], center_y=intrinsics_matrix[1,2])
        intrinsics.center_x = 320-0.5
        intrinsics.center_y = 240-0.5
        
        # load imgs, masks, depths
        paths_img, paths_mask, paths_depth = [], [], []
        png_cnt = 0
        str_cnt = '/000000.png'
        while os.path.exists(datadir + "/images" + str_cnt):
            paths_img.append(datadir + "/images" + str_cnt)
            paths_mask.append(datadir + "/gt_masks" + str_cnt)
            paths_depth.append(datadir + "/gt_depth" + str_cnt)
            png_cnt += 1
            str_cnt = '/' + '0' * \
                (6-len(str(png_cnt))) + str(png_cnt) + '.png'
            
        paths_interp_img, paths_interp_mask = [], []
        interp_img_folder = "/images_interp"
        interp_mask_folder = "/gt_masks_interp"
        
        png_cnt = 0
        str_cnt = '/000000.png'
        while os.path.exists(datadir + interp_img_folder + str_cnt):
            paths_interp_img.append(datadir + interp_img_folder + str_cnt)
            paths_interp_mask.append(datadir + interp_mask_folder + str_cnt)
            png_cnt += 1
            str_cnt = '/' + '0' * \
                (6-len(str(png_cnt))) + str(png_cnt) + '.png'

        imgs, masks, depths = self.load_from_paths(paths_img, paths_mask, paths_depth)
        imgs_interp, masks_interp = self.load_from_paths_interp(paths_interp_img, paths_interp_mask)
        
        assert len(imgs) == len(masks) == len(depths), "the number of images should equal to the number of masks and depths"
        print(f"imgs, masks, and depths loaded, total num:{len(imgs)}")
        print(f"interp imgs loaded, total num:{len(imgs_interp)}")
        
        imgs, masks, depths, imgs_interp, masks_interp = \
            [torch.stack(lst, dim=0) for lst in [imgs, masks, depths, imgs_interp, masks_interp]]
        
        # Fruther process
        # crop the images, masks and depths
        # in hamlyn dataset, we crop the image from the left side, with 40 pixels
        crop_size = 40 # a setting for hamlyn dataset
        imgs = imgs[:, :, :, crop_size:].contiguous()
        masks = masks[:, :, :, crop_size:].contiguous()
        depths = depths[:, :, :, crop_size:].contiguous()
        imgs_interp = imgs_interp[:, :, :, crop_size:].contiguous()
        masks_interp = masks_interp[:, :, :, crop_size:].contiguous()
        
        intrinsics.width = intrinsics.width - crop_size
        intrinsics.center_x = intrinsics.center_x - crop_size / 2
        # normalize depth
        close_depth = percentile_torch(depths, 3.0)
        inf_depth = percentile_torch(depths, 99.9)
        depths[depths > inf_depth] = inf_depth
        depths[depths < close_depth] = close_depth
        
        self.znear = 0.1
        self.zfar = 1.1 * inf_depth
        
        # timestamps
        timestamps = np.arange(len(imgs)).astype(np.float32) / len(imgs)
        
        # train/test split
        idxs = np.arange(len(imgs))
        self.train_idxs = idxs[::2]
        self.test_idxs = idxs[1::2]
        self.video_idxs = idxs
        
        # self assignment
        self.imgs = imgs
        self.imgs_interp = imgs_interp
        self.masks = masks
        self.masks_interp = masks_interp    
        self.inf_depth = inf_depth
        self.close_depth = close_depth 
        self.depths = depths
        self.poses = np.repeat(poses[None], len(imgs), axis=0)
        self.intrinsics = intrinsics
        self.timestamps = timestamps
        self.maxtime = 1.0
        self.crop_size = crop_size
           
    def load_from_paths(self, paths_img, paths_mask, paths_depth):
        imgs, masks, depths = [], [], []
        
        for path_img, path_mask, path_depth in zip(paths_img, paths_mask, paths_depth):
            # images
            img = Image.open(path_img).convert('RGB')
            img = img.resize((self.img_wh[0], self.img_wh[1]), Image.LANCZOS)
            img = self.transform(img) # [C, H, W]
            # masks
            mask = Image.open(path_mask).convert('L')
            mask = mask.resize((self.img_wh[0], self.img_wh[1]), Image.LANCZOS)
            mask = ~ self.transform(mask).bool() # 0 for tool, 1 for tissue
            # depths
            depth = Image.open(path_depth)
            depth = depth.resize((self.img_wh[0], self.img_wh[1]), Image.LANCZOS)
            depth = np.array(depth)
            depth = torch.from_numpy(depth).float().unsqueeze(0)
            
            imgs.append(img)
            masks.append(mask)
            depths.append(depth)
        
        return imgs, masks, depths
    
    def load_from_paths_interp(self, paths_interp_img, paths_interp_mask):
        imgs, masks = [], []
        
        for path_img, path_mask in zip(paths_interp_img, paths_interp_mask):
            # images
            img = Image.open(path_img).convert('RGB')
            img = img.resize((self.img_wh[0], self.img_wh[1]), Image.LANCZOS)
            img = self.transform(img)
            # masks
            mask = Image.open(path_mask).convert('L')
            mask = mask.resize((self.img_wh[0], self.img_wh[1]), Image.LANCZOS)
            mask = ~ self.transform(mask).bool() # 0 for tool, 1 for tissue
            
            imgs.append(img)
            masks.append(mask)
        
        return imgs, masks
        
    def format_infos(self, split):
        cameras = []
        
        if split != 'interp' and split != 'interp2':
            if split == 'train': idxs = self.train_idxs
            elif split == 'test': idxs = self.test_idxs
            elif split == 'video':
                idxs = self.video_idxs
            else:
                raise ValueError(f"{split} has not been implemented.")
            
            intrinsics = self.intrinsics
            focal_x, focal_y = intrinsics.focal_x, intrinsics.focal_y
            fov_x, fov_y = focal2fov(focal_x, intrinsics.width), focal2fov(focal_y, intrinsics.height)
                
            for idx in idxs:
                image = self.imgs[idx]
                mask = self.masks[idx]
                depth = self.depths[idx]
                time = self.timestamps[idx]
                pose = self.poses[idx]
                R, T = pose[:3, :3], pose[:3, -1]
                R = R.transpose()
                cameras.append(Camera(colmap_id=idx,R=R,T=T,FoVx=fov_x,FoVy=fov_y,
                                    image=image, depth=depth, mask=mask, gt_alpha_mask=None,
                                    image_name=f"{idx}",uid=idx,data_device=torch.device("cuda"),time=time,
                                    Znear=self.znear, Zfar=self.zfar))
        else:
            intrinsics = self.intrinsics
            focal_x, focal_y = intrinsics.focal_x, intrinsics.focal_y
            fov_x, fov_y = focal2fov(focal_x, intrinsics.width), focal2fov(focal_y, intrinsics.height)
                
            for idx in range(len(self.imgs_interp)):
                image = self.imgs_interp[idx]
                pose = self.poses[0]
                R, T = pose[:3, :3], pose[:3, -1]
                R = R.transpose()
                
                interp_step = 2 if split == 'interp' else 4
                mask = self.masks_interp[idx]
                if idx % interp_step == 0:
                    # mask = self.masks[self.train_idxs[idx//interp_step]]
                    time = self.timestamps[self.train_idxs[idx//interp_step]]
                    depth = self.depths[self.train_idxs[idx//interp_step]]
                else:
                    # mask_lower = self.masks[self.train_idxs[idx//interp_step]]
                    # mask_upper = self.masks[self.train_idxs[idx//interp_step+1]]
                    # mask = torch.logical_and(mask_lower, mask_upper)
                    time_lower = self.timestamps[self.train_idxs[idx//interp_step]]
                    time_upper = self.timestamps[self.train_idxs[idx//interp_step+1]]
                    time = float(time_lower + (idx % interp_step) * (1 / interp_step) * (time_upper - time_lower))

                    depth = None
                
                cameras.append(Camera(colmap_id=idx,R=R,T=T,FoVx=fov_x,FoVy=fov_y,
                                      image=image, depth=depth, mask=mask, gt_alpha_mask=None,
                                    image_name=f"{idx}",uid=idx,data_device=torch.device("cuda"),time=time,
                                    Znear=self.znear, Zfar=self.zfar))
                        
        return cameras
    
    def get_init_pts(self):
        if self.mode == 'binocular':
            
            initial_idx = self.train_idxs[0]
            color, depth, mask = self.imgs[initial_idx].numpy(), self.depths[initial_idx].numpy(), self.masks[initial_idx].numpy()
            pts, colors, _ = self.get_pts_cam(depth, mask, color)
            pts = self.get_pts_wld(pts, self.poses[initial_idx])
            idxs = np.random.choice(np.arange(pts.shape[0]), 30000, replace=False)
            pts = pts[idxs, :]
            colors = colors[idxs, :]
            normals = np.zeros((pts.shape[0], 3))
        else:
            raise ValueError(f"{self.mode} has not been implemented.")
        return pts, colors, normals
    
    def get_pts_cam(self, depth, mask, color):
        # import pdb; pdb.set_trace()
        W, H = self.intrinsics.width, self.intrinsics.height
        i, j = np.meshgrid(np.linspace(0, W-1, W), np.linspace(0, H-1, H))
        X_Z = (i-W/2) / self.intrinsics.focal_x
        Y_Z = (j-H/2) / self.intrinsics.focal_y
        Z = depth
        X, Y = X_Z * Z, Y_Z * Z
        pts_cam = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)
        color = color.reshape(-1, 3)
        mask = mask.transpose(2, 0, 1).reshape(-1).astype(bool)
        valid_region = (np.abs(pts_cam).sum(axis=-1)!=0)
        mask = np.logical_and(mask, valid_region)
        pts_valid = pts_cam[mask, :]
        color_valid = color[mask, :]

        return pts_valid, color_valid, mask
    
    def get_pts_wld(self, pts, pose):
        R, T = pose[:3, :3], pose[:3, -1]
        R = R.transpose()
        w2c = np.concatenate((R, T[...,None]), axis=-1)
        w2c = np.concatenate((w2c, np.array([[0, 0, 0, 1]])), axis=0)
        c2w = np.linalg.inv(w2c)
        pts_cam_homo = np.concatenate((pts, np.ones((pts.shape[0], 1))), axis=-1)
        pts_wld = np.transpose(c2w @ np.transpose(pts_cam_homo))
        pts_wld = pts_wld[:, :3]
        return pts_wld